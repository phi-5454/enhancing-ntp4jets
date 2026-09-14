#!/usr/bin/env python3
"""Measure lepton-to-hadron ratios in the three canonical tt samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PID_NAMES = (
    "neutral_hadron",
    "photon",
    "negative_hadron",
    "positive_hadron",
    "electron",
    "positron",
    "muon",
    "antimuon",
)
CHANNEL_SAMPLES = {
    "hadronic": "tt0123j_5f_ckm_LO_MLM_hadronic",
    "leptonic": "tt0123j_5f_ckm_LO_MLM_leptonic",
    "semileptonic": "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
}
PDG_COLUMN = "L1T_PUPPIPart_PID"
CHARGE_COLUMN = "L1T_PUPPIPart_Charge"
PUPPIW_COLUMN = "L1T_PUPPIPart_PuppiW"
PT_COLUMN = "L1T_PUPPIPart_PT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("train_val", "test"),
        default="test",
        help="Manifest split to sample (default: test).",
    )
    parser.add_argument(
        "--events-per-channel",
        type=int,
        default=2_000,
        help="Equal event count sampled from each tt decay channel (default: 2000).",
    )
    parser.add_argument("--puppiw-min", type=float, default=0.05)
    parser.add_argument("--max-particles", type=int, default=128)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def manifest_files(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    files = []
    for raw_line in path.read_text().splitlines():
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        entry = Path(value).expanduser()
        files.append(entry if entry.is_absolute() else path.parent / entry)
    if not files:
        raise ValueError(f"Manifest contains no files: {path}")
    return files


def pid_classes(pdg_id: np.ndarray, charge: np.ndarray) -> np.ndarray:
    """Match ``gabbro.data.orbit_parquet.map_pdg_charge_to_pid_class``."""
    classes = np.where(charge < 0, 2, np.where(charge > 0, 3, 0))
    classes = np.where(np.abs(pdg_id) == 22, 1, classes)
    classes = np.where(pdg_id == 11, 4, classes)
    classes = np.where(pdg_id == -11, 5, classes)
    classes = np.where(pdg_id == 13, 6, classes)
    return np.where(pdg_id == -13, 7, classes).astype(np.int64)


def empty_stage() -> dict:
    return {
        "class_counts": Counter(),
        "event_leptons": [],
        "event_hadrons": [],
        "event_photons": [],
    }


def add_event(stage: dict, classes: np.ndarray) -> None:
    stage["class_counts"].update(classes.tolist())
    stage["event_leptons"].append(int(np.count_nonzero(classes >= 4)))
    stage["event_hadrons"].append(int(np.count_nonzero(np.isin(classes, (0, 2, 3)))))
    stage["event_photons"].append(int(np.count_nonzero(classes == 1)))


def sample_channel(
    files: list[Path], events: int, puppiw_min: float, max_particles: int
) -> dict:
    stages = {name: empty_stage() for name in ("raw", "puppiw", "model_input")}
    events_read = 0
    columns = [PDG_COLUMN, CHARGE_COLUMN, PUPPIW_COLUMN, PT_COLUMN]
    for path in files:
        parquet_file = pq.ParquetFile(path)
        missing = set(columns) - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for batch in parquet_file.iter_batches(columns=columns, batch_size=2_048):
            pdg_events = batch.column(PDG_COLUMN).to_pylist()
            charge_events = batch.column(CHARGE_COLUMN).to_pylist()
            puppiw_events = batch.column(PUPPIW_COLUMN).to_pylist()
            pt_events = batch.column(PT_COLUMN).to_pylist()
            for pdg_values, charge_values, puppiw_values, pt_values in zip(
                pdg_events, charge_events, puppiw_events, pt_events
            ):
                pdg = np.asarray(pdg_values, dtype=np.int64)
                charge = np.asarray(charge_values, dtype=np.float64)
                puppiw = np.asarray(puppiw_values, dtype=np.float64)
                pt = np.asarray(pt_values, dtype=np.float64)
                classes = pid_classes(pdg, charge)
                physical = np.isfinite(pt) & (pt > 0.0)
                add_event(stages["raw"], classes[physical])
                selected = classes[physical & np.isfinite(puppiw) & (puppiw > puppiw_min)]
                add_event(stages["puppiw"], selected)
                add_event(stages["model_input"], selected[:max_particles])
                events_read += 1
                if events_read >= events:
                    return {"events": events_read, "stages": stages}
    return {"events": events_read, "stages": stages}


def merge_channels(channels: dict[str, dict]) -> dict:
    merged = {"events": sum(channel["events"] for channel in channels.values()), "stages": {}}
    for stage_name in ("raw", "puppiw", "model_input"):
        stage = empty_stage()
        for channel in channels.values():
            source = channel["stages"][stage_name]
            stage["class_counts"].update(source["class_counts"])
            for key in ("event_leptons", "event_hadrons", "event_photons"):
                stage[key].extend(source[key])
        merged["stages"][stage_name] = stage
    return merged


def summarize_stage(stage: dict) -> dict:
    counts = stage["class_counts"]
    leptons = sum(counts[index] for index in range(4, 8))
    hadrons = sum(counts[index] for index in (0, 2, 3))
    photons = counts[1]
    total = leptons + hadrons + photons
    per_event = {}
    for name in ("leptons", "hadrons", "photons"):
        values = np.asarray(stage[f"event_{name}"], dtype=np.float64)
        per_event[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }
    return {
        "particles": total,
        "class_counts": {PID_NAMES[index]: int(counts[index]) for index in range(8)},
        "leptons": leptons,
        "hadrons": hadrons,
        "photons": photons,
        "lepton_to_hadron": leptons / hadrons if hadrons else None,
        "lepton_to_hadron_plus_photon": leptons / (hadrons + photons)
        if hadrons + photons
        else None,
        "lepton_fraction": leptons / total if total else None,
        "per_event": per_event,
    }


def print_summary(results: dict[str, dict]) -> None:
    header = (
        f"{'sample':<18} {'stage':<12} {'events':>7} {'particles':>10} "
        f"{'leptons':>9} {'hadrons':>9} {'photons':>9} {'L/H':>9} {'L/(H+gamma)':>12}"
    )
    print(header)
    print("-" * len(header))
    for sample_name, sample in results.items():
        for stage_name, stage in sample["stages"].items():
            print(
                f"{sample_name:<18} {stage_name:<12} {sample['events']:7d} "
                f"{stage['particles']:10d} {stage['leptons']:9d} {stage['hadrons']:9d} "
                f"{stage['photons']:9d} {stage['lepton_to_hadron']:9.5f} "
                f"{stage['lepton_to_hadron_plus_photon']:12.5f}"
            )


def main() -> None:
    args = parse_args()
    if args.events_per_channel < 1 or args.max_particles < 1:
        raise ValueError("--events-per-channel and --max-particles must be positive")
    manifest_dir = args.manifest_dir.expanduser().resolve()
    sampled = {
        channel: sample_channel(
            manifest_files(manifest_dir / f"{sample_name}_{args.split}.txt"),
            args.events_per_channel,
            args.puppiw_min,
            args.max_particles,
        )
        for channel, sample_name in CHANNEL_SAMPLES.items()
    }
    if any(result["events"] != args.events_per_channel for result in sampled.values()):
        raise RuntimeError("At least one channel contained fewer events than requested")
    sampled["equal_tt_mix"] = merge_channels(sampled)
    results = {
        name: {
            "events": sample["events"],
            "stages": {
                stage_name: summarize_stage(stage)
                for stage_name, stage in sample["stages"].items()
            },
        }
        for name, sample in sampled.items()
    }
    print_summary(results)
    payload = {
        "split": args.split,
        "puppiw_min": args.puppiw_min,
        "max_particles": args.max_particles,
        "events_per_channel": args.events_per_channel,
        "results": results,
    }
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
