#!/usr/bin/env python
"""Evaluate truth-matched resolved and boosted Higgs mass fidelity on ggHbb."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VQTORCH_ROOT = PROJECT_ROOT / "vqtorch"
for import_root in (VQTORCH_ROOT, PROJECT_ROOT):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import awkward as ak
import fastjet
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.optimize import curve_fit
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from gabbro.data.orbit_parquet import SEQUENCE_SCHEMAS, OrbitParquetDataset
from gabbro.plotting.utils import set_mpl_style


TRUTH_COLUMNS = [
    "Gen_Part_PT",
    "Gen_Part_Eta",
    "Gen_Part_Phi",
    "Gen_Part_Mass",
    "Gen_Part_PID",
    "Gen_Part_D1",
    "Gen_Part_D2",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gghbb-test-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bootstrap-replicas", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(run_dir: Path, device):
    config_path = run_dir / "config_resolved.yaml"
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    cfg = OmegaConf.load(config_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec
    model = instantiate(cfg.model)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), cfg, checkpoint


def checkpoint_data_settings(cfg) -> tuple[str, int, str | None, float]:
    sequence_type = str(cfg.data.get("sequence_type", "particle"))
    if sequence_type not in SEQUENCE_SCHEMAS:
        raise ValueError(f"Unsupported checkpoint sequence type: {sequence_type!r}")
    configured_length = cfg.data.get("max_sequence_length")
    max_sequence_length = (
        int(configured_length)
        if configured_length is not None
        else int(SEQUENCE_SCHEMAS[sequence_type]["max_sequence_length"])
    )
    schema = SEQUENCE_SCHEMAS[sequence_type]
    configured_mask_column = cfg.data.get("mask_column")
    mask_column = (
        schema["mask_column"]
        if configured_mask_column is None
        else str(configured_mask_column)
    )
    configured_mask_minimum = cfg.data.get("mask_min_value")
    mask_min_value = (
        float(schema["mask_min_value"])
        if configured_mask_minimum is None
        else float(configured_mask_minimum)
    )
    return sequence_type, max_sequence_length, mask_column, mask_min_value


def decode_batch(model, batch, device):
    features = batch["part_features"].to(device)
    mask = batch["part_mask"].to(device)
    pid = batch.get("part_pid")
    if pid is not None:
        pid = pid.to(device)
    with torch.inference_mode():
        if hasattr(model, "_quantize_batch"):
            reconstructed, _, _ = model._quantize_batch(features, mask, pid)
        else:
            reconstructed, _ = model.forward(features, mask, pid_particle=pid)
    values = reconstructed.detach().cpu().numpy()
    return np.stack(
        [
            values[..., 0] * 3.0,
            np.arctan2(values[..., 2], values[..., 1]),
            np.clip(np.exp(values[..., 3] + 1.8) - 1e-8, 0.0, None),
        ],
        axis=-1,
    )


def prepare_test_time_baseline(model, cfg, output_dir: Path):
    if not (hasattr(model, "_fit_faiss") or getattr(model, "binning", None) == "learned"):
        return
    data_module = instantiate(cfg.data)
    model._trainer = SimpleNamespace(
        datamodule=data_module,
        is_global_zero=True,
        default_root_dir=str(output_dir),
        loggers=[],
    )
    if hasattr(model, "save_codebook"):
        model.save_codebook = False
    if hasattr(model, "upload_codebook_to_wandb"):
        model.upload_codebook_to_wandb = False
    if getattr(model, "binning", None) == "learned":
        model._fit_learned_bins()
    if hasattr(model, "_fit_faiss"):
        model._fit_faiss()


class TruthReader:
    """Read truth rows while caching only the currently used parquet row group."""

    def __init__(self):
        self._layouts = {}
        self._cache_key = None
        self._cache = None

    def row(self, path: str, row_index: int):
        if path not in self._layouts:
            parquet = pq.ParquetFile(path)
            offsets = [0]
            for index in range(parquet.num_row_groups):
                offsets.append(offsets[-1] + parquet.metadata.row_group(index).num_rows)
            self._layouts[path] = (parquet, offsets)
        parquet, offsets = self._layouts[path]
        row_group = bisect.bisect_right(offsets, row_index) - 1
        cache_key = (path, row_group)
        if cache_key != self._cache_key:
            self._cache = ak.from_arrow(parquet.read_row_group(row_group, columns=TRUTH_COLUMNS))
            self._cache_key = cache_key
        return self._cache[row_index - offsets[row_group]]


def decaying_higgs(truth):
    pids = np.asarray(ak.to_numpy(truth.Gen_Part_PID), dtype=np.int64)
    d1s = np.asarray(ak.to_numpy(truth.Gen_Part_D1), dtype=np.int64)
    d2s = np.asarray(ak.to_numpy(truth.Gen_Part_D2), dtype=np.int64)
    for higgs_index in np.flatnonzero(np.abs(pids) == 25)[::-1]:
        d1, d2 = int(d1s[higgs_index]), int(d2s[higgs_index])
        if 0 <= d1 < len(pids) and 0 <= d2 < len(pids) and {int(pids[d1]), int(pids[d2])} == {5, -5}:
            def particle(index):
                return np.array(
                    [
                        float(truth.Gen_Part_Eta[index]),
                        float(truth.Gen_Part_Phi[index]),
                        float(truth.Gen_Part_PT[index]),
                        float(truth.Gen_Part_Mass[index]),
                    ]
                )

            return particle(higgs_index), particle(d1), particle(d2)
    return None


def delta_r(eta_a, phi_a, eta_b, phi_b):
    delta_phi = np.remainder(phi_a - phi_b + np.pi, 2 * np.pi) - np.pi
    return np.hypot(eta_a - eta_b, delta_phi)


def cluster_jets(particles: np.ndarray, radius: float, min_pt: float):
    if not len(particles):
        return np.empty((0, 4))
    momenta = ak.zip(
        {
            "pt": [particles[:, 2]],
            "eta": [particles[:, 0]],
            "phi": [particles[:, 1]],
            "mass": [np.zeros(len(particles))],
        },
        with_name="Momentum4D",
    )
    jets = fastjet.ClusterSequence(
        momenta, fastjet.JetDefinition(fastjet.antikt_algorithm, radius)
    ).inclusive_jets(min_pt=min_pt)[0]
    if not len(jets):
        return np.empty((0, 4))
    values = np.stack(
        [ak.to_numpy(jets.eta), ak.to_numpy(jets.phi), ak.to_numpy(jets.pt), ak.to_numpy(jets.mass)],
        axis=-1,
    )
    return values[np.abs(values[:, 0]) < 2.5]


def combine_mass(first, second):
    def components(jet):
        eta, phi, pt, mass = jet
        px, py = pt * np.cos(phi), pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
        return np.array([energy, px, py, pz])

    total = components(first) + components(second)
    return float(np.sqrt(max(total[0] ** 2 - np.dot(total[1:], total[1:]), 0.0)))


def resolved_candidate(particles, b, bbar):
    jets = cluster_jets(particles, radius=0.4, min_pt=30.0)
    if len(jets) < 2:
        return None
    distances = np.array(
        [[delta_r(parton[0], parton[1], jet[0], jet[1]) for jet in jets] for parton in (b, bbar)]
    )
    parton_indices, jet_indices = linear_sum_assignment(distances)
    if len(jet_indices) != 2 or np.any(distances[parton_indices, jet_indices] >= 0.2):
        return None
    return combine_mass(jets[jet_indices[0]], jets[jet_indices[1]])


def boosted_candidate(particles, higgs, b, bbar):
    jets = cluster_jets(particles, radius=0.8, min_pt=250.0)
    if not len(jets):
        return None
    distances = np.array([delta_r(higgs[0], higgs[1], jet[0], jet[1]) for jet in jets])
    jet = jets[int(np.argmin(distances))]
    if np.min(distances) >= 0.4:
        return None
    if delta_r(b[0], b[1], jet[0], jet[1]) >= 0.8:
        return None
    if delta_r(bbar[0], bbar[1], jet[0], jet[1]) >= 0.8:
        return None
    return float(jet[3])


def double_crystal_ball(x, norm, mean, sigma, alpha_l, n_l, alpha_r, n_r):
    t = (x - mean) / sigma
    result = np.exp(-0.5 * t**2)
    left = t < -alpha_l
    right = t > alpha_r
    a_l = (n_l / alpha_l) ** n_l * np.exp(-0.5 * alpha_l**2)
    b_l = n_l / alpha_l - alpha_l
    a_r = (n_r / alpha_r) ** n_r * np.exp(-0.5 * alpha_r**2)
    b_r = n_r / alpha_r - alpha_r
    result[left] = a_l * (b_l - t[left]) ** (-n_l)
    result[right] = a_r * (b_r + t[right]) ** (-n_r)
    return norm * result


def fit_mass(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[(values >= 40.0) & (values <= 200.0)]
    counts, edges = np.histogram(values, bins=np.arange(40.0, 202.0, 2.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(values) < 50:
        return {"success": False, "events": int(len(values)), "mean": None, "sigma": None}
    initial = [max(counts.max(), 1), np.median(values), max(np.std(values), 5.0), 1.5, 3, 1.5, 3]
    try:
        parameters, _ = curve_fit(
            double_crystal_ball,
            centers,
            counts,
            p0=initial,
            bounds=([0, 80, 1, 0.2, 1.01, 0.2, 1.01], [np.inf, 170, 80, 8, 30, 8, 30]),
            maxfev=50_000,
        )
        return {
            "success": True,
            "events": int(len(values)),
            "mean": float(parameters[1]),
            "sigma": float(parameters[2]),
            "parameters": list(map(float, parameters)),
        }
    except (RuntimeError, ValueError):
        return {"success": False, "events": int(len(values)), "mean": None, "sigma": None}


def bootstrap_fit(values, replicas):
    if len(values) < 50 or replicas < 1:
        return {
            "successful_replicas": 0,
            "mean_uncertainty": None,
            "sigma_uncertainty": None,
        }
    rng = np.random.default_rng(12345)
    fits = [fit_mass(rng.choice(values, size=len(values), replace=True)) for _ in range(replicas)]
    means = [fit["mean"] for fit in fits if fit["success"]]
    sigmas = [fit["sigma"] for fit in fits if fit["success"]]
    return {
        "successful_replicas": len(means),
        "mean_uncertainty": float(np.std(means, ddof=1)) if len(means) > 1 else None,
        "sigma_uncertainty": float(np.std(sigmas, ddof=1)) if len(sigmas) > 1 else None,
    }


def summarize_topology(rows, topology, replicas):
    original = np.array([row[f"{topology}_original"] for row in rows], dtype=float)
    decoded = np.array([row[f"{topology}_decoded"] for row in rows], dtype=float)
    common = np.isfinite(original) & np.isfinite(decoded)
    original_common, decoded_common = original[common], decoded[common]
    original_fit, decoded_fit = fit_mass(original_common), fit_mass(decoded_common)
    if len(original_common):
        paired_mean = float(np.mean(decoded_common - original_common))
        paired_std = float(np.std(decoded_common - original_common))
        original_width = float(np.diff(np.quantile(original_common, [0.16, 0.84]))[0] / 2)
        decoded_width = float(np.diff(np.quantile(decoded_common, [0.16, 0.84]))[0] / 2)
    else:
        paired_mean = paired_std = original_width = decoded_width = None
    summary = {
        "events": len(rows),
        "original_efficiency": float(np.mean(np.isfinite(original))),
        "decoded_efficiency": float(np.mean(np.isfinite(decoded))),
        "common_events": int(common.sum()),
        "original_fit": original_fit | bootstrap_fit(original_common, replicas),
        "decoded_fit": decoded_fit | bootstrap_fit(decoded_common, replicas),
        "paired_mass_difference_mean": paired_mean,
        "paired_mass_difference_std": paired_std,
        "original_central68_half_width": original_width,
        "decoded_central68_half_width": decoded_width,
    }
    if original_fit["success"] and decoded_fit["success"]:
        summary["fit_peak_shift"] = decoded_fit["mean"] - original_fit["mean"]
        summary["fit_width_ratio"] = decoded_fit["sigma"] / original_fit["sigma"]
    else:
        summary["fit_peak_shift"] = None
        summary["fit_width_ratio"] = None
    pt = np.asarray([row["truth_higgs_pt"] for row in rows], dtype=float)
    summary["truth_higgs_pt_bins"] = {}
    for low, high in zip((0, 250, 350, 500, 750), (250, 350, 500, 750, np.inf)):
        selected = (pt >= low) & (pt < high)
        selected_common = selected & common
        name = f"{low:g}_{high:g}" if np.isfinite(high) else f"{low:g}_inf"
        summary["truth_higgs_pt_bins"][name] = {
            "events": int(selected.sum()),
            "original_efficiency": float(np.mean(np.isfinite(original[selected])))
            if np.any(selected)
            else None,
            "decoded_efficiency": float(np.mean(np.isfinite(decoded[selected])))
            if np.any(selected)
            else None,
            "common_events": int(selected_common.sum()),
            "paired_mass_difference_mean": float(
                np.mean(decoded[selected_common] - original[selected_common])
            )
            if np.any(selected_common)
            else None,
        }
    return summary


def plot_masses(rows, topology, output_dir):
    set_mpl_style()
    original = np.array([row[f"{topology}_original"] for row in rows], dtype=float)
    decoded = np.array([row[f"{topology}_decoded"] for row in rows], dtype=float)
    common = np.isfinite(original) & np.isfinite(decoded)
    figure, axis = plt.subplots(figsize=(7, 5))
    bins = np.arange(40, 202, 2)
    axis.hist(original[common], bins=bins, histtype="step", density=True, label="Original")
    axis.hist(decoded[common], bins=bins, histtype="step", density=True, label="Decoded")
    axis.set(xlabel="Higgs candidate mass [GeV]", ylabel="Normalized events", title=topology)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"higgs_mass_{topology}.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, checkpoint = load_model(args.run_dir.resolve(), device)
    sequence_type, max_sequence_length, mask_column, mask_min_value = (
        checkpoint_data_settings(cfg)
    )
    batch_size = args.batch_size or (16 if max_sequence_length > 128 else 256)
    prepare_test_time_baseline(model, cfg, args.output_dir)
    dataset = OrbitParquetDataset(
        args.gghbb_test_manifest,
        sequence_type=sequence_type,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        mask_column=mask_column,
        mask_min_value=mask_min_value,
        return_raw_features=True,
        return_event_metadata=True,
        pid_cfg=cfg.get("pid"),
    )
    truth_reader = TruthReader()
    rows = []
    with tqdm(total=args.events, desc="Evaluating ggHbb", unit="event") as progress:
        for batch in dataset:
            decoded = decode_batch(model, batch, device)
            mask = batch["part_mask"].numpy().astype(bool)
            for index in range(mask.shape[0]):
                source_file = str(batch["source_files"][index])
                source_row = int(batch["source_rows"][index])
                truth = decaying_higgs(truth_reader.row(source_file, source_row))
                if truth is None:
                    continue
                higgs, b, bbar = truth
                raw = np.asarray(ak.to_numpy(batch["raw_part_features"][index]), dtype=float)
                original = raw[:, [1, 2, 0]]
                decoded_event = decoded[index, mask[index]]
                rows.append(
                    {
                        "source_file": source_file,
                        "source_row": source_row,
                        "truth_higgs_pt": float(higgs[2]),
                        "resolved_original": resolved_candidate(original, b, bbar),
                        "resolved_decoded": resolved_candidate(decoded_event, b, bbar),
                        "boosted_original": boosted_candidate(original, higgs, b, bbar),
                        "boosted_decoded": boosted_candidate(decoded_event, higgs, b, bbar),
                    }
                )
                progress.update(1)
                if len(rows) == args.events:
                    break
            if len(rows) == args.events:
                break
    if len(rows) != args.events:
        raise RuntimeError(f"Expected {args.events} truth-selected events, got {len(rows)}")
    for row in rows:
        for key in ("resolved_original", "resolved_decoded", "boosted_original", "boosted_decoded"):
            if row[key] is None:
                row[key] = float("nan")
    with (args.output_dir / "higgs_candidates.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(checkpoint),
        "events": args.events,
        "sequence_type": sequence_type,
        "max_sequence_length": max_sequence_length,
        "particle_selection": {
            "mask_column": mask_column,
            "mask_min_value": mask_min_value,
            "weighted_pt": False,
        },
        "clustering": {
            "algorithm": "antikt",
            "resolved": {"radius": 0.4, "min_pt": 30.0, "max_abs_eta": 2.5, "match_dr": 0.2},
            "boosted": {"radius": 0.8, "min_pt": 250.0, "max_abs_eta": 2.5, "match_dr": 0.4},
        },
        "resolved": summarize_topology(rows, "resolved", args.bootstrap_replicas),
        "boosted": summarize_topology(rows, "boosted", args.bootstrap_replicas),
    }
    (args.output_dir / "higgs_mass_metrics.json").write_text(json.dumps(summary, indent=2))
    plot_masses(rows, "resolved", args.output_dir)
    plot_masses(rows, "boosted", args.output_dir)
    print(f"Wrote Higgs mass evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
