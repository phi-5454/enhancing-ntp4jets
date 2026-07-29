#!/usr/bin/env python
"""Evaluate ggHbb masses after PUPPI filtering and four-vector weighting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

import evaluate_orbit_higgs_mass_puppiw128_ak8 as legacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gghbb-test-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--puppi-weight-min", type=float, default=0.05)
    parser.add_argument("--max-particles", type=int, default=128)
    parser.add_argument("--ak8-min-pt", type=float, default=250.0)
    return parser.parse_args()


def weighted_legacy_particles(event, puppi_weight_min: float, max_particles: int) -> np.ndarray:
    pt = np.asarray(ak.to_numpy(event.L1T_PUPPIPart_PT), dtype=np.float32)
    eta = np.asarray(ak.to_numpy(event.L1T_PUPPIPart_Eta), dtype=np.float32)
    phi = np.asarray(ak.to_numpy(event.L1T_PUPPIPart_Phi), dtype=np.float32)
    weight = np.asarray(ak.to_numpy(event.L1T_PUPPIPart_PuppiW), dtype=np.float32)
    selected = (
        np.isfinite(pt)
        & np.isfinite(eta)
        & np.isfinite(phi)
        & np.isfinite(weight)
        & (pt > 0.0)
        & (weight > puppi_weight_min)
    )
    return np.stack([eta[selected], phi[selected], pt[selected] * weight[selected]], axis=-1)[:max_particles]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if args.events < 1 or args.max_particles < 1:
        raise ValueError("--events and --max-particles must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = legacy.PARTICLE_COLUMNS + (legacy.AK8_FILTER_COLUMN,) + legacy.control.TRUTH_COLUMNS
    paths = [Path(line) for line in args.gghbb_test_manifest.read_text().splitlines() if line.strip()]
    rows = []
    scanned_events = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=512):
            for event in ak.from_arrow(batch):
                scanned_events += 1
                if not legacy.passes_ak8_filter(event, args.ak8_min_pt):
                    continue
                truth = legacy.control.truth_higgs_and_bs(event)
                if truth is None:
                    continue
                higgs, b, bbar = truth
                particles = weighted_legacy_particles(event, args.puppi_weight_min, args.max_particles)
                rows.append(
                    {
                        "selected_particles": len(particles),
                        "resolved_mass": legacy.control.raw.resolved_candidate(
                            particles, b, bbar, 0.4, 30.0, 0.2
                        ),
                        "boosted_mass": legacy.control.boosted_candidate(particles, higgs),
                    }
                )
                if len(rows) == args.events:
                    break
            if len(rows) == args.events:
                break
        if len(rows) == args.events:
            break
    if len(rows) != args.events:
        raise RuntimeError(f"Expected {args.events} selected H -> bb events, got {len(rows)}")

    resolved = np.asarray([np.nan if row["resolved_mass"] is None else row["resolved_mass"] for row in rows])
    boosted = np.asarray([np.nan if row["boosted_mass"] is None else row["boosted_mass"] for row in rows])
    selected_counts = np.asarray([row["selected_particles"] for row in rows])
    label = f"PuppiW-weighted, PuppiW > {args.puppi_weight_min:g}, first {args.max_particles}"
    summary = {
        "events": len(rows),
        "scanned_events": scanned_events,
        "selection": {
            "puppi_weight_min": args.puppi_weight_min,
            "max_particles": args.max_particles,
            "ak8_min_pt": args.ak8_min_pt,
            "four_momenta_weighted_by_puppiw": True,
            "selected_particles_median": float(np.median(selected_counts)),
        },
        "resolved": legacy.control.topology_summary(resolved, len(rows)),
        "boosted": legacy.control.topology_summary(boosted, len(rows)),
    }
    (args.output_dir / "higgs_mass_puppiw128_weighted_ak8_metrics.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "higgs_mass_puppiw128_weighted_ak8_candidates.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("selected_particles", "resolved_mass", "boosted_mass"))
        writer.writeheader()
        writer.writerows(rows)
    legacy.plot_mass(resolved, "resolved", args.output_dir, label)
    legacy.plot_mass(boosted, "boosted", args.output_dir, label)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
