#!/usr/bin/env python
"""Evaluate resolved and boosted ggHbb masses using raw top-pT particles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import vector

from gabbro.plotting.utils import set_mpl_style
import evaluate_orbit_higgs_mass_raw as raw


vector.register_awkward()
PARTICLE_COLUMNS = raw.PARTICLE_COLUMNS
TRUTH_COLUMNS = raw.TRUTH_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gghbb-test-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--top-pt-particles", type=int, default=500)
    return parser.parse_args()


def truth_higgs_and_bs(event):
    pids = np.asarray(ak.to_numpy(event.Gen_Part_PID), dtype=np.int64)
    daughter_1 = np.asarray(ak.to_numpy(event.Gen_Part_D1), dtype=np.int64)
    daughter_2 = np.asarray(ak.to_numpy(event.Gen_Part_D2), dtype=np.int64)
    for higgs_index in np.flatnonzero(np.abs(pids) == 25)[::-1]:
        first, second = int(daughter_1[higgs_index]), int(daughter_2[higgs_index])
        if not (0 <= first < len(pids) and 0 <= second < len(pids)):
            continue
        if {int(pids[first]), int(pids[second])} != {5, -5}:
            continue

        def four_vector(index):
            return np.array(
                [
                    float(event.Gen_Part_Eta[index]),
                    float(event.Gen_Part_Phi[index]),
                    float(event.Gen_Part_PT[index]),
                    float(event.Gen_Part_Mass[index]),
                ]
            )

        return four_vector(higgs_index), four_vector(first), four_vector(second)
    return None


def boosted_candidate(particles, higgs):
    jets = raw.cluster_jets(particles, radius=0.8, min_pt=250.0)
    if not len(jets):
        return None
    distances = np.array([raw.delta_r(higgs[0], higgs[1], jet[0], jet[1]) for jet in jets])
    index = int(np.argmin(distances))
    return float(jets[index, 3]) if distances[index] < 0.4 else None


def topology_summary(values: np.ndarray, events: int) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "efficiency": float(len(finite) / events),
        "candidates": int(len(finite)),
        "mean": float(np.mean(finite)) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
    }


def plot_masses(values: np.ndarray, topology: str, output_dir: Path, top_pt_particles: int):
    finite = values[np.isfinite(values)]
    set_mpl_style()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(
        finite,
        bins=np.arange(40, 202, 2),
        histtype="step",
        density=True,
        label=f"Raw top-{top_pt_particles} pT particles",
    )
    axis.axvline(125.0, color="black", linestyle="--", linewidth=1, label="Higgs mass")
    axis.set(
        xlabel="Higgs candidate mass [GeV]",
        ylabel="Normalized events",
        title=f"{topology.title()} H to bb",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"higgs_mass_{topology}_raw_top_pt.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if args.events < 1 or args.top_pt_particles < 1:
        raise ValueError("--events and --top-pt-particles must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(line) for line in args.gghbb_test_manifest.read_text().splitlines() if line.strip()]
    rows = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            columns=PARTICLE_COLUMNS + TRUTH_COLUMNS, batch_size=512
        ):
            for event in ak.from_arrow(batch):
                truth = truth_higgs_and_bs(event)
                if truth is None:
                    continue
                higgs, b, bbar = truth
                particles = raw.top_pt_particles(event, args.top_pt_particles)
                rows.append(
                    {
                        "resolved_mass": raw.resolved_candidate(
                            particles, b, bbar, 0.4, 30.0, 0.2
                        ),
                        "boosted_mass": boosted_candidate(particles, higgs),
                    }
                )
                if len(rows) == args.events:
                    break
            if len(rows) == args.events:
                break
        if len(rows) == args.events:
            break
    if len(rows) != args.events:
        raise RuntimeError(f"Expected {args.events} H -> bb events, got {len(rows)}")

    resolved = np.asarray([np.nan if row["resolved_mass"] is None else row["resolved_mass"] for row in rows])
    boosted = np.asarray([np.nan if row["boosted_mass"] is None else row["boosted_mass"] for row in rows])
    summary = {
        "events": len(rows),
        "selection": {"raw_particles": True, "top_pt_particles": args.top_pt_particles},
        "resolved": topology_summary(resolved, len(rows)),
        "boosted": topology_summary(boosted, len(rows)),
    }
    (args.output_dir / "higgs_mass_raw_top_pt_metrics.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "higgs_mass_raw_top_pt_candidates.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("resolved_mass", "boosted_mass"))
        writer.writeheader()
        writer.writerows(rows)
    plot_masses(resolved, "resolved", args.output_dir, args.top_pt_particles)
    plot_masses(boosted, "boosted", args.output_dir, args.top_pt_particles)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
