#!/usr/bin/env python
"""Compare original-data AK4 jet multiplicities between ORBIT samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT / "vqtorch", PROJECT_ROOT):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from gabbro.plotting.orbit import multirun_color, multirun_marker
from scripts.compare_orbit_jet_counts import jet_counts
from scripts.compare_orbit_physical_particle_kinematics import evaluation_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        nargs=3,
        action="append",
        required=True,
        metavar=("RUN_DIR", "SUITE", "LABEL"),
        help="Resolved data config, test suite, and display label; repeat per sample.",
    )
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--jet-radius", type=float, default=0.4)
    parser.add_argument("--jet-min-pt", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sample_jet_counts(
    run_dir: Path,
    suite: str,
    events: int,
    num_workers: int,
    jet_radius: float,
    jet_min_pt: float,
) -> np.ndarray:
    counts = []
    loader = evaluation_loader(run_dir, suite, events, num_workers)
    with tqdm(total=events, desc=f"Clustering {suite}", unit="event") as bar:
        for batch in loader:
            features = batch["part_features"].detach().cpu().numpy()
            mask = batch["part_mask"].detach().cpu().numpy().astype(bool)
            counts.append(jet_counts(features, mask, jet_radius, jet_min_pt))
            bar.update(len(mask))
    return np.concatenate(counts)[:events]


def sample_style(label: str) -> str:
    """Use the established training-sample styles for the two canonical samples."""
    normalized = label.lower().replace("$", "").replace("\\", "").replace(" ", "_")
    if normalized in {"tt", "ttbar", "tbar_t", "t_tbar"}:
        return "TT_only"
    if normalized in {"sm_mixture", "standard_mixture"}:
        return "SM_mixture"
    return label


def main() -> None:
    args = parse_args()
    if args.events < 1 or args.jet_radius <= 0 or args.jet_min_pt < 0:
        raise ValueError("Events and jet radius must be positive; jet pT must be non-negative")

    records = []
    for run_dir_string, suite, label in args.sample:
        run_dir = Path(run_dir_string).resolve()
        values = sample_jet_counts(
            run_dir,
            suite,
            args.events,
            args.num_workers,
            args.jet_radius,
            args.jet_min_pt,
        )
        records.append(
            {
                "label": label,
                "style": sample_style(label),
                "run_dir": str(run_dir),
                "suite": suite,
                "values": values,
            }
        )

    maximum = max(int(record["values"].max(initial=0)) for record in records)
    bins = np.arange(-0.5, maximum + 1.5, 1.0)
    centers = (bins[:-1] + bins[1:]) / 2
    histogram_payload = {"n_jets_bins": bins}

    hep.style.use("CMS")
    figure, axis = plt.subplots(figsize=(8.2, 6.3))
    handles = []
    for record in records:
        counts, _ = np.histogram(record["values"], bins=bins)
        density = counts / counts.sum()
        color = multirun_color(record["style"])
        marker = multirun_marker(record["style"])
        axis.stairs(density, bins, color=color, linewidth=2.4)
        nonzero = np.flatnonzero(counts)
        axis.plot(
            centers[nonzero],
            density[nonzero],
            linestyle="none",
            marker=marker,
            markersize=8,
            color=color,
        )
        slug = record["label"].lower().replace(" ", "_")
        histogram_payload[f"n_jets_{slug}_counts"] = counts
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linewidth=2.4,
                markersize=11,
                label=record["label"],
            )
        )
    axis.set_xlabel(
        rf"Number of anti-$k_t$ AK4 jets ($p_T \geq {args.jet_min_pt:g}$ GeV)"
    )
    axis.set_ylabel("Fraction of events")
    axis.set_xlim(bins[0], bins[-1])
    axis.legend(handles=handles, fontsize=18, frameon=False)
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "original_ak4_jet_count_tt_vs_sm_mixture.png", dpi=220)
    plt.close(figure)
    np.savez_compressed(
        args.output_dir / "original_ak4_jet_count_tt_vs_sm_mixture_histograms.npz",
        **histogram_payload,
    )

    summary = {
        "events_per_sample": args.events,
        "jet_algorithm": "anti-kt",
        "jet_radius": args.jet_radius,
        "jet_min_pt_gev": args.jet_min_pt,
        "particle_selection": "canonical particle schema (PuppiW > 0.05)",
        "samples": [
            {
                "label": record["label"],
                "run_dir": record["run_dir"],
                "suite": record["suite"],
                "events": int(len(record["values"])),
                "mean_n_jets": float(np.mean(record["values"])),
                "std_n_jets": float(np.std(record["values"])),
                "color": multirun_color(record["style"]),
                "marker": multirun_marker(record["style"]),
            }
            for record in records
        ],
    }
    (args.output_dir / "original_ak4_jet_count_tt_vs_sm_mixture_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
