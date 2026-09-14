#!/usr/bin/env python
"""Compare reconstructed jet multiplicities from several ORBIT checkpoints."""

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

import awkward as ak
import fastjet
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import vector
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm.auto import tqdm

from gabbro.plotting.orbit import (
    TRUTH_REFERENCE_COLOR,
    TRUTH_REFERENCE_FILL_ALPHA,
    multirun_color,
    multirun_marker,
)
from scripts.compare_orbit_physical_particle_kinematics import (
    evaluation_loader,
    load_model,
    to_physical,
)

vector.register_awkward()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        required=True,
        metavar=("RUN_DIR", "LABEL"),
        help="Checkpoint run and display label; repeat for every model.",
    )
    parser.add_argument(
        "--evaluation-run-dir",
        type=Path,
        required=True,
        help="Run whose resolved data config defines the common evaluation sample.",
    )
    parser.add_argument("--suite", default="training_like")
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--jet-radius", type=float, default=0.8)
    parser.add_argument("--jet-min-pt", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def jet_counts(
    transformed: np.ndarray,
    mask: np.ndarray,
    jet_radius: float,
    jet_min_pt: float,
) -> np.ndarray:
    """Return anti-kt inclusive-jet counts for a padded batch of events."""
    physical = to_physical(transformed)
    momenta = ak.zip(
        {
            "pt": ak.Array([physical["pt"][i][mask[i]] for i in range(len(mask))]),
            "eta": ak.Array([physical["eta"][i][mask[i]] for i in range(len(mask))]),
            "phi": ak.Array([physical["phi"][i][mask[i]] for i in range(len(mask))]),
            "mass": ak.Array(
                [np.zeros(np.count_nonzero(mask[i]), dtype=np.float64) for i in range(len(mask))]
            ),
        },
        with_name="Momentum4D",
    )
    cluster = fastjet.ClusterSequence(
        momenta,
        fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius),
    )
    jets = cluster.inclusive_jets(min_pt=jet_min_pt)
    return np.asarray(ak.num(jets, axis=1), dtype=np.int64)


def legend_handles(payload: dict) -> list:
    handles = [
        Patch(
            facecolor=TRUTH_REFERENCE_COLOR,
            alpha=TRUTH_REFERENCE_FILL_ALPHA,
            label="Original SM mixture",
        )
    ]
    for decoded in payload["decoded"]:
        handles.append(
            Line2D(
                [],
                [],
                color=multirun_color(decoded["style"]),
                marker=multirun_marker(decoded["style"]),
                linewidth=2.2,
                markersize=11,
                label=decoded["label"],
            )
        )
    return handles


def stairs_with_markers(axis, values, bins, color, marker) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    axis.stairs(counts, bins, color=color, linewidth=2.2)
    centers = (bins[:-1] + bins[1:]) / 2
    nonzero = np.flatnonzero(counts)
    axis.plot(
        centers[nonzero],
        counts[nonzero],
        linestyle="none",
        marker=marker,
        markersize=7,
        color=color,
    )
    return counts


def save_plots(payload: dict, output_dir: Path, jet_radius: float, jet_min_pt: float) -> None:
    hep.style.use("CMS")
    output_dir.mkdir(parents=True, exist_ok=True)
    original = np.asarray(payload["original"], dtype=np.int64)
    maximum = max(
        [int(original.max(initial=0))]
        + [int(np.asarray(item["values"]).max(initial=0)) for item in payload["decoded"]]
    )
    count_bins = np.arange(-0.5, maximum + 1.5, 1.0)
    difference_values = [np.asarray(item["values"]) - original for item in payload["decoded"]]
    max_abs_difference = max(
        1,
        max(int(np.max(np.abs(values), initial=0)) for values in difference_values),
    )
    difference_bins = np.arange(-max_abs_difference - 0.5, max_abs_difference + 1.5, 1.0)

    saved = {"n_jets_bins": count_bins}
    original_counts, _ = np.histogram(original, bins=count_bins)
    saved["n_jets_original_counts"] = original_counts

    figure, axis = plt.subplots(figsize=(8.2, 6.3))
    axis.stairs(
        original_counts,
        count_bins,
        fill=True,
        color=TRUTH_REFERENCE_COLOR,
        alpha=TRUTH_REFERENCE_FILL_ALPHA,
        linewidth=1.2,
    )
    for item in payload["decoded"]:
        slug = item["style"].lower().replace(" ", "_")
        saved[f"n_jets_{slug}_counts"] = stairs_with_markers(
            axis,
            item["values"],
            count_bins,
            multirun_color(item["style"]),
            multirun_marker(item["style"]),
        )
    axis.set_xlabel(
        rf"Number of anti-$k_t$ $R={jet_radius:g}$ jets ($p_T \geq {jet_min_pt:g}$ GeV)"
    )
    axis.set_ylabel("Events")
    axis.set_xlim(count_bins[0], count_bins[-1])
    axis.legend(handles=legend_handles(payload), fontsize=17, frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "physical_jet_count.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 6.3))
    for item, differences in zip(payload["decoded"], difference_values):
        slug = item["style"].lower().replace(" ", "_")
        counts = stairs_with_markers(
            axis,
            differences,
            difference_bins,
            multirun_color(item["style"]),
            multirun_marker(item["style"]),
        )
        saved[f"delta_n_jets_{slug}_counts"] = counts
    saved["delta_n_jets_bins"] = difference_bins
    axis.axvline(0, color="black", linestyle="--", linewidth=1.3, alpha=0.65)
    axis.set_xlabel(r"$N_\mathrm{jets}^\mathrm{decoded}-N_\mathrm{jets}^\mathrm{original}$")
    axis.set_ylabel("Events")
    residual_handles = legend_handles(payload)[1:]
    axis.legend(handles=residual_handles, fontsize=17, frameon=False)
    figure.tight_layout()
    figure.savefig(
        output_dir / "physical_jet_count_difference.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)
    np.savez_compressed(output_dir / "physical_jet_count_histograms.npz", **saved)


def main() -> None:
    args = parse_args()
    if args.events < 1 or args.jet_radius <= 0 or args.jet_min_pt < 0:
        raise ValueError("Events and jet radius must be positive; jet pT must be non-negative")
    device = torch.device(args.device)
    run_specs = [(Path(path).resolve(), label) for path, label in args.run]
    models = []
    checkpoints = []
    for run_dir, label in run_specs:
        model, checkpoint = load_model(run_dir, device)
        models.append((label, model))
        checkpoints.append(str(checkpoint))

    original_counts = []
    decoded_counts = {label: [] for label, _ in models}
    processed_events = 0
    loader = evaluation_loader(
        args.evaluation_run_dir.resolve(), args.suite, args.events, args.num_workers
    )
    with tqdm(total=args.events, desc="Decoding and clustering events", unit="event") as bar:
        for batch in loader:
            features = batch["part_features"].to(device)
            mask = batch["part_mask"].to(device)
            pid = batch.get("part_pid")
            if pid is not None:
                pid = pid.to(device)
            mask_numpy = mask.detach().cpu().numpy().astype(bool)
            features_numpy = features.detach().cpu().numpy()
            original_counts.append(
                jet_counts(features_numpy, mask_numpy, args.jet_radius, args.jet_min_pt)
            )
            for label, model in models:
                with torch.inference_mode():
                    reconstructed, _ = model.forward(features, mask, pid_particle=pid)
                decoded_counts[label].append(
                    jet_counts(
                        reconstructed.detach().cpu().numpy(),
                        mask_numpy,
                        args.jet_radius,
                        args.jet_min_pt,
                    )
                )
            batch_events = len(mask_numpy)
            processed_events += batch_events
            bar.update(batch_events)

    original = np.concatenate(original_counts)
    payload = {
        "original": original,
        "decoded": [
            {
                "label": label,
                "style": label,
                "values": np.concatenate(decoded_counts[label]),
            }
            for label, _ in models
        ],
    }
    save_plots(payload, args.output_dir, args.jet_radius, args.jet_min_pt)
    summary = {
        "evaluation_run_dir": str(args.evaluation_run_dir.resolve()),
        "test_suite": args.suite,
        "events": processed_events,
        "jet_algorithm": "anti-kt",
        "jet_radius": args.jet_radius,
        "jet_min_pt_gev": args.jet_min_pt,
        "original_mean_n_jets": float(np.mean(original)),
        "runs": [],
    }
    for (run_dir, label), checkpoint, item in zip(
        run_specs, checkpoints, payload["decoded"]
    ):
        values = np.asarray(item["values"])
        summary["runs"].append(
            {
                "run_dir": str(run_dir),
                "label": label,
                "checkpoint": checkpoint,
                "mean_n_jets": float(np.mean(values)),
                "mean_delta_n_jets": float(np.mean(values - original)),
                "exact_count_fraction": float(np.mean(values == original)),
                "color": multirun_color(label),
                "marker": multirun_marker(label),
            }
        )
    (args.output_dir / "physical_jet_count_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
