#!/usr/bin/env python
"""Compare filtered and full-event canonical-tt rate--distortion results."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from gabbro.plotting.orbit import (
    PRESENTATION_CODEBOOK_MARKER_AREA_SCALE,
    codebook_marker_areas,
    multirun_color,
    multirun_marker,
)
from gabbro.plotting.utils import set_mpl_style
from scripts.collect_orbit_multirun import (
    _add_multirun_metric_aliases,
    _latest_artifact_file,
    _load_config,
    _load_json,
    _load_latest_csv_metrics,
    _select,
)


KEY_PATTERNS = (
    r"faiss_kmeans_codes_\d+",
    r"fsq_mu_\d+x3",
    r"vq_rotation_codes_\d+",
    r"vq_ste_codes_\d+",
    r"vq_mu_\d+_fsq_alpha_32",
    r"fsq_mu_[\d_]+_alpha_64",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-manifest", type=Path, required=True)
    parser.add_argument("--full-records", type=Path, required=True)
    parser.add_argument("--filtered-override-run", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compression-reference-bits", type=float, default=40.0)
    return parser.parse_args()


def configuration_key(text: str) -> str | None:
    for pattern in KEY_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def run_metrics(run_dir: Path) -> tuple[str, dict]:
    cfg = _load_config(run_dir / ".hydra" / "config.yaml")
    task = str(_select(cfg, "task_name", run_dir.name))
    key = configuration_key(task)
    if key is None:
        raise ValueError(f"Cannot infer configuration key from {task!r}")
    metrics_path = _latest_artifact_file(
        run_dir,
        "saved_metrics",
        "test_training_like_all_orbit_metrics_step_*.json",
    )
    metrics = _load_latest_csv_metrics(run_dir)
    metrics.update(_load_json(metrics_path))
    metrics["run_dir"] = str(run_dir)
    _add_multirun_metric_aliases([metrics])
    return key, metrics


def usable(record: dict) -> bool:
    return (
        record.get("plot_metrics/reco_mse_total") is not None
        and record.get("metrics/rate/marginal_bits_per_input_particle") is not None
    )


def prepare_records(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    filtered = json.loads(args.filtered_manifest.read_text())
    filtered_by_key = {}
    for record in filtered:
        key = configuration_key(str(record.get("label", "")))
        if key is not None:
            record["configuration"] = key
            filtered_by_key[key] = record
    for run_dir in args.filtered_override_run:
        key, metrics = run_metrics(run_dir)
        if key not in filtered_by_key:
            raise ValueError(f"Filtered override {key} is absent from the golden manifest")
        filtered_by_key[key].update(metrics)

    full_source = json.loads(args.full_records.read_text())
    full = []
    for key, metrics in full_source.items():
        template = filtered_by_key.get(key)
        if template is None:
            continue
        record = {
            "configuration": key,
            "label": template["label"],
            "plot_family": template.get("plot_family") or template["label"],
            "total_codebook_size": template.get("total_codebook_size"),
            **metrics,
        }
        full.append(record)
    return (
        [record for record in filtered_by_key.values() if usable(record)],
        [record for record in full if usable(record)],
    )


def plot_comparison(filtered: list[dict], full: list[dict], reference_bits: float):
    set_mpl_style()
    figure, axis = plt.subplots(figsize=(9, 6.5))
    all_records = filtered + full
    sizes = [float(record["total_codebook_size"]) for record in all_records]
    size_range = (min(sizes), max(sizes))
    selections = (
        ("PUPPIW > 0.05, top 128", filtered, "-", True),
        ("Full event, top 500", full, "--", False),
    )
    families = sorted(
        {record.get("plot_family") or record["label"] for record in all_records}
    )
    for _, records, linestyle, filled in selections:
        grouped = defaultdict(list)
        for record in records:
            grouped[record.get("plot_family") or record["label"]].append(record)
        for family, family_records in sorted(grouped.items()):
            ordered = sorted(
                family_records,
                key=lambda record: (
                    float(record["metrics/rate/marginal_bits_per_input_particle"]),
                    record["configuration"],
                ),
            )
            x = np.asarray(
                [
                    float(record["metrics/rate/marginal_bits_per_input_particle"])
                    / reference_bits
                    for record in ordered
                ]
            )
            y = np.asarray([float(record["plot_metrics/reco_mse_total"]) for record in ordered])
            color = multirun_color(family)
            marker = multirun_marker(family)
            if len(ordered) > 1:
                axis.plot(x, y, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.7)
            areas = PRESENTATION_CODEBOOK_MARKER_AREA_SCALE * codebook_marker_areas(
                ordered, size_range
            )
            axis.scatter(
                x,
                y,
                s=areas,
                marker=marker,
                facecolors=color if filled else "none",
                edgecolors=color,
                linewidths=1.7,
                zorder=3,
            )
    axis.set_yscale("log")
    axis.set_xlabel(f"Compressed rate / {reference_bits:g}-bit particle payload")
    axis.set_ylabel("Reconstruction MSE")

    family_handles = [
        mlines.Line2D(
            [], [], color=multirun_color(family), marker=multirun_marker(family),
            linestyle="none", markersize=8, label=family
        )
        for family in families
    ]
    selection_handles = [
        mlines.Line2D(
            [], [], color="black", marker="o", linestyle="-", markersize=8,
            markerfacecolor="black", label="PUPPIW > 0.05, top 128"
        ),
        mlines.Line2D(
            [], [], color="black", marker="o", linestyle="--", markersize=8,
            markerfacecolor="none", label="Full event, top 500"
        ),
    ]
    family_legend = axis.legend(handles=family_handles, loc="upper right", fontsize=11)
    axis.add_artist(family_legend)
    axis.legend(handles=selection_handles, loc="lower left", fontsize=11)
    figure.tight_layout()
    return figure


def write_summary(filtered: list[dict], full: list[dict], output_dir: Path) -> None:
    rows = []
    for selection, records in (("puppiw_top128", filtered), ("full_top500", full)):
        for record in records:
            rows.append(
                {
                    "selection": selection,
                    "configuration": record["configuration"],
                    "plot_family": record.get("plot_family") or record["label"],
                    "total_codebook_size": record.get("total_codebook_size"),
                    "reconstruction_mse": record["plot_metrics/reco_mse_total"],
                    "marginal_bits_per_input_particle": record[
                        "metrics/rate/marginal_bits_per_input_particle"
                    ],
                    "run_dir": record.get("run_dir"),
                }
            )
    (output_dir / "tt_selection_rate_distortion.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "tt_selection_rate_distortion.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.compression_reference_bits <= 0:
        raise ValueError("--compression-reference-bits must be positive")
    filtered, full = prepare_records(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_comparison(filtered, full, args.compression_reference_bits)
    stem = args.output_dir / "full_tt_vs_puppiw_tt_mse_vs_compression_ratio"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    write_summary(filtered, full, args.output_dir)
    print(f"Wrote {len(filtered)} filtered and {len(full)} full-event points to {args.output_dir}")


if __name__ == "__main__":
    main()
