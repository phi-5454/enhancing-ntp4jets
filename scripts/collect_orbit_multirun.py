#!/usr/bin/env python
"""Aggregate ORBIT-style artifacts produced by Hydra multirun jobs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from gabbro.plotting.orbit import (
    plot_multirun_feature_histograms,
    plot_multirun_metric,
    plot_multirun_residual_histograms,
)


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = list(directory.glob(pattern))
    if not files:
        return None

    def sort_key(path: Path) -> tuple[int, float]:
        match = re.search(r"_step_(\d+)", path.stem)
        return (int(match.group(1)) if match else -1, path.stat().st_mtime)

    return max(files, key=sort_key)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open() as file:
        return json.load(file)


def _load_latest_csv_metrics(run_dir: Path) -> dict[str, Any]:
    csv_files = list(run_dir.glob("csv/**/metrics.csv"))
    if not csv_files:
        return {}
    metrics: dict[str, Any] = {}
    with max(csv_files, key=lambda path: path.stat().st_mtime).open(newline="") as file:
        for row in csv.DictReader(file):
            for key, value in row.items():
                if value not in ("", None):
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = value
    return metrics


def _resolved_name(cfg: DictConfig, path: str, default: str) -> str:
    try:
        value = OmegaConf.select(cfg, path)
        return str(value) if value else default
    except Exception:
        return default


def _branch_metadata(branch: str, split_cfg: DictConfig, default_quantizer: str) -> dict[str, Any]:
    quantizer = split_cfg.get(f"{branch}_quantizer") or default_quantizer
    if quantizer == "fsq":
        levels = list(split_cfg.get(f"fsq_{branch}_levels", []))
        return {
            "quantizer": quantizer,
            "levels": levels,
            "num_codes": int(np.prod(levels, dtype=np.int64)) if levels else 1,
            "active": bool(levels),
        }
    if quantizer == "vq":
        dim = int(split_cfg.get(f"vq_{branch}_dim", 0))
        return {
            "quantizer": quantizer,
            "dim": dim,
            "num_codes": int(split_cfg.get(f"vq_{branch}_num_codes", 1)),
            "active": dim > 0,
        }
    raise ValueError(f"Unsupported quantizer type {quantizer!r} for branch {branch!r}")


def _quantizer_metadata(cfg: DictConfig) -> dict[str, Any]:
    model_kwargs = cfg.model.model_kwargs
    split_cfg = model_kwargs.get("split_quantizer_cfg")
    if not split_cfg or split_cfg.get("mode", "vq") == "vq":
        num_codes = int(model_kwargs.vq_kwargs.num_codes)
        return {
            "quantizer_mode": "single_vq",
            "quantizer_family": "vq",
            "total_codebook_size": num_codes,
            "branches": {},
        }

    default_quantizer = split_cfg.get("quantizer", "fsq")
    branches = {
        branch: _branch_metadata(branch, split_cfg, default_quantizer)
        for branch in split_cfg.get("branch_order", ["mu", "alpha"])
    }
    active_branches = {name: metadata for name, metadata in branches.items() if metadata["active"]}
    families = {metadata["quantizer"] for metadata in active_branches.values()}
    return {
        "quantizer_mode": "split",
        "quantizer_family": next(iter(families)) if len(families) == 1 else "mixed",
        "total_codebook_size": int(
            np.prod([metadata["num_codes"] for metadata in active_branches.values()], dtype=np.int64)
        ),
        "branches": branches,
    }


def _histogram_features(histograms: dict[str, np.ndarray]) -> list[str]:
    suffix = "_orig_counts"
    return sorted(key[: -len(suffix)] for key in histograms if key.endswith(suffix))


def _load_histograms(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    with np.load(path) as histograms:
        return {key: histograms[key] for key in histograms.files}


def _serializable_record(record: dict[str, Any]) -> dict[str, Any]:
    serialized = {}
    for key, value in record.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, (dict, list)):
            serialized[key] = json.dumps(value, sort_keys=True)
        else:
            serialized[key] = value
    return serialized


def _collect_record(run_dir: Path, stage: str) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    histogram_path = _latest_file(run_dir / "saved_histograms", f"{stage}_orbit_hists_step_*.npz")
    metrics_path = _latest_file(run_dir / "saved_metrics", f"{stage}_orbit_metrics_step_*.json")
    metrics = _load_latest_csv_metrics(run_dir)
    metrics.update(_load_json(metrics_path))
    metadata = _quantizer_metadata(cfg)
    label = _resolved_name(cfg, "logger.wandb.name", _resolved_name(cfg, "task_name", run_dir.name))
    record = {
        "job": run_dir.name,
        "label": label,
        "run_dir": run_dir,
        "histogram_path": histogram_path,
        "metrics_path": metrics_path,
        "seed": int(cfg.get("seed", 0)),
        **metadata,
        **metrics,
    }
    return record, _load_histograms(histogram_path)


def _write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_records = [_serializable_record(record) for record in records]
    with (output_dir / "manifest.json").open("w") as file:
        json.dump(serializable_records, file, indent=2, sort_keys=True)
    fields = sorted({key for record in serializable_records for key in record})
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serializable_records)


def _save_figures(records: list[dict[str, Any]], histogram_runs: list[dict], output_dir: Path) -> None:
    figures = {
        "codebook_size_vs_mse_total": plot_multirun_metric(
            records,
            "metrics/mse_total",
            "Total reconstruction MSE",
            "Codebook size vs. reconstruction MSE",
            log_y=True,
        ),
        "codebook_size_vs_utilization_total": plot_multirun_metric(
            records,
            "metrics/utilization_total",
            "Codebook utilization",
            "Codebook size vs. utilization",
        ),
        "codebook_size_vs_val_loss": plot_multirun_metric(
            records,
            "val_loss_epoch",
            "Validation loss",
            "Codebook size vs. validation loss",
            log_y=True,
        ),
    }
    if histogram_runs:
        common_features = set(_histogram_features(histogram_runs[0]["histograms"]))
        for run in histogram_runs[1:]:
            common_features &= set(_histogram_features(run["histograms"]))
        feature_names = sorted(common_features)
        if feature_names:
            figures["combined_reconstruction_features"] = plot_multirun_feature_histograms(
                histogram_runs,
                feature_names,
            )
            figures["combined_reconstruction_residuals"] = plot_multirun_residual_histograms(
                histogram_runs,
                feature_names,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        if figure is None:
            continue
        figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multirun-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_paths = sorted(args.multirun_dir.glob("*/.hydra/config.yaml"))
    if not config_paths:
        raise SystemExit(f"No Hydra job directories found under {args.multirun_dir}")

    records = []
    loaded_histograms = []
    for config_path in config_paths:
        record, histograms = _collect_record(config_path.parent.parent, args.stage)
        records.append(record)
        loaded_histograms.append(histograms)

    label_counts = Counter(record["label"] for record in records)
    for record in records:
        if label_counts[record["label"]] > 1:
            record["label"] = f"{record['label']} [{record['job']}]"
    histogram_runs = [
        {"label": record["label"], "histograms": histograms}
        for record, histograms in zip(records, loaded_histograms)
        if histograms is not None
    ]

    output_dir = args.output_dir or args.multirun_dir / "comparisons"
    _write_outputs(records, output_dir)
    _save_figures(records, histogram_runs, output_dir)
    print(f"Collected {len(records)} runs into {output_dir}")


if __name__ == "__main__":
    main()
