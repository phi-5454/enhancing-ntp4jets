#!/usr/bin/env python
"""Aggregate ORBIT-style artifacts produced by ORBIT training jobs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import pyrootutils

    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = dict
    OmegaConf = None

from gabbro.plotting.orbit import (
    plot_multirun_feature_histograms,
    plot_multirun_metric,
    plot_multirun_residual_histograms,
)


def _load_config(path: Path) -> Any:
    if OmegaConf is not None:
        return OmegaConf.load(path)
    with path.open() as file:
        return yaml.safe_load(file)


def _select(cfg: Any, path: str, default: Any = None) -> Any:
    if OmegaConf is not None:
        return OmegaConf.select(cfg, path, default=default)
    value = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def _get_nested(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


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
        value = _select(cfg, path)
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
    model_target = str(_select(cfg, "model._target_", ""))
    if model_target.endswith("FaissKMeansBaselineLightning"):
        return {
            "quantizer_mode": "kmeans",
            "quantizer_family": "kmeans",
            "total_codebook_size": int(_select(cfg, "model.num_codes")),
            "branches": {},
        }
    if model_target.endswith("DumbQuantizationBaselineLightning"):
        levels = list(_select(cfg, "model.q_levels", []))
        return {
            "quantizer_mode": "scalar_baseline",
            "quantizer_family": "scalar_baseline",
            "total_codebook_size": int(np.prod(levels, dtype=np.int64)),
            "branches": {},
        }

    model_kwargs = _select(cfg, "model.model_kwargs")
    if not _get_nested(model_kwargs, "quantization_enabled", True):
        return {
            "quantizer_mode": "continuous",
            "quantizer_family": "continuous",
            "total_codebook_size": None,
            "branches": {},
        }
    split_cfg = _get_nested(model_kwargs, "split_quantizer_cfg")
    if not split_cfg or split_cfg.get("mode", "vq") == "vq":
        num_codes = int(_get_nested(_get_nested(model_kwargs, "vq_kwargs"), "num_codes"))
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


def _histogram_residual_features(histograms: dict[str, np.ndarray]) -> list[str]:
    suffix = "_diff_counts"
    return sorted(
        key[: -len(suffix)]
        for key in histograms
        if key.endswith(suffix) and f"{key[: -len(suffix)]}_diff_bins" in histograms
    )


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


def _artifact_prefix(stage: str, group: str) -> str:
    return f"{stage}_orbit" if group == "all" else f"{stage}_{group}_orbit"


def _artifact_dirs(run_dir: Path) -> list[Path]:
    """Return candidate directories containing saved ORBIT artifacts for a run."""
    candidates = [run_dir]
    best_ckpt = run_dir / "evaluation" / "best.ckpt"
    if best_ckpt.is_dir():
        candidates.append(best_ckpt)
    evaluation_dir = run_dir / "evaluation"
    if evaluation_dir.is_dir():
        candidates.extend(
            path
            for path in sorted(evaluation_dir.iterdir())
            if path.is_dir() and path != best_ckpt
        )
    return candidates


def _latest_artifact_file(run_dir: Path, subdir: str, pattern: str) -> Path | None:
    for artifact_dir in _artifact_dirs(run_dir):
        path = _latest_file(artifact_dir / subdir, pattern)
        if path is not None:
            return path
    return None


def _collect_record(
    run_dir: Path,
    stage: str,
    group: str,
    label_override: str | None = None,
    family_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    cfg = _load_config(run_dir / ".hydra" / "config.yaml")
    artifact_prefix = _artifact_prefix(stage, group)
    histogram_path = _latest_artifact_file(
        run_dir,
        "saved_histograms",
        f"{artifact_prefix}_hists_step_*.npz",
    )
    metrics_path = _latest_artifact_file(
        run_dir,
        "saved_metrics",
        f"{artifact_prefix}_metrics_step_*.json",
    )
    metrics = _load_latest_csv_metrics(run_dir)
    metrics.update(_load_json(metrics_path))
    metadata = _quantizer_metadata(cfg)
    label = label_override or _resolved_name(
        cfg,
        "logger.wandb.name",
        _resolved_name(cfg, "task_name", run_dir.name),
    )
    record = {
        "job": run_dir.name,
        "label": label,
        "run_dir": run_dir,
        "histogram_path": histogram_path,
        "metrics_path": metrics_path,
        "artifact_dir": histogram_path.parents[1] if histogram_path else None,
        "stage": stage,
        "group": group,
        "plot_family": family_override or (
            "FAISS k-means" if metadata["quantizer_family"] == "kmeans" else None
        ),
        "seed": int(_cfg_get(cfg, "seed", 0)),
        **metadata,
        **metrics,
    }
    return record, _load_histograms(histogram_path)


def _run_dirs_from_multirun(multirun_dir: Path) -> list[Path]:
    config_paths = sorted(multirun_dir.glob("*/.hydra/config.yaml"))
    if not config_paths:
        raise SystemExit(f"No Hydra job directories found under {multirun_dir}")
    return [config_path.parent.parent for config_path in config_paths]


def _validate_run_dirs(run_dirs: list[Path]) -> list[Path]:
    if not run_dirs:
        raise SystemExit("No run directories were provided.")
    missing_configs = [
        run_dir for run_dir in run_dirs if not (run_dir / ".hydra" / "config.yaml").is_file()
    ]
    if missing_configs:
        formatted = "\n".join(f"  {path}" for path in missing_configs)
        raise SystemExit(
            "Every --run-dir must point at a single Hydra run output directory with "
            f".hydra/config.yaml. Missing configs:\n{formatted}"
        )
    return run_dirs


def _parse_named_runs(run_specs: list[list[str]]) -> list[tuple[Path, str | None]]:
    runs = []
    for spec in run_specs:
        if len(spec) not in (1, 2):
            raise SystemExit(
                "Each --run entry must be either '--run RUN_DIR' or "
                "'--run RUN_DIR LABEL'."
            )
        runs.append((Path(spec[0]), spec[1] if len(spec) == 2 else None))
    _validate_run_dirs([run_dir for run_dir, _ in runs])
    return runs


def _parse_family_runs(family_specs: list[list[str]]) -> list[tuple[Path, str | None, str]]:
    runs = []
    for spec in family_specs:
        if len(spec) < 2:
            raise SystemExit(
                "Each --family entry must be '--family FAMILY RUN_DIR [RUN_DIR ...]'."
            )
        family = spec[0]
        for run_dir in spec[1:]:
            runs.append((Path(run_dir), None, family))
    _validate_run_dirs([run_dir for run_dir, _, _ in runs])
    return runs


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


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _add_multirun_metric_aliases(records: list[dict[str, Any]]) -> None:
    """Add stable plotting aliases across old and new ORBIT metric names."""
    alias_specs = {
        "plot_metrics/reco_mse_total": (
            "metrics/mse_total",
            "metrics/transformed_mse_total",
            "loss_reco_l2_per_value",
        ),
        "plot_metrics/transformed_mse_eta_scaled": (
            "metrics/transformed_mse_eta_scaled",
            "metrics/mse_L1T_PUPPIPart_Eta",
        ),
        "plot_metrics/transformed_mse_cos_phi": (
            "metrics/transformed_mse_cos_phi",
            "metrics/mse_L1T_PUPPIPart_Phi_cos",
        ),
        "plot_metrics/transformed_mse_sin_phi": (
            "metrics/transformed_mse_sin_phi",
            "metrics/mse_L1T_PUPPIPart_Phi_sin",
        ),
        "plot_metrics/transformed_mse_log_pt_shifted": (
            "metrics/transformed_mse_log_pt_shifted",
            "metrics/mse_L1T_PUPPIPart_PT",
        ),
        "plot_metrics/physical_mse_eta": (
            "metrics/mse_Eta",
            "metrics/physical_mse_Eta",
            "metrics/mse_eta",
        ),
        "plot_metrics/physical_mse_phi": (
            "metrics/mse_Phi",
            "metrics/physical_mse_Phi",
            "metrics/mse_phi",
        ),
        "plot_metrics/physical_mse_pT": (
            "metrics/mse_pT",
            "metrics/physical_mse_pT",
        ),
        "plot_metrics/physical_mse_total": (
            "metrics/physical_mse_total",
        ),
    }
    for record in records:
        for alias, keys in alias_specs.items():
            value = _first_present(record, keys)
            if value is not None:
                record[alias] = value


def _save_figures(
    records: list[dict[str, Any]],
    histogram_runs: list[dict],
    output_dir: Path,
    *,
    original_bits_per_input_particle: float = 43.0,
    continuous_autoencoder_mse: float | None = None,
) -> None:
    figures = {
        "codebook_size_vs_reco_mse": plot_multirun_metric(
            records,
            "plot_metrics/reco_mse_total",
            "Reconstruction MSE",
            "Codebook size vs. reconstruction MSE",
            log_y=True,
        ),
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
        "codebook_size_vs_marginal_entropy_bits_per_token": plot_multirun_metric(
            records,
            "metrics/entropy/marginal_bits_per_token",
            "Marginal entropy [bits/token]",
            "Codebook size vs. marginal token entropy",
            reference_fn=np.log2,
            reference_label=r"$\log_2 |\mathcal{C}|$ upper bound",
        ),
        "codebook_size_vs_normalized_entropy": plot_multirun_metric(
            records,
            "metrics/entropy/normalized_to_log2_codebook",
            r"Marginal entropy / $\log_2 |\mathcal{C}|$",
            "Codebook size vs. normalized marginal entropy",
        ),
        "codebook_size_vs_marginal_bits_per_event": plot_multirun_metric(
            records,
            "metrics/rate/marginal_bits_per_event",
            "Marginal rate [bits/event]",
            "Codebook size vs. marginal event rate",
            log_y=True,
        ),
        "codebook_size_vs_marginal_bits_per_input_particle": plot_multirun_metric(
            records,
            "metrics/rate/marginal_bits_per_input_particle",
            "Marginal rate [bits/input particle]",
            "Codebook size vs. marginal rate per input particle",
            log_y=True,
        ),
        "marginal_bits_per_input_particle_vs_reco_mse": plot_multirun_metric(
            records,
            "plot_metrics/reco_mse_total",
            "Reconstruction MSE",
            "Marginal rate per input particle vs. reconstruction MSE",
            log_y=True,
            x_metric="metrics/rate/marginal_bits_per_input_particle",
            xlabel="Marginal rate [bits/input particle]",
            vertical_reference=original_bits_per_input_particle,
            vertical_reference_label=(
                f"Original particle representation ({original_bits_per_input_particle:g} bits)"
            ),
            horizontal_reference=continuous_autoencoder_mse,
            horizontal_reference_label="Continuous autoencoder reference",
        ),
        "marginal_bits_per_event_vs_reco_mse": plot_multirun_metric(
            records,
            "plot_metrics/reco_mse_total",
            "Reconstruction MSE",
            "Marginal event rate vs. reconstruction MSE",
            log_y=True,
            x_metric="metrics/rate/marginal_bits_per_event",
            xlabel="Marginal rate [bits/event]",
        ),
        "codebook_size_vs_val_loss": plot_multirun_metric(
            records,
            "val_loss_epoch",
            "Validation loss",
            "Codebook size vs. validation loss",
            log_y=True,
        ),
        "codebook_size_vs_transformed_mse_eta_scaled": plot_multirun_metric(
            records,
            "plot_metrics/transformed_mse_eta_scaled",
            r"MSE $\eta/3$",
            r"Codebook size vs. transformed $\eta/3$ MSE",
            log_y=True,
        ),
        "codebook_size_vs_transformed_mse_cos_phi": plot_multirun_metric(
            records,
            "plot_metrics/transformed_mse_cos_phi",
            r"MSE $\cos\phi$",
            r"Codebook size vs. transformed $\cos\phi$ MSE",
            log_y=True,
        ),
        "codebook_size_vs_transformed_mse_sin_phi": plot_multirun_metric(
            records,
            "plot_metrics/transformed_mse_sin_phi",
            r"MSE $\sin\phi$",
            r"Codebook size vs. transformed $\sin\phi$ MSE",
            log_y=True,
        ),
        "codebook_size_vs_transformed_mse_log_pt_shifted": plot_multirun_metric(
            records,
            "plot_metrics/transformed_mse_log_pt_shifted",
            r"MSE $\log(p_T)-1.8$",
            r"Codebook size vs. transformed $\log(p_T)-1.8$ MSE",
            log_y=True,
        ),
        "codebook_size_vs_physical_mse_eta": plot_multirun_metric(
            records,
            "plot_metrics/physical_mse_eta",
            r"Physical MSE $\eta$",
            r"Codebook size vs. physical $\eta$ MSE",
            log_y=True,
        ),
        "codebook_size_vs_physical_mse_phi": plot_multirun_metric(
            records,
            "plot_metrics/physical_mse_phi",
            r"Physical MSE $\phi$",
            r"Codebook size vs. physical $\phi$ MSE",
            log_y=True,
        ),
        "codebook_size_vs_physical_mse_pT": plot_multirun_metric(
            records,
            "plot_metrics/physical_mse_pT",
            r"Physical MSE $p_T$",
            r"Codebook size vs. physical $p_T$ MSE",
            log_y=True,
        ),
        "codebook_size_vs_physical_mse_total": plot_multirun_metric(
            records,
            "plot_metrics/physical_mse_total",
            "Mean physical MSE",
            "Codebook size vs. mean physical MSE",
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
        common_residual_features = set(_histogram_residual_features(histogram_runs[0]["histograms"]))
        for run in histogram_runs[1:]:
            common_residual_features &= set(_histogram_residual_features(run["histograms"]))
        residual_feature_names = sorted(common_residual_features)
        if residual_feature_names:
            figures["combined_reconstruction_residuals"] = plot_multirun_residual_histograms(
                histogram_runs,
                residual_feature_names,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        if figure is None:
            continue
        figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def _upload_to_wandb(
    output_dir: Path,
    project: str,
    name: str | None,
    group: str | None,
    entity: str | None,
) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("W&B upload requested, but wandb is not installed.") from exc

    run = wandb.init(project=project, name=name, group=group, entity=entity, job_type="comparison")
    try:
        payload = {}
        for image_path in sorted(output_dir.glob("*.png")):
            payload[f"multirun_plots/{image_path.stem}"] = wandb.Image(str(image_path))
        for artifact_path in (output_dir / "manifest.json", output_dir / "summary.csv"):
            if artifact_path.exists():
                run.save(str(artifact_path), base_path=str(output_dir))
        if payload:
            run.log(payload)
    finally:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--multirun-dir",
        type=Path,
        help="Hydra multirun directory whose immediate children are run output dirs.",
    )
    source.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        action="append",
        help="One or more explicit Hydra run output directories to compare.",
    )
    source.add_argument(
        "--run",
        nargs="+",
        action="append",
        metavar="RUN_SPEC",
        help=(
            "Repeatable explicit run entry with optional label: "
            "--run RUN_DIR or --run RUN_DIR LABEL."
        ),
    )
    source.add_argument(
        "--family",
        nargs="+",
        action="append",
        metavar="FAMILY_SPEC",
        help=(
            "Repeatable family entry for grouped scalar plots: "
            "--family FAMILY RUN_DIR [RUN_DIR ...]."
        ),
    )
    parser.add_argument("--stage", choices=("val", "test"), default="val")
    parser.add_argument("--group", default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-project", help="Upload generated comparison outputs to W&B.")
    parser.add_argument("--wandb-name", help="W&B comparison run name.")
    parser.add_argument("--wandb-group", help="W&B comparison run group.")
    parser.add_argument("--wandb-entity", help="W&B entity/team.")
    parser.add_argument(
        "--original-bits-per-input-particle",
        type=float,
        default=43.0,
        help="Vertical source-size reference in the particle rate-distortion plot (default: 43).",
    )
    parser.add_argument(
        "--continuous-autoencoder-mse",
        type=float,
        help="Optional horizontal MSE reference from a continuous autoencoder.",
    )
    args = parser.parse_args()

    if args.original_bits_per_input_particle <= 0:
        parser.error("--original-bits-per-input-particle must be positive")
    if args.continuous_autoencoder_mse is not None and args.continuous_autoencoder_mse <= 0:
        parser.error("--continuous-autoencoder-mse must be positive")

    if args.multirun_dir is not None:
        run_specs = [
            (run_dir, None, None) for run_dir in _run_dirs_from_multirun(args.multirun_dir)
        ]
        default_output_dir = args.multirun_dir / "comparisons" / args.stage / args.group
    elif args.family is not None:
        run_specs = _parse_family_runs(args.family)
        default_output_dir = Path.cwd() / "orbit_run_comparison" / args.stage / args.group
    elif args.run is not None:
        run_specs = [(run_dir, label, None) for run_dir, label in _parse_named_runs(args.run)]
        default_output_dir = Path.cwd() / "orbit_run_comparison" / args.stage / args.group
    else:
        run_dirs = _validate_run_dirs([run_dir for group in args.run_dir for run_dir in group])
        run_specs = [(run_dir, None, None) for run_dir in run_dirs]
        default_output_dir = Path.cwd() / "orbit_run_comparison" / args.stage / args.group

    records = []
    loaded_histograms = []
    for run_dir, label_override, family_override in run_specs:
        record, histograms = _collect_record(
            run_dir,
            args.stage,
            args.group,
            label_override,
            family_override,
        )
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

    output_dir = args.output_dir or default_output_dir
    _add_multirun_metric_aliases(records)
    _write_outputs(records, output_dir)
    _save_figures(
        records,
        histogram_runs,
        output_dir,
        original_bits_per_input_particle=args.original_bits_per_input_particle,
        continuous_autoencoder_mse=args.continuous_autoencoder_mse,
    )
    if args.wandb_project:
        _upload_to_wandb(
            output_dir,
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group,
            entity=args.wandb_entity,
        )
    print(f"Collected {len(records)} runs into {output_dir}")


if __name__ == "__main__":
    main()
