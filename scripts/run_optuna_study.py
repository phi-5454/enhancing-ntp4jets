#!/usr/bin/env python3
"""Run a resumable native Optuna study using the repository's Hydra training configs.

Example:
    uv run python scripts/run_optuna_study.py --study-root /path/to/study \
        experiment=orbit_jet_puppi_ak8_ggHbb_minbias \
        hparams_search=orbit_optuna_fsq_mu_16x3_lr
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict
import optuna

from gabbro.train import train
from gabbro.utils.utils import get_metric_value


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _storage_url(study_root: Path) -> str:
    return f"sqlite:///{(study_root / 'study.sqlite3').resolve()}"


def _compose(overrides: list[str], trial_overrides: dict[str, object], trial_dir: Path):
    """Compose an ordinary training config with deterministic trial paths."""
    sampled_override_strings = [f"{key}={value}" for key, value in trial_overrides.items()]
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="train.yaml",
            overrides=[*overrides, *sampled_override_strings],
            return_hydra_config=True,
        )

    with open_dict(cfg):
        cfg.paths.output_dir = str(trial_dir)
        cfg.paths.work_dir = str(PROJECT_ROOT)
        cfg.hydra.run.dir = str(trial_dir)
        cfg.hydra.runtime.output_dir = str(trial_dir)
        cfg.hydra.runtime.cwd = str(PROJECT_ROOT)
        cfg.trainer.default_root_dir = str(trial_dir)
        if cfg.logger.get("wandb") is not None:
            cfg.logger.wandb.name = f"{cfg.task_name}_trial_{trial_dir.name}"

    hydra_dir = trial_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml", resolve=False)
    OmegaConf.save(cfg, hydra_dir / "config_resolved.yaml", resolve=True)
    OmegaConf.save(OmegaConf.create([*overrides, *sampled_override_strings]), hydra_dir / "overrides.yaml")
    return cfg


def _suggest(trial: optuna.Trial, search_space) -> dict[str, object]:
    values = {}
    for name, spec in dict(search_space).items():
        spec = dict(spec)
        kind = spec["type"]
        if kind == "float":
            values[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "int":
            values[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "categorical":
            values[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported Optuna search-space type {kind!r} for {name!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True, help="Persistent study directory")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, including hparams_search=...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not any(value.startswith("hparams_search=") for value in args.overrides):
        raise SystemExit("Pass a native Optuna config, e.g. hparams_search=orbit_optuna_fsq_mu_16x3_lr")

    args.study_root.mkdir(parents=True, exist_ok=True)
    base_cfg = _compose(args.overrides, {}, args.study_root / "_config_preview")
    study_cfg = base_cfg.optuna
    if not study_cfg.get("search_space"):
        raise SystemExit("The selected hparams_search config defines no optuna.search_space")

    metadata = {
        "study_name": str(study_cfg.study_name),
        "direction": str(study_cfg.direction),
        "seed": int(study_cfg.seed),
        "target_trials": int(study_cfg.target_trials),
        "overrides": args.overrides,
        "optimized_metric": str(base_cfg.optimized_metric),
    }
    (args.study_root / "study_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    study = optuna.create_study(
        study_name=metadata["study_name"],
        direction=metadata["direction"],
        sampler=optuna.samplers.TPESampler(seed=metadata["seed"]),
        storage=_storage_url(args.study_root),
        load_if_exists=True,
    )
    remaining_trials = max(metadata["target_trials"] - len(study.trials), 0)

    def objective(trial: optuna.Trial) -> float:
        trial_dir = args.study_root / "trials" / f"{trial.number:04d}"
        sampled_overrides = _suggest(trial, study_cfg.search_space)
        trial.set_user_attr("output_dir", str(trial_dir))
        trial.set_user_attr("overrides", sampled_overrides)
        try:
            cfg = _compose(args.overrides, sampled_overrides, trial_dir)
            metric_dict, _ = train(cfg)
            value = get_metric_value(metric_dict, cfg.optimized_metric)
            trial.set_user_attr("optimized_metric", str(cfg.optimized_metric))
            return float(value)
        except Exception as error:
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "failure.txt").write_text(traceback.format_exc())
            trial.set_user_attr("error", f"{type(error).__name__}: {error}")
            raise

    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, catch=(Exception,))
    print(f"Study {study.study_name!r}: {len(study.trials)}/{metadata['target_trials']} trials")
    from report_optuna_study import write_report

    write_report(args.study_root)


if __name__ == "__main__":
    main()
