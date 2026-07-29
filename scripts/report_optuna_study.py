#!/usr/bin/env python3
"""Create CSV, static PNG, and interactive HTML reports for a native Optuna study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd


def _storage_url(study_root: Path) -> str:
    return f"sqlite:///{(study_root / 'study.sqlite3').resolve()}"


def _write_static_plots(frame: pd.DataFrame, report_dir: Path) -> None:
    states = frame["state"].astype(str).value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.bar(states.index, states.values)
    axis.set_ylabel("trials")
    axis.set_title("Optuna trial states")
    figure.savefig(report_dir / "trial_states.png", dpi=160)
    plt.close(figure)

    complete = frame[frame["state"].astype(str).str.contains("COMPLETE")].dropna(subset=["value"])
    if complete.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(complete["number"], complete["value"], "o", label="trial objective")
    best = complete["value"].cummin()
    axis.plot(complete["number"], best, label="best so far")
    axis.set(xlabel="trial", ylabel="objective", title="Optimization history")
    axis.legend()
    figure.savefig(report_dir / "optimization_history.png", dpi=160)
    plt.close(figure)

    for column in [name for name in complete.columns if name.startswith("params_")]:
        values = pd.to_numeric(complete[column], errors="coerce")
        if values.notna().sum() < 2:
            continue
        figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
        axis.scatter(values, complete["value"])
        axis.set(xlabel=column.removeprefix("params_"), ylabel="objective")
        figure.savefig(report_dir / f"objective_vs_{column.removeprefix('params_').replace('.', '_')}.png", dpi=160)
        plt.close(figure)


def _write_interactive_plots(study: optuna.Study, report_dir: Path) -> list[str]:
    names = []
    plotters = {
        "optimization_history": optuna.visualization.plot_optimization_history,
        "param_importances": optuna.visualization.plot_param_importances,
        "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
    }
    for name, plotter in plotters.items():
        try:
            figure = plotter(study)
            output = report_dir / f"{name}.html"
            figure.write_html(output, include_plotlyjs="cdn")
            names.append(output.name)
        except (RuntimeError, ValueError):
            continue
    return names


def write_report(study_root: Path) -> Path:
    study_root = study_root.resolve()
    metadata = json.loads((study_root / "study_metadata.json").read_text())
    study = optuna.load_study(study_name=metadata["study_name"], storage=_storage_url(study_root))
    report_dir = study_root / "report"
    report_dir.mkdir(exist_ok=True)
    frame = study.trials_dataframe()
    frame.to_csv(report_dir / "trials.csv", index=False)

    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    summary = {"study_name": study.study_name, "trials": len(study.trials), "complete_trials": len(complete)}
    if complete:
        summary["best_trial"] = {"number": study.best_trial.number, "value": study.best_value, "params": study.best_params}
    (report_dir / "best_trial.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_static_plots(frame, report_dir)
    html_files = _write_interactive_plots(study, report_dir)
    (report_dir / "index.html").write_text(
        "<h1>Optuna study report</h1><ul>" + "".join(f'<li><a href="{name}">{name}</a></li>' for name in html_files) + "</ul>\n"
    )
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    args = parser.parse_args()
    print(f"Wrote Optuna report to {write_report(args.study_root)}")


if __name__ == "__main__":
    main()
