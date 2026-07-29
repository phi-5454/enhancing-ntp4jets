"""Regression coverage for persisted native Optuna study reports."""

import importlib.util
import json
from pathlib import Path

import optuna


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "report_optuna_study.py"
SPEC = importlib.util.spec_from_file_location("report_optuna_study", SCRIPT_PATH)
report_optuna_study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_optuna_study)


def test_report_contains_machine_and_human_readable_outputs(tmp_path):
    (tmp_path / "study_metadata.json").write_text(json.dumps({"study_name": "report-test"}))
    storage = f"sqlite:///{tmp_path / 'study.sqlite3'}"
    study = optuna.create_study(study_name="report-test", storage=storage, direction="minimize")
    study.optimize(lambda trial: trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True), n_trials=3)

    report_dir = report_optuna_study.write_report(tmp_path)

    assert (report_dir / "trials.csv").is_file()
    assert (report_dir / "best_trial.json").is_file()
    assert (report_dir / "trial_states.png").is_file()
    assert (report_dir / "optimization_history.png").is_file()
    assert (report_dir / "index.html").is_file()
