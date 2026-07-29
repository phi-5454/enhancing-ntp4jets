#!/usr/bin/env python
"""Aggregate five-seed original/decoded classifier runs into a 2x2 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-runs", nargs=5, required=True, type=Path)
    parser.add_argument("--decoded-runs", nargs=5, required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_run(run_dir: Path):
    metrics_dir = run_dir / "downstream_metrics"
    summary = json.loads((metrics_dir / "classifier_metrics.json").read_text())
    predictions = {
        representation: np.load(
            metrics_dir / f"test_{representation}_predictions.npz", allow_pickle=True
        )
        for representation in ("original", "decoded")
    }
    for representation, values in predictions.items():
        if len(np.unique(values["event_ids"])) != len(values["event_ids"]):
            raise ValueError(f"Duplicate event IDs in {run_dir} ({representation})")
    return summary, predictions


def paired_shift(original, decoded):
    original_order = np.argsort(original["event_ids"])
    decoded_order = np.argsort(decoded["event_ids"])
    if not np.array_equal(
        original["event_ids"][original_order], decoded["event_ids"][decoded_order]
    ):
        raise ValueError("Original and decoded prediction files contain different events")
    original_logits = original["logits"][original_order]
    decoded_logits = decoded["logits"][decoded_order]
    original_probabilities = softmax(original_logits, axis=1)
    decoded_probabilities = softmax(decoded_logits, axis=1)
    return {
        "mean_absolute_logit_shift": float(np.mean(np.abs(decoded_logits - original_logits))),
        "mean_absolute_probability_shift": float(
            np.mean(np.abs(decoded_probabilities - original_probabilities))
        ),
        "prediction_change_fraction": float(
            np.mean(
                original_probabilities.argmax(axis=1) != decoded_probabilities.argmax(axis=1)
            )
        ),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    shifts = []
    data_contract = None
    for seed, (original_dir, decoded_dir) in enumerate(
        zip(args.original_runs, args.decoded_runs)
    ):
        for expected, run_dir in (("original", original_dir), ("decoded", decoded_dir)):
            summary, predictions = load_run(run_dir)
            if summary["trained_on"] != expected:
                raise ValueError(f"Expected {run_dir} to be trained on {expected}")
            run_contract = {
                "sequence_type": summary.get("sequence_type", "particle"),
                "max_sequence_length": int(summary.get("max_sequence_length", 128)),
            }
            if data_contract is None:
                data_contract = run_contract
            elif run_contract != data_contract:
                raise ValueError(
                    f"Classifier matrix mixes data contracts: {data_contract} and {run_contract}"
                )
            for tested_on, metrics in summary["representations"].items():
                rows.append(
                    {
                        "seed": seed,
                        "trained_on": expected,
                        "tested_on": tested_on,
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_auroc": metrics["macro_auroc"],
                        "expected_calibration_error": metrics[
                            "expected_calibration_error"
                        ],
                        "gghbb_rejection_50": metrics[
                            "ggHbb_rejection_at_50pct_efficiency"
                        ],
                        "gghbb_rejection_80": metrics[
                            "ggHbb_rejection_at_80pct_efficiency"
                        ],
                    }
                )
            shifts.append(
                {"seed": seed, "trained_on": expected}
                | paired_shift(predictions["original"], predictions["decoded"])
            )

    with (args.output_dir / "classifier_matrix_per_seed.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for trained_on in ("original", "decoded"):
        for tested_on in ("original", "decoded"):
            selected = [
                row
                for row in rows
                if row["trained_on"] == trained_on and row["tested_on"] == tested_on
            ]
            key = f"{trained_on}_to_{tested_on}"
            aggregate[key] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in selected])),
                    "std": float(np.std([row[metric] for row in selected], ddof=1)),
                }
                for metric in rows[0]
                if metric not in {"seed", "trained_on", "tested_on"}
            }
    payload = {
        "seeds": 5,
        "data_contract": data_contract,
        "matrix": aggregate,
        "paired_shifts": shifts,
    }
    (args.output_dir / "classifier_matrix_summary.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(f"Wrote classifier matrix to {args.output_dir}")


if __name__ == "__main__":
    main()
