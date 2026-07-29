#!/usr/bin/env python
"""Plot the L1 PUPPI-weight distribution for an ORBIT Parquet manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from gabbro.plotting.utils import set_mpl_style


WEIGHT_COLUMN = "L1T_PUPPIPart_PuppiW"
DEFAULT_MANIFEST = Path(__file__).parents[1] / "manifests/production_final/ggHbb_test.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda/puppiw"))
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--weight-range", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=150)
    parser.add_argument("--selection-threshold", type=float, default=0.05)
    return parser.parse_args()


def manifest_paths(manifest: Path, max_files: int | None) -> list[Path]:
    paths = [Path(line) for line in manifest.read_text().splitlines() if line.strip()]
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise ValueError("No Parquet files selected")
    return paths


def read_weights(paths: list[Path]) -> tuple[np.ndarray, int]:
    values = []
    events = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=[WEIGHT_COLUMN], batch_size=4096):
            array = ak.from_arrow(batch)[WEIGHT_COLUMN]
            events += len(array)
            weight = ak.to_numpy(ak.flatten(array, axis=None))
            values.append(weight[np.isfinite(weight)])
    return np.concatenate(values), events


def main() -> None:
    args = parse_args()
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive or omitted")
    if args.weight_range <= 0 or args.bins < 1:
        raise ValueError("--weight-range and --bins must be positive")
    paths = manifest_paths(args.manifest, args.max_files)
    weights, events = read_weights(paths)
    positive = weights[weights > 0]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_mpl_style()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    bins = np.linspace(-0.005, args.weight_range, args.bins + 1)
    axes[0].hist(weights, bins=bins, histtype="step", linewidth=1.8)
    axes[0].set_yscale("log")
    axes[0].axvline(args.selection_threshold, color="black", linestyle="--", linewidth=1, label=f"{args.selection_threshold:g} selection")
    axes[0].set(xlabel="PuppiW", ylabel="Particles", title="All PUPPI weights", xlim=(-0.005, args.weight_range))
    axes[0].legend()

    axes[1].hist(positive, bins=np.linspace(0, args.weight_range, args.bins + 1), histtype="step", linewidth=1.8)
    axes[1].set_yscale("log")
    axes[1].axvline(args.selection_threshold, color="black", linestyle="--", linewidth=1, label=f"{args.selection_threshold:g} selection")
    axes[1].set(xlabel="PuppiW", ylabel="Particles with PuppiW > 0", title="Nonzero PUPPI weights", xlim=(0, args.weight_range))
    axes[1].legend()
    output = output_dir / "puppiw_histogram.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)

    summary = {
        "manifest": str(args.manifest.resolve()),
        "files": len(paths),
        "events": events,
        "particles": int(len(weights)),
        "weight_range": args.weight_range,
        "zero_fraction": float(np.mean(weights == 0)),
        "selection_fraction": float(np.mean(weights > args.selection_threshold)),
        "maximum": float(np.max(weights)),
        "above_plot_range_fraction": float(np.mean(weights > args.weight_range)),
        "positive_quantiles": {
            str(q): float(value)
            for q, value in zip((0.05, 0.16, 0.5, 0.84, 0.95), np.quantile(positive, (0.05, 0.16, 0.5, 0.84, 0.95)))
        },
    }
    (output_dir / "puppiw_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
