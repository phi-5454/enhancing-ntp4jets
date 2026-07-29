#!/usr/bin/env python3
"""Compare total and valid PUPPI-particle multiplicities in ORBIT Parquet samples.

Example
-------
export ORBIT_MANIFEST_DIR=/path/to/manifests
uv run --locked python scripts/plot_orbit_particle_counts.py \
    --max-files 2 --output-dir outputs/particle-counts

The script writes ``particle_count_histograms.png`` to the requested output
directory and prints its absolute path.  It reads
the same ``ggHbb_*`` and ``minbias_*`` manifests used by the two-class ORBIT
dataloader configuration.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from gabbro.plotting.utils import set_mpl_style


PT_COLUMN = "L1T_PUPPIPart_PT"
PUPPI_WEIGHT_COLUMN = "L1T_PUPPIPart_PuppiW"
MANIFEST_SUFFIXES = {".txt", ".list", ".lst"}


def expand_inputs(paths: Iterable[Path], seen: set[Path] | None = None) -> list[Path]:
    """Expand Parquet files, directories, and ORBIT-style path manifests."""
    seen = set() if seen is None else seen
    files: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(file for file in path.rglob("*.parquet") if file.is_file()))
        elif path.suffix.lower() in MANIFEST_SUFFIXES:
            if path in seen:
                raise ValueError(f"recursive input manifest: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"input manifest does not exist: {path}")
            seen.add(path)
            entries = []
            for line in path.read_text().splitlines():
                value = line.strip()
                if value and not value.startswith("#"):
                    entry = Path(value).expanduser()
                    entries.append(entry if entry.is_absolute() else path.parent / entry)
            files.extend(expand_inputs(entries, seen))
        elif path.suffix.lower() == ".parquet" and path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"expected a Parquet file, directory, or manifest: {path}")
    return sorted(set(files))


def manifests_for_split(manifest_dir: Path, split: str) -> dict[str, Path]:
    """Return the two class manifests configured for the ORBIT dataloader split."""
    suffix = {"train-val": "train_val", "test": "test"}[split]
    manifests = {
        "ggHbb": manifest_dir / f"ggHbb_{suffix}.txt",
        "minbias": manifest_dir / f"minbias_{suffix}.txt",
    }
    missing = [str(path) for path in manifests.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing ORBIT dataloader manifests:\n  " + "\n  ".join(missing)
        )
    return manifests


def particle_counts(
    files: Iterable[Path],
    max_events_per_file: int,
    puppi_weight_min: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return raw and PUPPI-valid particle counts for the selected Parquet files."""
    total_counts: list[int] = []
    valid_counts: list[int] = []
    all_files_have_weights = True

    for path in files:
        parquet_file = pq.ParquetFile(path)
        schema_names = set(parquet_file.schema_arrow.names)
        if PT_COLUMN not in schema_names:
            raise ValueError(f"{path} does not contain {PT_COLUMN}")
        has_weights = PUPPI_WEIGHT_COLUMN in schema_names
        all_files_have_weights &= has_weights
        columns = [PT_COLUMN] + ([PUPPI_WEIGHT_COLUMN] if has_weights else [])
        events_read = 0
        for batch in parquet_file.iter_batches(columns=columns, batch_size=4096):
            pts_per_event = batch.column(PT_COLUMN).to_pylist()
            weights_per_event = batch.column(PUPPI_WEIGHT_COLUMN).to_pylist() if has_weights else None
            for event_index, pts in enumerate(pts_per_event):
                if events_read == max_events_per_file:
                    break
                pt = np.asarray(pts, dtype=float)
                total_counts.append(len(pt))
                valid = np.isfinite(pt) & (pt > 0.0)
                if has_weights:
                    weights = np.asarray(weights_per_event[event_index], dtype=float)
                    valid &= np.isfinite(weights) & (weights > puppi_weight_min)
                valid_counts.append(int(np.count_nonzero(valid)))
                events_read += 1
            if events_read == max_events_per_file:
                break
    return np.asarray(total_counts), np.asarray(valid_counts), all_files_have_weights


def _histogram_edges(values: np.ndarray, bins: int) -> np.ndarray:
    maximum = int(np.max(values)) if len(values) else 1
    return np.linspace(-0.5, maximum + 0.5, min(bins, maximum + 1) + 1)


def plot_histograms(
    totals: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    output: Path,
    bins: int,
    puppi_weight_min: float,
) -> None:
    """Write separate panels for each class and total/valid count definition."""
    set_mpl_style()
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True, sharey=False)
    panels = (
        (axes[0, 0], "ggHbb", "total", totals["ggHbb"]),
        (axes[0, 1], "ggHbb", "valid", valid["ggHbb"]),
        (axes[1, 0], "minbias", "total", totals["minbias"]),
        (axes[1, 1], "minbias", "valid", valid["minbias"]),
    )
    for axis, label, count_type, values in panels:
        edges = _histogram_edges(values, bins)
        axis.hist(
            values, bins=edges, histtype="step", linewidth=2,
            label=f"n={len(values):,}",
        )
        title = f"{label}: {count_type} particles per event"
        if count_type == "valid":
            title += f" (PuppiW > {puppi_weight_min:g})"
        axis.set(title=title, xlabel="particles per event", ylabel="events")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Directory containing the dataloader manifests (default: $ORBIT_MANIFEST_DIR)",
    )
    parser.add_argument(
        "--split",
        choices=("train-val", "test"),
        default="train-val",
        help="Dataloader manifest split to sample (default: train-val)",
    )
    parser.add_argument("--max-files", type=int, default=2, help="Maximum files sampled per class (default: 2)")
    parser.add_argument("--max-events-per-file", type=int, default=10_000, help="Maximum events read from each selected file")
    parser.add_argument("--puppi-weight-min", type=float, default=0.05, help="Valid particle requires PuppiW > value (default: 0.05)")
    parser.add_argument("--bins", type=int, default=100, help="Maximum number of histogram bins")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/particle-counts"), help="Directory for PNG plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_files < 1 or args.max_events_per_file < 1 or args.bins < 1:
        raise ValueError("--max-files, --max-events-per-file, and --bins must be positive")

    manifest_dir = args.manifest_dir or os.environ.get("ORBIT_MANIFEST_DIR")
    if manifest_dir is None:
        raise ValueError("set ORBIT_MANIFEST_DIR or pass --manifest-dir")
    manifest_dir = Path(manifest_dir).expanduser().resolve()
    manifests = manifests_for_split(manifest_dir, args.split)
    print(f"Sampling the dataloader's {args.split} manifests from {manifest_dir}")
    inputs = {label: expand_inputs([manifest])[: args.max_files] for label, manifest in manifests.items()}
    for label, files in inputs.items():
        if not files:
            raise ValueError(f"no Parquet files found for {label}")
        print(f"{label}: sampling {len(files)} file(s)")
        for path in files:
            print(f"  {path}")

    totals: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    for label, files in inputs.items():
        totals[label], valid[label], has_weights = particle_counts(
            files, args.max_events_per_file, args.puppi_weight_min
        )
        if not has_weights:
            print(f"WARNING: {label} input lacks {PUPPI_WEIGHT_COLUMN}; valid counts use finite pT > 0 only")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "particle_count_histograms.png"
    plot_histograms(totals, valid, output, args.bins, args.puppi_weight_min)
    print(f"Wrote four-panel particle-count histograms: {output}")


if __name__ == "__main__":
    main()
