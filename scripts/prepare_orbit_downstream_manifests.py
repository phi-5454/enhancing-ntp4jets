#!/usr/bin/env python
"""Prepare additive five-class train_val/test manifests without changing canonical files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gabbro.data.orbit_taxonomy import FIVE_CLASS_GROUPS



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest-dir", required=True, type=Path)
    parser.add_argument(
        "--eos-root",
        default=Path("/eos/project/f/foundational-model-dataset/samples/production_final"),
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_manifest(path: Path):
    files = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            candidate = Path(value)
            files.append(str(candidate if candidate.is_absolute() else path.parent / candidate))
    return files


def write_manifest(path: Path, files):
    path.write_text("\n".join(map(str, sorted(files))) + "\n")


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for processes in FIVE_CLASS_GROUPS.values():
        for process in processes:
            test_source = args.canonical_manifest_dir / f"{process}_test.txt"
            if not test_source.is_file():
                raise FileNotFoundError(test_source)
            test_files = set(read_manifest(test_source))
            train_source = args.canonical_manifest_dir / f"{process}_train_val.txt"
            if train_source.is_file():
                train_files = set(read_manifest(train_source))
            elif process == "ggHbb":
                process_dir = args.eos_root / process
                train_files = set(map(str, sorted(process_dir.rglob("*.parquet")))) - test_files
            else:
                raise FileNotFoundError(train_source)
            if not train_files:
                raise ValueError(f"No downstream train/validation files for {process}")
            if train_files & test_files:
                raise ValueError(f"Train/test overlap for {process}")
            write_manifest(args.output_dir / f"{process}_train_val.txt", train_files)
            write_manifest(args.output_dir / f"{process}_test.txt", test_files)
            print(f"{process}: {len(train_files)} train_val files, {len(test_files)} test files")


if __name__ == "__main__":
    main()

