"""Optional pre-filtering utility for sparse ORBIT parquet datasets.

This is intentionally separate from the training dataloader. Use it only if
profiling shows that repeatedly reading empty events is a material I/O cost.
"""

import argparse
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def filter_non_empty_events(
    source: Path,
    destination: Path,
    sequence_column: str,
) -> tuple[int, int]:
    """Write rows containing at least one sequence element and return row counts."""
    table = pq.read_table(source)
    non_empty = pc.greater(pc.list_value_length(table[sequence_column]), 0)
    filtered = table.filter(non_empty)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, destination)
    return len(table), len(filtered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--sequence-column",
        default="L1T_PUPPIPart_PT",
        help="List-valued column used to identify empty events.",
    )
    args = parser.parse_args()

    before, after = filter_non_empty_events(
        args.source,
        args.destination,
        args.sequence_column,
    )
    print(f"{args.source}: {before:,} -> {after:,} events")


if __name__ == "__main__":
    main()
