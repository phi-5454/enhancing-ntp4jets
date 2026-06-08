"""Generate small synthetic ORBIT-format parquet files for smoke tests."""

import argparse
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _write(output_path: pathlib.Path, table: pa.Table, n_events: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    print(f"Wrote {n_events} events to {output_path}")


def generate_particle_parquet(output_path: pathlib.Path, n_events: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    eta, phi, pt, puppiw = [], [], [], []
    for _ in range(n_events):
        n = int(rng.integers(5, 30))
        eta.append(rng.uniform(-3.0, 3.0, n).astype(np.float32))
        phi.append(rng.uniform(-np.pi, np.pi, n).astype(np.float32))
        pt.append(rng.exponential(10.0, n).astype(np.float32))
        puppiw.append(rng.uniform(0.0, 1.0, n).astype(np.float32))
    table = pa.table({
        "L1T_PUPPIPart_Eta": pa.array(eta, type=pa.list_(pa.float32())),
        "L1T_PUPPIPart_Phi": pa.array(phi, type=pa.list_(pa.float32())),
        "L1T_PUPPIPart_PT": pa.array(pt, type=pa.list_(pa.float32())),
        "L1T_PUPPIPart_PuppiW": pa.array(puppiw, type=pa.list_(pa.float32())),
    })
    _write(output_path, table, n_events)


def generate_jet_parquet(output_path: pathlib.Path, prefix: str = "L1T_JetPuppiAK8",
                         n_jets_per_event: int = 7, n_events: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    eta, phi, pt = [], [], []
    for _ in range(n_events):
        n = int(rng.integers(1, n_jets_per_event + 1))
        eta.append(rng.uniform(-3.0, 3.0, n).astype(np.float32))
        phi.append(rng.uniform(-np.pi, np.pi, n).astype(np.float32))
        pt.append(rng.exponential(50.0, n).astype(np.float32))
    table = pa.table({
        f"{prefix}_Eta": pa.array(eta, type=pa.list_(pa.float32())),
        f"{prefix}_Phi": pa.array(phi, type=pa.list_(pa.float32())),
        f"{prefix}_PT": pa.array(pt, type=pa.list_(pa.float32())),
    })
    _write(output_path, table, n_events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--type", choices=["particle", "jet"], default="particle")
    parser.add_argument("--prefix", default="L1T_JetPuppiAK8")
    parser.add_argument("--n-events", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.type == "particle":
        generate_particle_parquet(args.output, n_events=args.n_events, seed=args.seed)
    else:
        generate_jet_parquet(args.output, prefix=args.prefix, n_events=args.n_events, seed=args.seed)
