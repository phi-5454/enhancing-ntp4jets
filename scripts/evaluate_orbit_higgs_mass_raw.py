#!/usr/bin/env python
"""Plot the resolved ggHbb mass from raw, highest-pT L1 PUPPI particles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import awkward as ak
import fastjet
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

from gabbro.plotting.utils import set_mpl_style


PARTICLE_COLUMNS = (
    "L1T_PUPPIPart_PT",
    "L1T_PUPPIPart_Eta",
    "L1T_PUPPIPart_Phi",
)
TRUTH_COLUMNS = (
    "Gen_Part_PT",
    "Gen_Part_Eta",
    "Gen_Part_Phi",
    "Gen_Part_Mass",
    "Gen_Part_PID",
    "Gen_Part_D1",
    "Gen_Part_D2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gghbb-test-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--top-pt-particles", type=int, default=500)
    parser.add_argument("--jet-radius", type=float, default=0.4)
    parser.add_argument("--jet-min-pt", type=float, default=30.0)
    parser.add_argument("--match-dr", type=float, default=0.2)
    return parser.parse_args()


def decaying_higgs(event):
    pids = np.asarray(ak.to_numpy(event.Gen_Part_PID), dtype=np.int64)
    daughters_1 = np.asarray(ak.to_numpy(event.Gen_Part_D1), dtype=np.int64)
    daughters_2 = np.asarray(ak.to_numpy(event.Gen_Part_D2), dtype=np.int64)
    for higgs_index in np.flatnonzero(np.abs(pids) == 25)[::-1]:
        daughter_1, daughter_2 = int(daughters_1[higgs_index]), int(daughters_2[higgs_index])
        if not (0 <= daughter_1 < len(pids) and 0 <= daughter_2 < len(pids)):
            continue
        if {int(pids[daughter_1]), int(pids[daughter_2])} != {5, -5}:
            continue

        def four_vector(index):
            return np.array(
                [
                    float(event.Gen_Part_Eta[index]),
                    float(event.Gen_Part_Phi[index]),
                    float(event.Gen_Part_PT[index]),
                    float(event.Gen_Part_Mass[index]),
                ]
            )

        return four_vector(daughter_1), four_vector(daughter_2)
    return None


def delta_r(eta_a, phi_a, eta_b, phi_b):
    return np.hypot(eta_a - eta_b, np.remainder(phi_a - phi_b + np.pi, 2 * np.pi) - np.pi)


def cluster_jets(particles: np.ndarray, radius: float, min_pt: float):
    if not len(particles):
        return np.empty((0, 4))
    momenta = ak.zip(
        {
            "pt": [particles[:, 2]],
            "eta": [particles[:, 0]],
            "phi": [particles[:, 1]],
            "mass": [np.zeros(len(particles))],
        },
        with_name="Momentum4D",
    )
    jets = fastjet.ClusterSequence(momenta, fastjet.JetDefinition(fastjet.antikt_algorithm, radius)).inclusive_jets(min_pt=min_pt)[0]
    if not len(jets):
        return np.empty((0, 4))
    values = np.stack([ak.to_numpy(jets.eta), ak.to_numpy(jets.phi), ak.to_numpy(jets.pt), ak.to_numpy(jets.mass)], axis=-1)
    return values[np.abs(values[:, 0]) < 2.5]


def combine_mass(first, second):
    def components(jet):
        eta, phi, pt, mass = jet
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        return np.array([np.sqrt(px * px + py * py + pz * pz + mass * mass), px, py, pz])

    total = components(first) + components(second)
    return float(np.sqrt(max(total[0] ** 2 - np.dot(total[1:], total[1:]), 0.0)))


def resolved_candidate(particles, b, bbar, radius, min_pt, match_dr):
    jets = cluster_jets(particles, radius, min_pt)
    if len(jets) < 2:
        return None
    distances = np.array(
        [[delta_r(parton[0], parton[1], jet[0], jet[1]) for jet in jets] for parton in (b, bbar)]
    )
    parton_indices, jet_indices = linear_sum_assignment(distances)
    if len(jet_indices) != 2 or np.any(distances[parton_indices, jet_indices] >= match_dr):
        return None
    return combine_mass(jets[jet_indices[0]], jets[jet_indices[1]])


def top_pt_particles(event, count):
    pt = np.asarray(ak.to_numpy(event.L1T_PUPPIPart_PT), dtype=np.float32)
    eta = np.asarray(ak.to_numpy(event.L1T_PUPPIPart_Eta), dtype=np.float32)
    phi = np.asarray(ak.to_numpy(event.L1T_PUPPIPart_Phi), dtype=np.float32)
    valid = np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi) & (pt > 0)
    particles = np.stack([eta[valid], phi[valid], pt[valid]], axis=-1)
    return particles[np.argsort(particles[:, 2])[::-1][:count]]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if args.events < 1 or args.top_pt_particles < 1:
        raise ValueError("--events and --top-pt-particles must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(line) for line in args.gghbb_test_manifest.read_text().splitlines() if line.strip()]
    rows = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=PARTICLE_COLUMNS + TRUTH_COLUMNS, batch_size=512):
            events = ak.from_arrow(batch)
            for event in events:
                truth = decaying_higgs(event)
                if truth is None:
                    continue
                b, bbar = truth
                particles = top_pt_particles(event, args.top_pt_particles)
                rows.append({"resolved_mass": resolved_candidate(particles, b, bbar, args.jet_radius, args.jet_min_pt, args.match_dr)})
                if len(rows) == args.events:
                    break
            if len(rows) == args.events:
                break
        if len(rows) == args.events:
            break
    if len(rows) != args.events:
        raise RuntimeError(f"Expected {args.events} H -> bb events, got {len(rows)}")
    masses = np.asarray([np.nan if row["resolved_mass"] is None else row["resolved_mass"] for row in rows])
    finite = masses[np.isfinite(masses)]
    summary = {
        "events": len(rows),
        "resolved_efficiency": float(len(finite) / len(rows)),
        "resolved_mass_mean": float(np.mean(finite)) if len(finite) else None,
        "resolved_mass_median": float(np.median(finite)) if len(finite) else None,
        "selection": {"raw_particles": True, "top_pt_particles": args.top_pt_particles},
        "clustering": {"algorithm": "antikt", "radius": args.jet_radius, "min_pt": args.jet_min_pt, "match_dr": args.match_dr},
    }
    (args.output_dir / "higgs_mass_raw_top_pt_metrics.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "higgs_mass_raw_top_pt_candidates.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["resolved_mass"])
        writer.writeheader()
        writer.writerows(rows)
    set_mpl_style()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(finite, bins=np.arange(40, 202, 2), histtype="step", density=True, label=f"Raw top-{args.top_pt_particles} $p_T$ particles")
    axis.axvline(125.0, color="black", linestyle="--", linewidth=1, label="Higgs mass")
    axis.set(xlabel="Higgs candidate mass [GeV]", ylabel="Normalized events", title="Resolved $H \to b\bar{b}$")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "higgs_mass_resolved_raw_top_pt.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
