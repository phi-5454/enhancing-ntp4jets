"""Tests for one-to-one delta-R matching in ORBIT physics comparisons."""

import numpy as np
import matplotlib.pyplot as plt

from gabbro.callbacks.orbit_plotting_callback import (
    OrbitPlottingCallback,
    match_jets_by_delta_r,
)
from gabbro.plotting.orbit import (
    collect_physical_reconstruction_histograms,
    physical_reconstruction_plots,
)


def test_delta_r_matching_is_one_to_one_and_wraps_phi():
    true_eta = np.array([0.0, 1.0, 3.0])
    true_phi = np.array([np.pi - 0.02, 0.5, 0.0])
    reco_eta = np.array([1.03, 0.01, -3.0])
    reco_phi = np.array([0.48, -np.pi + 0.02, 0.0])

    true_idx, reco_idx, delta_r = match_jets_by_delta_r(
        true_eta,
        true_phi,
        reco_eta,
        reco_phi,
        max_delta_r=0.2,
    )

    assert list(zip(true_idx, reco_idx)) == [(0, 1), (1, 0)]
    assert np.all(delta_r < 0.06)


def test_direct_jet_metrics_use_geometry_instead_of_sequence_order():
    callback = OrbitPlottingCallback(jet_matching_radius_fraction=0.5)
    original = np.array([[[0.0, 0.0, 100.0], [1.0, 1.0, 50.0]]])
    reconstructed = np.array([[[1.02, 0.98, 48.0], [0.01, 0.02, 95.0]]])
    mask = np.ones((1, 2), dtype=bool)

    (
        true_pt,
        reco_pt,
        true_eta,
        reco_eta,
        _,
        _,
        all_true_pt,
        all_reco_pt,
        _,
        _,
        _,
        _,
        stats,
    ) = callback._collect_direct_jet_metrics(
        original,
        reconstructed,
        mask,
        jet_radii=np.array([0.8]),
    )

    assert np.allclose(true_pt, [100.0, 50.0])
    assert np.allclose(reco_pt, [95.0, 48.0])
    assert np.allclose(true_eta, [0.0, 1.0])
    assert np.allclose(reco_eta, [0.01, 1.02])
    assert np.allclose(all_true_pt, true_pt)
    assert np.allclose(all_reco_pt, reco_pt)
    assert stats["matched"] == 2


def test_particle_reclustering_returns_matched_per_jet_mass_and_tau32():
    callback = OrbitPlottingCallback(jet_matching_radius_fraction=0.5)
    original = np.array(
        [
            [
                [0.00, 0.00, 100.0],
                [0.10, 0.05, 40.0],
                [-0.08, -0.04, 20.0],
                [2.00, 2.00, 80.0],
                [2.10, 2.05, 30.0],
                [1.92, 1.96, 15.0],
            ]
        ]
    )
    reconstructed = original.copy()
    reconstructed[0, :, 0] += 0.01
    reconstructed[0, :, 1] -= 0.01
    reconstructed[0, :, 2] *= 0.95
    mask = np.ones((1, original.shape[1]), dtype=bool)

    (
        true_pt,
        reco_pt,
        true_mass,
        reco_mass,
        true_tau32,
        reco_tau32,
        all_true_pt,
        all_reco_pt,
        all_true_mass,
        all_reco_mass,
        all_true_tau32,
        all_reco_tau32,
        true_count,
        reco_count,
        stats,
    ) = callback._collect_particle_jet_metrics(
        original,
        reconstructed,
        mask,
        jet_radii=np.array([0.8]),
    )

    assert true_count == reco_count == [2]
    assert len(true_pt) == len(reco_pt) == 2
    assert len(true_mass) == len(reco_mass) == 2
    assert len(true_tau32) == len(reco_tau32) == 2
    assert len(all_true_pt) == len(all_reco_pt) == 2
    assert len(all_true_mass) == len(all_reco_mass) == 2
    assert len(all_true_tau32) == len(all_reco_tau32) == 2
    assert stats["matched"] == 2


def test_unfiltered_assignment_keeps_pairs_outside_matching_radius():
    callback = OrbitPlottingCallback(jet_matching_radius_fraction=0.5)
    original = np.array([[[0.0, 0.0, 100.0], [1.0, 1.0, 50.0]]])
    reconstructed = np.array([[[0.02, 0.01, 95.0], [-2.0, -2.0, 45.0]]])
    mask = np.ones((1, 2), dtype=bool)

    result = callback._collect_direct_jet_metrics(
        original,
        reconstructed,
        mask,
        jet_radii=np.array([0.8]),
    )

    filtered_true_pt, filtered_reco_pt = result[:2]
    unfiltered_true_pt, unfiltered_reco_pt = result[6:8]
    stats = result[-1]
    assert len(filtered_true_pt) == len(filtered_reco_pt) == 1
    assert len(unfiltered_true_pt) == len(unfiltered_reco_pt) == 2
    assert stats["matched"] == 1


def test_residual_plots_include_cutoff_and_unfiltered_assignments():
    features = np.array([[0.0, 0.0, 10.0]])
    histograms = collect_physical_reconstruction_histograms(
        ["Eta", "Phi", "pT"],
        features,
        features,
        true_jet_pts=[100.0],
        reco_jet_pts=[95.0],
        true_jet_masses=[20.0],
        reco_jet_masses=[19.0],
        true_tau32s=[0.5],
        reco_tau32s=[0.45],
        unfiltered_true_jet_pts=[100.0, 50.0],
        unfiltered_reco_jet_pts=[95.0, 5.0],
        unfiltered_true_jet_masses=[20.0, 10.0],
        unfiltered_reco_jet_masses=[19.0, 1.0],
        unfiltered_true_tau32s=[0.5, 0.4],
        unfiltered_reco_tau32s=[0.45, 0.05],
        data_level="particle",
    )

    assert "jet_pt_resolution_counts" in histograms
    assert "jet_pt_resolution_unfiltered_counts" in histograms
    assert "jet_mass_diff_counts" in histograms
    assert "jet_mass_diff_unfiltered_counts" in histograms
    figures = physical_reconstruction_plots(
        ["Eta", "Phi", "pT"],
        np.zeros(3),
        histograms,
        data_level="particle",
        jet_matching_cut_label=r"$\Delta R \leq 0.5R$",
    )
    assert len(figures["jet_pt_resolution"].axes[0].get_legend().texts) == 2
    assert len(figures["jet_substructure"].axes[1].get_legend().texts) == 2
    for figure in figures.values():
        plt.close(figure)
