"""Tests for one-to-one delta-R matching in ORBIT physics comparisons."""

from types import SimpleNamespace

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


def test_validation_plot_cadence_is_one_indexed(monkeypatch):
    callback = OrbitPlottingCallback(validation_plot_every_n_epochs=5)
    plotted_epochs = []
    concatenated_epochs = []
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=0)
    module = SimpleNamespace(
        concat_validation_loop_predictions=lambda: concatenated_epochs.append(
            trainer.current_epoch
        )
    )
    monkeypatch.setattr(
        callback,
        "plot",
        lambda trainer, _module, stage: plotted_epochs.append((trainer.current_epoch, stage)),
    )

    for epoch in range(10):
        trainer.current_epoch = epoch
        callback.on_validation_epoch_end(trainer, module)

    assert concatenated_epochs == [4, 9]
    assert plotted_epochs == [(4, "val"), (9, "val")]


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


def test_particle_reclustering_applies_pt_cut_and_skips_tau32():
    callback = OrbitPlottingCallback(jet_min_pt=30.0, include_tau32=False)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("tau32 calculation must be skipped")

    callback._calculate_tau32 = fail_if_called
    original = np.array([[[0.0, 0.0, 40.0], [3.0, 0.0, 25.0]]])
    reconstructed = original.copy()
    mask = np.ones((1, 2), dtype=bool)

    result = callback._collect_particle_jet_metrics(
        original,
        reconstructed,
        mask,
        jet_radii=np.array([0.8]),
    )

    assert result[12] == result[13] == [1]
    assert np.allclose(result[0], [40.0])
    assert np.allclose(result[1], [40.0])
    assert result[4] == result[5] == []
    assert result[10] == result[11] == []
    assert result[-1]["matched"] == 1


def test_cached_event_results_can_be_reused_for_subsets_and_all(monkeypatch):
    callback = OrbitPlottingCallback(include_tau32=False)
    original = np.array(
        [
            [[0.0, 0.0, 40.0], [2.0, 0.0, 20.0]],
            [[0.5, 1.0, 60.0], [-2.0, 0.0, 30.0]],
        ]
    )
    reconstructed = original.copy()
    mask = np.ones((2, 2), dtype=bool)
    reconstruction_calls = 0
    reconstruct = callback._reconstruct_event_jets

    def counted_reconstruct(*args, **kwargs):
        nonlocal reconstruction_calls
        reconstruction_calls += 1
        return reconstruct(*args, **kwargs)

    monkeypatch.setattr(callback, "_reconstruct_event_jets", counted_reconstruct)
    event_results = callback._evaluate_particle_event_batch(
        original,
        reconstructed,
        mask,
        jet_radii=np.array([0.8, 0.8]),
    )
    first = callback._combine_particle_event_jet_results(event_results[:1])
    second = callback._combine_particle_event_jet_results(event_results[1:])
    combined = callback._combine_particle_event_jet_results(event_results)

    assert reconstruction_calls == 4
    assert len(combined[0]) == len(first[0]) + len(second[0])
    assert combined[-1]["matched"] == first[-1]["matched"] + second[-1]["matched"]


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


def test_substructure_plot_omits_tau32_panel_when_tau32_is_absent():
    features = np.array([[0.0, 0.0, 10.0]])
    histograms = collect_physical_reconstruction_histograms(
        ["Eta", "Phi", "pT"],
        features,
        features,
        true_jet_pts=[100.0],
        reco_jet_pts=[95.0],
        true_jet_masses=[20.0],
        reco_jet_masses=[19.0],
        data_level="particle",
    )

    assert "tau32_diff_counts" not in histograms
    figures = physical_reconstruction_plots(
        ["Eta", "Phi", "pT"],
        np.zeros(3),
        histograms,
        data_level="particle",
    )
    assert len(figures["jet_substructure"].axes) == 2
    for figure in figures.values():
        plt.close(figure)
