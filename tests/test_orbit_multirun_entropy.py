"""Tests for entropy and rate-distortion multirun outputs."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mplhep")

from scripts.collect_orbit_multirun import _save_figures
from scripts.collect_orbit_multirun import _quantizer_metadata
from scripts.collect_orbit_multirun import _wandb_upload_settings
from scripts.collect_orbit_multirun import _artifact_prefix
from gabbro.plotting.orbit import plot_multirun_metric


def test_entropy_multirun_figures_are_written(tmp_path):
    records = []
    for label, codebook_size, entropy, rate, error in (
        ("small", 16, 3.0, 1.5, 0.10),
        ("large", 256, 6.0, 3.0, 0.05),
    ):
        records.append(
            {
                "label": label,
                "plot_family": "FSQ",
                "total_codebook_size": codebook_size,
                "metrics/entropy/marginal_bits_per_token": entropy,
                "metrics/entropy/normalized_to_log2_codebook": 0.75,
                "metrics/rate/marginal_bits_per_event": rate * 20,
                "metrics/rate/marginal_bits_per_input_particle": rate,
                "plot_metrics/reco_mse_total": error,
            }
        )

    _save_figures(records, histogram_runs=[], output_dir=tmp_path)

    expected = {
        "codebook_size_vs_marginal_entropy_bits_per_token.png",
        "codebook_size_vs_normalized_entropy.png",
        "codebook_size_vs_marginal_bits_per_event.png",
        "codebook_size_vs_marginal_bits_per_input_particle.png",
        "marginal_bits_per_input_particle_vs_reco_mse.png",
        "marginal_bits_per_event_vs_reco_mse.png",
    }
    assert expected.issubset({path.name for path in tmp_path.glob("*.png")})


def test_faiss_kmeans_multirun_metadata():
    metadata = _quantizer_metadata(
        {
            "model": {
                "_target_": "gabbro.models.vqvae.FaissKMeansBaselineLightning",
                "num_codes": 8192,
            }
        }
    )

    assert metadata["quantizer_mode"] == "kmeans"
    assert metadata["quantizer_family"] == "kmeans"
    assert metadata["total_codebook_size"] == 8192


def test_named_test_suite_artifact_prefix():
    assert _artifact_prefix("test", "all", "training_like") == "test_training_like_all_orbit"
    assert _artifact_prefix("test", "tt", "tt_vs_gghbb") == "test_tt_vs_gghbb_tt_orbit"


def test_multirun_metric_reference_lines():
    figure = plot_multirun_metric(
        [{"label": "FSQ", "rate": 4.0, "mse": 0.1}],
        "mse",
        "MSE",
        "Rate distortion",
        x_metric="rate",
        vertical_reference=43.0,
        vertical_reference_label="Original particle representation (43 bits)",
        horizontal_reference=0.01,
        horizontal_reference_label="Continuous autoencoder",
    )

    labelled_lines = {line.get_label(): line for line in figure.axes[0].lines}
    assert np.all(
        np.asarray(
            labelled_lines["Original particle representation (43 bits)"].get_xdata()
        )
        == 43
    )
    assert np.all(
        np.asarray(labelled_lines["Continuous autoencoder"].get_ydata()) == 0.01
    )


def test_multirun_wandb_upload_uses_detected_credentials(monkeypatch, tmp_path):
    args = SimpleNamespace(
        no_wandb=False,
        wandb_project=None,
        wandb_name=None,
        wandb_group=None,
        wandb_entity=None,
    )
    records = [{"wandb_project": "orbit-tokenizer", "run_dir": tmp_path / "run"}]
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert _wandb_upload_settings(args, records, tmp_path / "comparison") is None

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    settings = _wandb_upload_settings(args, records, tmp_path / "comparison")

    assert settings == {
        "project": "orbit-tokenizer",
        "name": "orbit-multirun-comparison",
        "group": "orbit-multirun-comparisons",
        "entity": "test-entity",
    }
