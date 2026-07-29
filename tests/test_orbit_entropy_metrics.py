"""Tests for sparse ORBIT entropy and rate metrics."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gabbro.plotting.orbit import (
    CodeBranchSpec,
    code_entropy_metrics,
    plot_multirun_metric,
)


def test_uniform_entropy_and_length_aware_rates():
    codes = np.array([[0, 1], [2, 3]])
    latent_mask = np.ones_like(codes, dtype=bool)
    input_mask = np.ones((2, 4), dtype=bool)

    metrics = code_entropy_metrics(codes, latent_mask, input_mask, num_codes=4)

    assert metrics["metrics/entropy/marginal_bits_per_token"] == 2.0
    assert metrics["metrics/entropy/perplexity"] == 4.0
    assert metrics["metrics/entropy/normalized_to_log2_codebook"] == 1.0
    assert metrics["metrics/active_codes_total"] == 4
    assert metrics["metrics/utilization_total"] == 1.0
    assert metrics["metrics/rate/mean_latent_tokens_per_event"] == 2.0
    assert metrics["metrics/rate/effective_token_ratio"] == 0.5
    assert metrics["metrics/rate/marginal_bits_per_event"] == 4.0
    assert metrics["metrics/rate/marginal_bits_per_input_particle"] == 1.0


def test_reduced_length_changes_rate_not_token_entropy():
    input_mask = np.ones((2, 4), dtype=bool)
    full_codes = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    half_codes = np.array([[0, 1], [2, 3]])
    full = code_entropy_metrics(
        full_codes,
        np.ones_like(full_codes, dtype=bool),
        input_mask,
        num_codes=4,
    )
    half = code_entropy_metrics(
        half_codes,
        np.ones_like(half_codes, dtype=bool),
        input_mask,
        num_codes=4,
    )

    assert full["metrics/entropy/marginal_bits_per_token"] == half[
        "metrics/entropy/marginal_bits_per_token"
    ]
    assert half["metrics/rate/marginal_bits_per_event"] == (
        full["metrics/rate/marginal_bits_per_event"] / 2
    )
    assert half["metrics/rate/marginal_bits_per_input_particle"] == (
        full["metrics/rate/marginal_bits_per_input_particle"] / 2
    )


def test_masks_and_pooled_entropy_are_handled_directly():
    class_a = code_entropy_metrics(
        np.array([[0, 99]]),
        np.array([[True, False]]),
        np.array([[True, True]]),
        num_codes=2,
    )
    class_b = code_entropy_metrics(
        np.array([[1, 99]]),
        np.array([[True, False]]),
        np.array([[True, True]]),
        num_codes=2,
    )
    pooled = code_entropy_metrics(
        np.array([[0, 99], [1, 99]]),
        np.array([[True, False], [True, False]]),
        np.ones((2, 2), dtype=bool),
        num_codes=2,
    )

    assert class_a["metrics/entropy/marginal_bits_per_token"] == 0.0
    assert class_b["metrics/entropy/marginal_bits_per_token"] == 0.0
    assert pooled["metrics/entropy/marginal_bits_per_token"] == 1.0
    assert pooled["metrics/rate/sample_latent_tokens"] == 2


def test_split_branch_and_component_total_correlation():
    # mu is the first radix (4 codes from two binary FSQ dimensions), while
    # alpha is the second radix (2 VQ codes). The two branches are identical.
    codes = np.array([[0, 5]])
    metrics = code_entropy_metrics(
        codes,
        np.ones_like(codes, dtype=bool),
        np.ones_like(codes, dtype=bool),
        num_codes=8,
        branch_specs=(
            CodeBranchSpec(name="mu", num_codes=4, levels=(2, 2)),
            CodeBranchSpec(name="alpha", num_codes=2),
        ),
    )

    assert metrics["metrics/entropy/marginal_bits_per_token"] == 1.0
    assert metrics["metrics/entropy/branch/mu/bits_per_token"] == 1.0
    assert metrics["metrics/entropy/branch/alpha/bits_per_token"] == 1.0
    assert metrics["metrics/entropy/branch_total_correlation_bits_per_token"] == 1.0
    assert metrics["metrics/entropy/component/mu/dim_0/bits_per_token"] == 1.0
    assert metrics["metrics/entropy/component/mu/dim_1/bits_per_token"] == 0.0
    assert metrics["metrics/entropy/component_total_correlation_bits_per_token"] == 1.0


def test_large_nominal_codebook_uses_sparse_counts():
    metrics = code_entropy_metrics(
        np.array([[0, 63_999_999]]),
        np.array([[True, True]]),
        np.array([[True, True]]),
        num_codes=64_000_000,
    )
    assert metrics["metrics/active_codes_total"] == 2
    assert metrics["metrics/entropy/marginal_bits_per_token"] == 1.0


def test_entropy_metrics_reject_inconsistent_fsq_morphology():
    with pytest.raises(ValueError, match="levels represent"):
        code_entropy_metrics(
            code_idx=np.array([[0, 1]]),
            code_mask=np.ones((1, 2), dtype=bool),
            input_mask=np.ones((1, 2), dtype=bool),
            num_codes=8,
            branch_specs=(CodeBranchSpec("mu", 8, levels=(2, 2)),),
        )


def test_multirun_metric_supports_rate_axis_and_reference():
    pytest.importorskip("mplhep")
    records = [
        {
            "label": "small",
            "plot_family": "FSQ",
            "total_codebook_size": 16,
            "rate": 2.0,
            "error": 0.1,
        },
        {
            "label": "large",
            "plot_family": "FSQ",
            "total_codebook_size": 256,
            "rate": 4.0,
            "error": 0.05,
        },
    ]
    figure = plot_multirun_metric(
        records,
        "error",
        "MSE",
        "Rate distortion",
        x_metric="rate",
        xlabel="Bits/input particle",
        reference_fn=lambda values: values / 10,
        reference_label="reference",
    )

    assert figure is not None
    assert figure.axes[0].get_xlabel() == "Bits/input particle"
    assert len(figure.axes[0].lines) == 2  # Family line and reference line.
    plt.close(figure)
