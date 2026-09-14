"""Tests for toggleable eight-class ORBIT particle PID support."""

from functools import partial
from types import SimpleNamespace

import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from gabbro.data.orbit_parquet import (
    OrbitParquetDataset,
    map_pdg_charge_to_pid_class,
)
from gabbro.models.vqvae import DumbQuantizationBaselineLightning, VQVAELightning
from gabbro.callbacks.orbit_plotting_callback import OrbitPlottingCallback
from gabbro.plotting.orbit import pid_residual_arrays, plot_pid_conditional_residuals
from gabbro.train import _load_feature_expanded_weights


def test_pid_mapping_uses_pdg_for_leptons_and_charge_for_hadrons():
    pdg = ak.Array([[130, 22, -211, 211, 11, -11, 13, -13, 321, -2212]])
    charge = ak.Array([[0, 0, -1, 1, -1, 1, -1, 1, 1, -1]])

    mapped = map_pdg_charge_to_pid_class(pdg, charge)

    assert ak.to_list(mapped) == [[0, 1, 2, 3, 4, 5, 6, 7, 3, 2]]


def _write_particle_fixture(path):
    pq.write_table(
        pa.table(
            {
                "L1T_PUPPIPart_Eta": [[0.1, 0.2, 0.3]],
                "L1T_PUPPIPart_Phi": [[0.0, 1.0, 2.0]],
                "L1T_PUPPIPart_PT": [[10.0, 20.0, 30.0]],
                "L1T_PUPPIPart_E": [[11.0, 22.0, 33.0]],
                "L1T_PUPPIPart_PuppiW": [[1.0, 0.01, 1.0]],
                "L1T_PUPPIPart_PID": [[22, 11, 321]],
                "L1T_PUPPIPart_Charge": [[0, -1, 1]],
            }
        ),
        path,
    )


def test_pid_loader_toggle_and_selection_alignment(tmp_path):
    path = tmp_path / "particles.parquet"
    _write_particle_fixture(path)

    enabled = next(
        iter(
            OrbitParquetDataset(
                [path],
                batch_size=1,
                max_sequence_length=4,
                pid_cfg={"enabled": True, "num_classes": 8},
            )
        )
    )
    assert enabled["part_features"].shape == (1, 4, 4)
    assert enabled["part_pid"].tolist() == [[1, 3, -1, -1]]
    assert enabled["part_mask"].tolist() == [[True, True, False, False]]

    disabled = next(
        iter(
            OrbitParquetDataset(
                [path],
                batch_size=1,
                max_sequence_length=4,
                pid_cfg={"enabled": False},
            )
        )
    )
    assert "part_pid" not in disabled
    assert disabled["part_features"].shape == (1, 4, 4)


def test_optional_log_energy_feature_is_centered_and_aligned(tmp_path):
    path = tmp_path / "particles.parquet"
    _write_particle_fixture(path)

    batch = next(
        iter(
            OrbitParquetDataset(
                [path],
                batch_size=1,
                max_sequence_length=4,
                include_energy=True,
                energy_shift=2.5,
            )
        )
    )

    assert batch["part_features"].shape == (1, 4, 5)
    assert torch.allclose(
        batch["part_features"][0, :2, 4],
        torch.log(torch.tensor([11.0, 33.0])) - 2.5,
    )


def test_particle_full_keeps_zero_weight_candidates_and_clips_at_500(tmp_path):
    path = tmp_path / "full_particles.parquet"
    particle_count = 600
    pt = [float(particle_count - index) for index in range(particle_count)]
    pq.write_table(
        pa.table(
            {
                "L1T_PUPPIPart_Eta": [[index / 1000 for index in range(particle_count)]],
                "L1T_PUPPIPart_Phi": [[0.0] * particle_count],
                "L1T_PUPPIPart_PT": [pt],
                "L1T_PUPPIPart_PuppiW": [[0.0] * particle_count],
                "L1T_PUPPIPart_PID": [[22] * particle_count],
                "L1T_PUPPIPart_Charge": [[0] * particle_count],
            }
        ),
        path,
    )

    batch = next(
        iter(
            OrbitParquetDataset(
                [path],
                sequence_type="particle_full",
                batch_size=1,
                return_raw_features=True,
                return_event_metadata=True,
                pid_cfg={"enabled": True, "num_classes": 8},
            )
        )
    )

    assert batch["part_features"].shape == (1, 500, 4)
    assert batch["part_mask"].all()
    assert batch["part_pid"].tolist() == [[1] * 500]
    assert len(batch["raw_part_features"][0]) == 500
    assert batch["raw_part_features"][0, 0, 0] == 600.0
    assert batch["raw_part_features"][0, -1, 0] == 101.0
    assert batch["source_rows"].tolist() == [0]


def _continuous_module(pid_enabled):
    return VQVAELightning(
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        model_type="VQVAETransformer",
        model_kwargs={
            "input_dim": 4,
            "hidden_dim": 8,
            "latent_dim": 4,
            "num_heads": 2,
            "num_blocks": 1,
            "alpha": 1.0,
            "quantization_enabled": False,
            "pid_cfg": {
                "enabled": pid_enabled,
                "num_classes": 8,
                "loss_weight": 1.0,
            },
        },
    )


def test_pid_head_and_combined_loss_are_toggleable():
    features = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    batch = {
        "part_features": features,
        "part_mask": mask,
        "part_pid": torch.tensor([[1, 3, -1], [6, -1, -1]]),
        "jet_type_labels": torch.zeros(2, dtype=torch.long),
    }

    enabled = _continuous_module(True)
    loss, metrics = enabled.model_step(batch)
    assert enabled.model.input_projection.in_features == 12
    assert enabled.model.pid_output_projection.out_features == 8
    assert torch.isfinite(loss)
    assert metrics["loss_pid"] > 0
    assert "pid_macro_recall" in metrics
    assert "pid_confusion_matrix" in metrics
    assert metrics["pid_confusion_matrix"].shape == (8, 8)
    assert not any(name.startswith("pid_confusion/") for name in metrics)
    assert torch.allclose(
        metrics["loss_total"],
        metrics["loss_reco"] + metrics["loss_pid_weighted"],
    )

    disabled = _continuous_module(False)
    disabled_batch = {key: value for key, value in batch.items() if key != "part_pid"}
    disabled_loss, disabled_metrics = disabled.model_step(disabled_batch)
    assert disabled.model.input_projection.in_features == 4
    assert disabled.model.pid_output_projection is None
    assert torch.isfinite(disabled_loss)
    assert disabled_metrics["loss_pid"] == 0


def test_pid_weights_apply_only_to_reconstruction_loss():
    module = _continuous_module(True)
    module.model.register_buffer(
        "reconstruction_class_weights",
        torch.tensor([1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]),
    )
    features = torch.zeros(1, 2, 4)
    reconstructed = torch.tensor([[[1.0] * 4, [2.0] * 4]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    pid = torch.tensor([[0, 3]])
    pid_logits = torch.zeros(1, 2, 8)

    def forward(**_):
        return reconstructed, {
            "q": torch.zeros(1, 2, dtype=torch.long),
            "pid_logits": pid_logits,
            "quantization_bypassed": True,
        }

    module.forward = forward
    _, metrics = module.model_step(
        {
            "part_features": features,
            "part_mask": mask,
            "part_pid": pid,
            "jet_type_labels": torch.zeros(1, dtype=torch.long),
        }
    )

    assert torch.allclose(metrics["loss_reco_l2"], torch.tensor(10.0))
    assert torch.allclose(metrics["loss_reco_l2_weighted"], torch.tensor(13.0))
    assert torch.allclose(metrics["loss_reco"], torch.tensor(13.0))


def test_feature_expansion_warm_start_preserves_common_and_pid_columns():
    source = _continuous_module(True)
    target = VQVAELightning(
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        model_type="VQVAETransformer",
        model_kwargs={
            "input_dim": 5,
            "hidden_dim": 8,
            "latent_dim": 4,
            "num_heads": 2,
            "num_blocks": 1,
            "alpha": 1.0,
            "quantization_enabled": False,
            "pid_cfg": {
                "enabled": True,
                "num_classes": 8,
                "loss_weight": 1.0,
                "reconstruction_class_weights": [1.0] * 8,
            },
        },
    )
    with torch.no_grad():
        source.model.input_projection.weight.copy_(
            torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12)
        )
        source.model.output_projection.weight.copy_(
            torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
        )
        source.model.output_projection.bias.copy_(torch.arange(4, dtype=torch.float32))
    source_features = [
        "L1T_PUPPIPart_Eta",
        "L1T_PUPPIPart_Phi_cos",
        "L1T_PUPPIPart_Phi_sin",
        "L1T_PUPPIPart_PT",
    ]
    target_features = source_features + ["L1T_PUPPIPart_E"]
    source_cfg = OmegaConf.create(
        {"feature_dict": {name: {} for name in source_features}, "pid": {"enabled": True, "num_classes": 8}}
    )
    target_cfg = OmegaConf.create(
        {"feature_dict": {name: {} for name in target_features}, "pid": {"enabled": True, "num_classes": 8}}
    )
    expansion_cfg = OmegaConf.create(
        {
            "output_initializers": {
                "L1T_PUPPIPart_E": {
                    "copy_from": "L1T_PUPPIPart_PT",
                    "bias_offset": -0.7,
                }
            }
        }
    )

    _load_feature_expanded_weights(
        target, source.state_dict(), source_cfg, target_cfg, expansion_cfg
    )

    assert torch.equal(
        target.model.input_projection.weight[:, :4],
        source.model.input_projection.weight[:, :4],
    )
    assert torch.count_nonzero(target.model.input_projection.weight[:, 4]) == 0
    assert torch.equal(
        target.model.input_projection.weight[:, 5:13],
        source.model.input_projection.weight[:, 4:12],
    )
    assert torch.equal(
        target.model.output_projection.weight[4],
        source.model.output_projection.weight[3],
    )
    assert torch.allclose(
        target.model.output_projection.bias[4],
        source.model.output_projection.bias[3] - 0.7,
    )


def test_pid_residual_plot_adds_energy_panel():
    feature_names = [
        "L1T_PUPPIPart_Eta",
        "L1T_PUPPIPart_Phi_cos",
        "L1T_PUPPIPart_Phi_sin",
        "L1T_PUPPIPart_PT",
        "L1T_PUPPIPart_E",
    ]
    original = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]).numpy()
    reconstructed = torch.tensor([[0.1, 1.0, 0.0, 0.2, -0.3]]).numpy()
    residuals = pid_residual_arrays(original, reconstructed, feature_names)
    by_pid = {
        index: {
            key: value if index == 0 else value[:0]
            for key, value in residuals.items()
        }
        for index in range(8)
    }

    figure = plot_pid_conditional_residuals(by_pid, [f"class_{i}" for i in range(8)])

    assert set(residuals) == {
        "delta_eta",
        "delta_phi",
        "delta_log_pt",
        "delta_log_energy",
    }
    assert len(figure.axes) == 4


def test_pid_residual_artifacts_are_written_once_for_a_suite(tmp_path):
    feature_names = [
        "L1T_PUPPIPart_Eta",
        "L1T_PUPPIPart_Phi_cos",
        "L1T_PUPPIPart_Phi_sin",
        "L1T_PUPPIPart_PT",
        "L1T_PUPPIPart_E",
    ]
    trainer = SimpleNamespace(
        default_root_dir=str(tmp_path),
        global_step=5,
        loggers=[],
        datamodule=SimpleNamespace(
            hparams=OmegaConf.create({"selected_features": feature_names})
        ),
    )
    module = SimpleNamespace(
        test_pid_concat=torch.tensor([[0, 6]]).numpy(),
        model=SimpleNamespace(pid_class_names=tuple(f"class_{i}" for i in range(8))),
    )
    original = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0, 0.0], [0.1, 0.0, 1.0, 0.2, 0.3]]]
    ).numpy()
    reconstructed = original + 0.01
    callback = OrbitPlottingCallback(enable_physics_plots=False)

    callback._plot_pid_residuals_for_suite(
        trainer,
        module,
        "training_like",
        torch.tensor([True]).numpy(),
        original,
        reconstructed,
        torch.tensor([[True, True]]).numpy(),
    )

    assert len(list((tmp_path / "plots").glob("*pid_conditional_residuals.png"))) == 1
    assert len(list((tmp_path / "saved_metrics").glob("*pid_conditional_residuals*.json"))) == 1
    assert len(list((tmp_path / "saved_histograms").glob("*pid_conditional_residuals*.npz"))) == 1


def test_scalar_baseline_packs_pid_into_token_id():
    baseline = DumbQuantizationBaselineLightning(
        q_levels=[2, 2, 2],
        pt_range=(-1.0, 1.0),
        pid_cfg={"enabled": True, "num_classes": 8},
    )
    features = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    mask = torch.tensor([[True]])
    pid = torch.tensor([[7]])

    _, code_idx, pid_logits = baseline._quantize_batch(features, mask, pid)

    assert baseline.model.vqlayer.num_codes == 64
    assert 0 <= code_idx.item() < 64
    assert code_idx.item() % 8 == 7
    assert pid_logits.argmax(dim=-1).item() == 7
