"""Tests for toggleable eight-class ORBIT particle PID support."""

from functools import partial

import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gabbro.data.orbit_parquet import (
    OrbitParquetDataset,
    map_pdg_charge_to_pid_class,
)
from gabbro.models.vqvae import DumbQuantizationBaselineLightning, VQVAELightning


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
