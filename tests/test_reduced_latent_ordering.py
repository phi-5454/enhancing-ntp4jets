import torch

from gabbro.models.vqvae import VQVAETransformer


def test_quantization_bypass_preserves_compressed_latent_shapes():
    model = VQVAETransformer(
        latent_dim=4,
        hidden_dim=8,
        input_dim=4,
        num_heads=2,
        num_blocks=1,
        max_sequence_len=6,
        quantization_enabled=False,
        latent_sequence_compression={
            "enabled": True,
            "ratio": 0.5,
            "min_tokens": 1,
            "rounding": "ceil",
            "mode": "learned_cross_attention",
            "num_heads": 2,
        },
    )
    x = torch.randn(2, 6, 4)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool
    )

    x_reco, vq_out = model(x, mask)

    assert model.vqlayer is None
    assert x_reco.shape == x.shape
    assert vq_out["quantization_bypassed"] is True
    assert vq_out["latent_mask"].sum(dim=1).tolist() == [2, 2]
    assert torch.all(vq_out["q"] == -1)
    assert vq_out["loss"].item() == 0.0


def test_direct_prefix_bypass_uses_only_retained_latents():
    model = VQVAETransformer(
        latent_dim=4,
        hidden_dim=8,
        input_dim=4,
        num_heads=2,
        num_blocks=1,
        max_sequence_len=6,
        quantization_enabled=False,
        latent_sequence_compression={
            "enabled": True,
            "ratio": 0.5,
            "min_tokens": 1,
            "rounding": "ceil",
            "mode": "direct_prefix_masking",
        },
    )
    x = torch.randn(2, 6, 4)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool
    )

    x_reco, vq_out = model(x, mask)

    assert x_reco.shape == x.shape
    assert vq_out["latent_mask"].sum(dim=1).tolist() == [3, 1]
    assert torch.all(vq_out["q"] == -1)


def test_direct_prefix_ratio_one_matches_uncompressed_model():
    model_kwargs = {
        "latent_dim": 4,
        "hidden_dim": 8,
        "input_dim": 4,
        "num_heads": 2,
        "num_blocks": 1,
        "max_sequence_len": 6,
        "quantization_enabled": False,
    }
    torch.manual_seed(19)
    direct_model = VQVAETransformer(**model_kwargs)
    torch.manual_seed(19)
    prefix_model = VQVAETransformer(
        **model_kwargs,
        latent_sequence_compression={
            "enabled": True,
            "ratio": 1.0,
            "min_tokens": 1,
            "rounding": "ceil",
            "mode": "direct_prefix_masking",
        },
    )
    x = torch.randn(2, 6, 4)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool
    )

    direct_reco, _ = direct_model(x, mask)
    prefix_reco, prefix_out = prefix_model(x, mask)

    assert torch.equal(prefix_out["latent_mask"], mask)
    assert torch.equal(prefix_reco, direct_reco)
