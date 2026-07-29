import torch

from gabbro.models.latent_sequence import DirectPrefixLatentMasker, LatentSequenceCompressor


def test_latent_sequence_compressor_uses_per_event_ceil_lengths():
    compressor = LatentSequenceCompressor(
        latent_dim=8,
        max_sequence_len=8,
        ratio=0.5,
        min_tokens=1,
        num_heads=2,
    )
    z = torch.randn(3, 8, 8)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    z_latent, latent_mask = compressor.compress(z, mask)

    assert z_latent.shape == (3, 4, 8)
    assert latent_mask.sum(dim=1).tolist() == [4, 1, 0]


def test_latent_sequence_compressor_expands_to_particle_length():
    compressor = LatentSequenceCompressor(
        latent_dim=8,
        max_sequence_len=8,
        ratio=0.5,
        min_tokens=1,
        num_heads=2,
    )
    z_latent = torch.randn(2, 4, 8)
    latent_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    particle_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    z_full = compressor.expand(
        z_latent,
        latent_mask,
        target_len=6,
        particle_mask=particle_mask,
    )

    assert z_full.shape == (2, 6, 8)
    assert torch.all(z_full[~particle_mask] == 0)


def test_query_residual_preserves_queries_when_attention_is_zero():
    with_residual = LatentSequenceCompressor(
        latent_dim=8,
        max_sequence_len=6,
        ratio=0.5,
        num_heads=2,
        query_residual=True,
    )
    without_residual = LatentSequenceCompressor(
        latent_dim=8,
        max_sequence_len=6,
        ratio=0.5,
        num_heads=2,
        query_residual=False,
    )
    without_residual.load_state_dict(with_residual.state_dict())
    for compressor in (with_residual, without_residual):
        for attention in (compressor.compress_attn, compressor.expand_attn):
            for parameter in attention.parameters():
                torch.nn.init.zeros_(parameter)

    z = torch.randn(1, 6, 8)
    particle_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    compressed_with, latent_mask = with_residual.compress(z, particle_mask)
    compressed_without, _ = without_residual.compress(z, particle_mask)

    assert torch.any(compressed_with[latent_mask] != 0)
    assert torch.all(compressed_without == 0)
    assert torch.all(compressed_with[~latent_mask] == 0)

    expanded_with = with_residual.expand(
        compressed_with,
        latent_mask,
        target_len=6,
        particle_mask=particle_mask,
    )
    expanded_without = without_residual.expand(
        compressed_with,
        latent_mask,
        target_len=6,
        particle_mask=particle_mask,
    )

    assert torch.any(expanded_with[particle_mask] != 0)
    assert torch.all(expanded_without == 0)
    assert torch.all(expanded_with[~particle_mask] == 0)


def test_direct_prefix_masking_uses_per_event_ceil_lengths_and_zeros_tail():
    masker = DirectPrefixLatentMasker(
        latent_dim=3,
        max_sequence_len=7,
        ratio=0.5,
        min_tokens=1,
    )
    z = torch.arange(4 * 7 * 3, dtype=torch.float32).reshape(4, 7, 3) + 1
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    z_prefix, latent_mask = masker.compress(z, mask)

    assert latent_mask.sum(dim=1).tolist() == [4, 2, 1, 0]
    assert torch.equal(z_prefix[latent_mask], z[latent_mask])
    assert torch.all(z_prefix[~latent_mask] == 0)


def test_direct_prefix_masking_expands_with_position_specific_placeholders():
    masker = DirectPrefixLatentMasker(
        latent_dim=3,
        max_sequence_len=6,
        ratio=0.5,
    )
    z = torch.randn(2, 6, 3)
    particle_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool
    )
    z_prefix, latent_mask = masker.compress(z, particle_mask)
    z_quantized = z_prefix.clone()
    z_quantized[~latent_mask] = 1e6

    z_expanded = masker.expand(
        z_quantized,
        latent_mask,
        target_len=6,
        particle_mask=particle_mask,
    )

    dropped_mask = particle_mask & ~latent_mask
    expected_placeholders = masker.mask_embeddings.unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(z_expanded[latent_mask], z[latent_mask])
    assert torch.equal(z_expanded[dropped_mask], expected_placeholders[dropped_mask])
    assert torch.all(z_expanded[~particle_mask] == 0)


def test_direct_prefix_ratio_one_is_identity_on_valid_tokens():
    masker = DirectPrefixLatentMasker(
        latent_dim=4,
        max_sequence_len=5,
        ratio=1.0,
    )
    z = torch.randn(2, 5, 4)
    particle_mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.bool
    )

    z_prefix, latent_mask = masker.compress(z, particle_mask)
    z_expanded = masker.expand(z_prefix, latent_mask, 5, particle_mask)

    assert torch.equal(latent_mask, particle_mask)
    assert torch.equal(z_prefix, z * particle_mask.unsqueeze(-1))
    assert torch.equal(z_expanded, z_prefix)


def test_direct_prefix_masker_initialization_does_not_advance_global_rng():
    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state()

    DirectPrefixLatentMasker(latent_dim=4, max_sequence_len=5, ratio=0.5)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
