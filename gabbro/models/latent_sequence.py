"""Latent sequence compression modules for particle autoencoders."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DirectPrefixLatentMasker(nn.Module):
    """Retain a prefix of encoder latents and fill dropped decoder positions."""

    mode = "direct_prefix_masking"

    def __init__(
        self,
        latent_dim: int,
        max_sequence_len: int,
        ratio: float = 0.5,
        min_tokens: int = 1,
        rounding: str = "ceil",
    ):
        super().__init__()
        if not 0 < ratio <= 1:
            raise ValueError(
                "direct prefix latent sequence compression ratio must satisfy "
                f"0 < ratio <= 1, got {ratio}"
            )
        if min_tokens < 0:
            raise ValueError(f"min_tokens must be >= 0, got {min_tokens}")
        if rounding != "ceil":
            raise ValueError(f"Only rounding='ceil' is currently supported, got {rounding!r}")

        self.latent_dim = int(latent_dim)
        self.max_sequence_len = int(max_sequence_len)
        self.ratio = float(ratio)
        self.min_tokens = int(min_tokens)
        self.rounding = rounding
        self.query_residual = False

        # A local generator keeps ratio-one controls from shifting initialization
        # of the layers that follow this module.
        generator = torch.Generator()
        generator.manual_seed(0)
        mask_embeddings = torch.randn(
            self.max_sequence_len,
            self.latent_dim,
            generator=generator,
        ) * 0.02
        self.mask_embeddings = nn.Parameter(mask_embeddings)

    def _latent_lengths(self, particle_mask: torch.Tensor) -> torch.Tensor:
        valid_counts = particle_mask.to(dtype=torch.float32).sum(dim=1)
        latent_lengths = torch.ceil(valid_counts * self.ratio).to(dtype=torch.long)
        if self.min_tokens > 0:
            latent_lengths = torch.where(
                valid_counts > 0,
                latent_lengths.clamp_min(self.min_tokens),
                torch.zeros_like(latent_lengths),
            )
        return torch.minimum(latent_lengths, valid_counts.to(dtype=torch.long))

    def compress(
        self,
        z_full: torch.Tensor,
        particle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero dropped latents and return the prefix used by the quantizer."""
        _, sequence_len, _ = z_full.shape
        if sequence_len > self.max_sequence_len:
            raise ValueError(
                f"Input sequence length {sequence_len} exceeds configured "
                f"max_sequence_len={self.max_sequence_len}"
            )

        latent_lengths = self._latent_lengths(particle_mask)
        latent_mask = (
            torch.arange(sequence_len, device=z_full.device).unsqueeze(0)
            < latent_lengths.unsqueeze(1)
        ) & particle_mask.bool()
        return z_full * latent_mask.unsqueeze(-1), latent_mask

    def expand(
        self,
        z_latent: torch.Tensor,
        latent_mask: torch.Tensor,
        target_len: int,
        particle_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Restore full length with learned placeholders at dropped positions."""
        if target_len > self.max_sequence_len:
            raise ValueError(
                f"Target sequence length {target_len} exceeds configured "
                f"max_sequence_len={self.max_sequence_len}"
            )
        if z_latent.shape[1] != target_len or latent_mask.shape[1] != target_len:
            raise ValueError(
                "Direct prefix masking keeps a full-length latent array; got "
                f"latent length {z_latent.shape[1]}, mask length {latent_mask.shape[1]}, "
                f"and target length {target_len}"
            )

        particle_mask_bool = particle_mask.bool()
        dropped_mask = particle_mask_bool & ~latent_mask.bool()
        placeholders = self.mask_embeddings[:target_len].unsqueeze(0)
        z_retained = z_latent * latent_mask.bool().unsqueeze(-1)
        z_full = z_retained + placeholders * dropped_mask.unsqueeze(-1)
        return z_full * particle_mask_bool.unsqueeze(-1)


class LatentSequenceCompressor(nn.Module):
    """Compress a full latent sequence to fewer learned slots and expand it back."""

    mode = "learned_cross_attention"

    def __init__(
        self,
        latent_dim: int,
        max_sequence_len: int,
        ratio: float = 1.0,
        min_tokens: int = 1,
        rounding: str = "ceil",
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        query_residual: bool = True,
    ):
        super().__init__()
        if ratio <= 0:
            raise ValueError(f"latent sequence compression ratio must be > 0, got {ratio}")
        if min_tokens < 0:
            raise ValueError(f"min_tokens must be >= 0, got {min_tokens}")
        if rounding != "ceil":
            raise ValueError(f"Only rounding='ceil' is currently supported, got {rounding!r}")
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim={latent_dim} must be divisible by num_heads={num_heads}"
            )

        self.latent_dim = int(latent_dim)
        self.max_sequence_len = int(max_sequence_len)
        self.ratio = float(ratio)
        self.min_tokens = int(min_tokens)
        self.rounding = rounding
        self.query_residual = bool(query_residual)
        self.max_latent_len = max(1, math.ceil(self.max_sequence_len * self.ratio))

        self.latent_queries = nn.Parameter(torch.randn(self.max_latent_len, latent_dim) * 0.02)
        self.output_queries = nn.Parameter(torch.randn(self.max_sequence_len, latent_dim) * 0.02)

        self.compress_query_norm = nn.LayerNorm(latent_dim)
        self.compress_key_norm = nn.LayerNorm(latent_dim)
        self.compress_attn = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.compress_out_norm = nn.LayerNorm(latent_dim)

        self.expand_query_norm = nn.LayerNorm(latent_dim)
        self.expand_key_norm = nn.LayerNorm(latent_dim)
        self.expand_attn = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.expand_out_norm = nn.LayerNorm(latent_dim)

    def _latent_lengths(self, particle_mask: torch.Tensor, max_latent_len: int) -> torch.Tensor:
        valid_counts = particle_mask.to(dtype=torch.float32).sum(dim=1)
        latent_lengths = torch.ceil(valid_counts * self.ratio).to(dtype=torch.long)
        if self.min_tokens > 0:
            latent_lengths = torch.where(
                valid_counts > 0,
                latent_lengths.clamp_min(self.min_tokens),
                torch.zeros_like(latent_lengths),
            )
        return latent_lengths.clamp_max(max_latent_len)

    @staticmethod
    def _safe_attention_mask(mask: torch.Tensor) -> torch.Tensor:
        safe_mask = mask.bool().clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
        return safe_mask

    def compress(
        self,
        z_full: torch.Tensor,
        particle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress full sequence latents to learned slots."""
        batch_size, sequence_len, _ = z_full.shape
        if sequence_len > self.max_sequence_len:
            raise ValueError(
                f"Input sequence length {sequence_len} exceeds configured "
                f"max_sequence_len={self.max_sequence_len}"
            )

        latent_len = max(1, math.ceil(sequence_len * self.ratio))
        queries = self.latent_queries[:latent_len].unsqueeze(0).expand(batch_size, -1, -1)
        latent_lengths = self._latent_lengths(particle_mask, latent_len)
        latent_mask = (
            torch.arange(latent_len, device=z_full.device).unsqueeze(0)
            < latent_lengths.unsqueeze(1)
        )

        safe_particle_mask = self._safe_attention_mask(particle_mask)
        z_safe = z_full * particle_mask.bool().unsqueeze(-1)
        z_compressed, _ = self.compress_attn(
            query=self.compress_query_norm(queries),
            key=self.compress_key_norm(z_safe),
            value=z_safe,
            key_padding_mask=~safe_particle_mask,
            need_weights=False,
        )
        if self.query_residual:
            z_compressed = z_compressed + queries
        z_compressed = self.compress_out_norm(z_compressed)
        return z_compressed * latent_mask.unsqueeze(-1), latent_mask

    def expand(
        self,
        z_latent: torch.Tensor,
        latent_mask: torch.Tensor,
        target_len: int,
        particle_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Expand compressed latent slots back to the original sequence length."""
        if target_len > self.max_sequence_len:
            raise ValueError(
                f"Target sequence length {target_len} exceeds configured "
                f"max_sequence_len={self.max_sequence_len}"
            )

        batch_size = z_latent.shape[0]
        queries = self.output_queries[:target_len].unsqueeze(0).expand(batch_size, -1, -1)
        safe_latent_mask = self._safe_attention_mask(latent_mask)
        z_safe = z_latent * latent_mask.bool().unsqueeze(-1)
        z_expanded, _ = self.expand_attn(
            query=self.expand_query_norm(queries),
            key=self.expand_key_norm(z_safe),
            value=z_safe,
            key_padding_mask=~safe_latent_mask,
            need_weights=False,
        )
        if self.query_residual:
            z_expanded = z_expanded + queries
        z_expanded = self.expand_out_norm(z_expanded)
        return z_expanded * particle_mask.unsqueeze(-1)
