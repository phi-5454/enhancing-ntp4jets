"""Quantization for the common 40-bit L1 PUPPI candidate payload."""

from __future__ import annotations

import numpy as np


PT_BITS = 14
ETA_BITS = 12
PHI_BITS = 11
PID_BITS = 3
PUPPI_COMMON_BITS = PT_BITS + ETA_BITS + PHI_BITS + PID_BITS
PT_LSB_GEV = 0.25
ANGLE_LSB = np.pi / 720.0


def _quantize(values, lsb: float, bits: int, *, signed: bool) -> np.ndarray:
    quantized = np.rint(np.asarray(values, dtype=np.float64) / lsb)
    if signed:
        lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lower, upper = 0, (1 << bits) - 1
    return np.clip(quantized, lower, upper).astype(np.int64)


def quantize_puppi_common(features: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """Return packed uint64 words whose low 40 bits follow the PUPPI common format.

    ``features`` is ordered as ``[pT, eta, phi]``. Values are rounded to the
    nearest representable LSB and saturated. Phi is first mapped to ``[-pi, pi)``.
    """
    features = np.asarray(features)
    pid = np.asarray(pid)
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError(f"Expected features with shape (N, 3), got {features.shape}")
    if pid.shape != (features.shape[0],):
        raise ValueError(f"Expected PID shape {(features.shape[0],)}, got {pid.shape}")
    if np.any((pid < 0) | (pid >= 1 << PID_BITS)):
        raise ValueError("PID values must be in [0, 8)")

    pt = _quantize(features[:, 0], PT_LSB_GEV, PT_BITS, signed=False)
    eta = _quantize(features[:, 1], ANGLE_LSB, ETA_BITS, signed=True)
    phi_values = (features[:, 2] + np.pi) % (2 * np.pi) - np.pi
    phi = _quantize(phi_values, ANGLE_LSB, PHI_BITS, signed=True)

    eta_encoded = eta & ((1 << ETA_BITS) - 1)
    phi_encoded = phi & ((1 << PHI_BITS) - 1)
    return (
        pt.astype(np.uint64)
        | (eta_encoded.astype(np.uint64) << PT_BITS)
        | (phi_encoded.astype(np.uint64) << (PT_BITS + ETA_BITS))
        | (pid.astype(np.uint64) << (PT_BITS + ETA_BITS + PHI_BITS))
    )


def _decode_signed(values: np.ndarray, bits: int) -> np.ndarray:
    sign = 1 << (bits - 1)
    values = values.astype(np.int64)
    return (values ^ sign) - sign


def unpack_puppi_common(words: np.ndarray) -> dict[str, np.ndarray]:
    """Unpack common PUPPI words into quantized integer fields."""
    words = np.asarray(words, dtype=np.uint64)
    pt = words & ((1 << PT_BITS) - 1)
    eta = (words >> PT_BITS) & ((1 << ETA_BITS) - 1)
    phi = (words >> (PT_BITS + ETA_BITS)) & ((1 << PHI_BITS) - 1)
    pid = (words >> (PT_BITS + ETA_BITS + PHI_BITS)) & ((1 << PID_BITS) - 1)
    return {
        "pt": pt.astype(np.uint16),
        "eta": _decode_signed(eta, ETA_BITS).astype(np.int16),
        "phi": _decode_signed(phi, PHI_BITS).astype(np.int16),
        "pid": pid.astype(np.uint8),
    }
