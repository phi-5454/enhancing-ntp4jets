"""Compact, self-describing binary files for ragged ORBIT events."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MAGIC = b"ORBTBIN1"
_HEADER_LENGTH = struct.Struct("<I")
_EVENT_LENGTH = struct.Struct("<I")


def unsigned_dtype_for_size(size: int) -> np.dtype:
    """Return the narrowest little-endian unsigned dtype that stores ``size - 1``."""
    if size < 1:
        raise ValueError("size must be positive")
    if size <= 1 << 8:
        return np.dtype("u1")
    if size <= 1 << 16:
        return np.dtype("<u2")
    if size <= 1 << 32:
        return np.dtype("<u4")
    return np.dtype("<u8")


def write_ragged_binary(
    path: str | Path,
    events: Iterable[np.ndarray],
    *,
    metadata: dict[str, Any],
    dtype: str | np.dtype,
    width: int = 1,
) -> dict[str, Any]:
    """Write length-prefixed arrays and return the embedded header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    if dtype.byteorder == "=" and dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    if width < 1:
        raise ValueError("width must be positive")

    arrays = []
    for event in events:
        array = np.asarray(event, dtype=dtype)
        expected_shape = (array.shape[0],) if width == 1 else (array.shape[0], width)
        if array.shape != expected_shape:
            raise ValueError(
                f"Expected event shape {expected_shape} for width={width}, got {array.shape}"
            )
        arrays.append(np.ascontiguousarray(array))

    header = {
        "format": "orbit-ragged-binary",
        "version": 1,
        "dtype": dtype.str,
        "width": int(width),
        "event_count": len(arrays),
        "record_layout": "uint32_le length followed by length*width values",
        **metadata,
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")

    with path.open("wb") as output:
        output.write(MAGIC)
        output.write(_HEADER_LENGTH.pack(len(encoded_header)))
        output.write(encoded_header)
        for array in arrays:
            output.write(_EVENT_LENGTH.pack(array.shape[0]))
            output.write(array.tobytes(order="C"))
    return header


def read_ragged_binary(path: str | Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Read a file produced by :func:`write_ragged_binary`."""
    with Path(path).open("rb") as source:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError("Not an ORBIT ragged binary file")
        header_size_raw = source.read(_HEADER_LENGTH.size)
        if len(header_size_raw) != _HEADER_LENGTH.size:
            raise ValueError("Truncated ORBIT binary header length")
        header_size = _HEADER_LENGTH.unpack(header_size_raw)[0]
        header = json.loads(source.read(header_size))
        dtype = np.dtype(header["dtype"])
        width = int(header["width"])
        events = []
        for _ in range(int(header["event_count"])):
            length_raw = source.read(_EVENT_LENGTH.size)
            if len(length_raw) != _EVENT_LENGTH.size:
                raise ValueError("Truncated ORBIT binary event length")
            length = _EVENT_LENGTH.unpack(length_raw)[0]
            value_count = length * width
            payload = source.read(value_count * dtype.itemsize)
            if len(payload) != value_count * dtype.itemsize:
                raise ValueError("Truncated ORBIT binary event payload")
            event = np.frombuffer(payload, dtype=dtype).copy()
            events.append(event if width == 1 else event.reshape(length, width))
        if source.read(1):
            raise ValueError("Unexpected trailing bytes in ORBIT binary file")
    return header, events
