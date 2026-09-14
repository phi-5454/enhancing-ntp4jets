"""Compact, self-describing binary files for ragged ORBIT events."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MAGIC = b"ORBTBIN1"
PACKED_MAGIC = b"ORBTPK01"
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


def bits_for_size(size: int) -> int:
    """Return the minimum positive bit width needed for values in ``[0, size)``."""
    if size < 1:
        raise ValueError("size must be positive")
    return max(1, (size - 1).bit_length())


def pack_unsigned_values(values: np.ndarray, bits_per_value: int) -> bytes:
    """Pack unsigned integers LSB-first into a dense byte stream."""
    if not 1 <= bits_per_value <= 64:
        raise ValueError("bits_per_value must be in [1, 64]")
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional array, got {values.shape}")
    limit = 1 << bits_per_value
    output = bytearray()
    accumulator = 0
    accumulator_bits = 0
    for raw_value in values:
        value = int(raw_value)
        if value < 0 or value >= limit:
            raise ValueError(f"Value {value} does not fit in {bits_per_value} bits")
        accumulator |= value << accumulator_bits
        accumulator_bits += bits_per_value
        while accumulator_bits >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            accumulator_bits -= 8
    if accumulator_bits:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_unsigned_values(payload: bytes, count: int, bits_per_value: int) -> np.ndarray:
    """Inverse of :func:`pack_unsigned_values`."""
    if count < 0:
        raise ValueError("count must be non-negative")
    expected_bytes = (count * bits_per_value + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} packed bytes, got {len(payload)}")
    mask = (1 << bits_per_value) - 1
    values = np.empty(count, dtype=np.uint64)
    accumulator = 0
    accumulator_bits = 0
    offset = 0
    for index in range(count):
        while accumulator_bits < bits_per_value:
            accumulator |= payload[offset] << accumulator_bits
            accumulator_bits += 8
            offset += 1
        values[index] = accumulator & mask
        accumulator >>= bits_per_value
        accumulator_bits -= bits_per_value
    return values


def write_packed_ragged_binary(
    path: str | Path,
    events: Iterable[np.ndarray],
    *,
    bits_per_value: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write length-prefixed, densely bit-packed unsigned event sequences."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(event).reshape(-1) for event in events]
    header = {
        "format": "orbit-packed-ragged-binary",
        "version": 1,
        "bits_per_value": int(bits_per_value),
        "bit_order": "lsb0",
        "event_count": len(arrays),
        "record_layout": "uint32_le count followed by ceil(count*bits_per_value/8) bytes",
        **metadata,
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as output:
        output.write(PACKED_MAGIC)
        output.write(_HEADER_LENGTH.pack(len(encoded_header)))
        output.write(encoded_header)
        for array in arrays:
            output.write(_EVENT_LENGTH.pack(len(array)))
            output.write(pack_unsigned_values(array, bits_per_value))
    return header


def read_packed_ragged_binary(path: str | Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Read a file produced by :func:`write_packed_ragged_binary`."""
    with Path(path).open("rb") as source:
        if source.read(len(PACKED_MAGIC)) != PACKED_MAGIC:
            raise ValueError("Not an ORBIT packed ragged binary file")
        header_size_raw = source.read(_HEADER_LENGTH.size)
        if len(header_size_raw) != _HEADER_LENGTH.size:
            raise ValueError("Truncated ORBIT packed binary header length")
        header_size = _HEADER_LENGTH.unpack(header_size_raw)[0]
        encoded_header = source.read(header_size)
        if len(encoded_header) != header_size:
            raise ValueError("Truncated ORBIT packed binary header")
        header = json.loads(encoded_header)
        bits_per_value = int(header["bits_per_value"])
        events = []
        for _ in range(int(header["event_count"])):
            length_raw = source.read(_EVENT_LENGTH.size)
            if len(length_raw) != _EVENT_LENGTH.size:
                raise ValueError("Truncated ORBIT packed binary event length")
            count = _EVENT_LENGTH.unpack(length_raw)[0]
            byte_count = (count * bits_per_value + 7) // 8
            payload = source.read(byte_count)
            if len(payload) != byte_count:
                raise ValueError("Truncated ORBIT packed binary event payload")
            events.append(unpack_unsigned_values(payload, count, bits_per_value))
        if source.read(1):
            raise ValueError("Unexpected trailing bytes in ORBIT packed binary file")
    return header, events


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
