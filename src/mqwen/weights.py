"""Validated safetensors index and zero-copy CPU views. No GPU claim implied."""
from __future__ import annotations
import json
import math
import mmap
import struct
from pathlib import Path

ITEMSIZE = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
            "F16": 2, "BF16": 2, "F32": 4, "I32": 4, "U32": 4,
            "F64": 8, "I64": 8, "U64": 8, "F8_E4M3": 1, "F8_E5M2": 1}

class WeightFile:
    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self._mapping = None
        with self.path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("Truncated safetensors length")
            n = struct.unpack("<Q", raw)[0]
            if not 2 <= n <= min(100_000_000, self.size - 8):
                raise ValueError("Invalid safetensors header length")
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"Duplicate header key: {key}")
                    result[key] = value
                return result
            header = json.loads(f.read(n), object_pairs_hook=unique)
        if not isinstance(header, dict):
            raise ValueError("Header must be an object")
        self.data_start = 8 + n
        self.metadata = header.pop("__metadata__", {})
        self.tensors = header
        spans = []
        for name, desc in header.items():
            shape, dtype, offsets = desc["shape"], desc["dtype"], desc["data_offsets"]
            if dtype not in ITEMSIZE or any(type(v) is not int or v < 0 for v in shape):
                raise ValueError(f"Unsupported dtype or invalid shape: {name}")
            if len(offsets) != 2 or any(type(v) is not int for v in offsets):
                raise ValueError(f"Invalid offsets: {name}")
            start, end = offsets
            if not 0 <= start <= end <= self.size - self.data_start:
                raise ValueError(f"Tensor outside file: {name}")
            if end - start != math.prod(shape) * ITEMSIZE[dtype]:
                raise ValueError(f"Tensor byte size mismatch: {name}")
            spans.append((start, end))
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError("Tensor payload has gaps or overlaps")
            cursor = end
        if cursor != self.size - self.data_start:
            raise ValueError("Unindexed trailing payload")

    def __enter__(self):
        with self.path.open("rb") as f:
            self._mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def view(self, name):
        if self._mapping is None:
            raise RuntimeError("Use WeightFile as a context manager")
        start, end = self.tensors[name]["data_offsets"]
        return memoryview(self._mapping)[self.data_start + start:self.data_start + end]

    def __exit__(self, *args):
        # Callers must release exported views before exiting.
        self._mapping.close()
        self._mapping = None

    def summary(self):
        dtypes = {}
        for desc in self.tensors.values():
            a, b = desc["data_offsets"]
            dtypes[desc["dtype"]] = dtypes.get(desc["dtype"], 0) + b - a
        return {"path": str(self.path), "bytes": self.size,
                "tensor_count": len(self.tensors), "dtype_bytes": dtypes,
                "metadata": self.metadata}
