from __future__ import annotations

import json
import os
import threading
import time
from itertools import count
from typing import Any, List

import torch

from lmcache.v1.memory_management import MemoryObj

_CODEC_NAME_KEY = "lmcache.fidelity.codec"
_CODEC_VERSION_KEY = "lmcache.fidelity.codec_version"
_ORIGINAL_SHAPES_KEY = "lmcache.fidelity.original_shapes"
_ORIGINAL_DTYPES_KEY = "lmcache.fidelity.original_dtypes"
_GROUP_COUNT_KEY = "lmcache.fidelity.group_count"


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype)


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str.replace("torch.", ""))


_INT8_ERROR_TRACE_LOCK = threading.Lock()
_INT8_ERROR_TRACE_COUNTER = count()


def _sample_quantile(values: torch.Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values, q).item())


def _maybe_trace_int8_error(
    tensor: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    trace_path = os.environ.get("LMCACHE_INT8_ERROR_TRACE_PATH")
    if not trace_path:
        return
    try:
        with torch.no_grad():
            original = tensor.detach().to(torch.float32)
            reconstructed = quantized.to(torch.float32) * scale.to(torch.float32)
            abs_err = (original - reconstructed).abs()
            flat = abs_err.reshape(-1)
            numel = int(flat.numel())
            sample_limit = int(os.environ.get("LMCACHE_INT8_ERROR_TRACE_SAMPLE_LIMIT", "262144"))
            stride = max(1, numel // max(1, sample_limit))
            sample = flat[::stride][:sample_limit].contiguous()
            row = {
                "event": "int8_real_kv_error_chunk",
                "trace_tag": os.environ.get("LMCACHE_INT8_ERROR_TRACE_TAG", ""),
                "trace_index": next(_INT8_ERROR_TRACE_COUNTER),
                "pid": os.getpid(),
                "time_unix": time.time(),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": numel,
                "max_abs_original": float(original.abs().max().item()) if numel else 0.0,
                "mean_abs_original": float(original.abs().mean().item()) if numel else 0.0,
                "scale_min": float(scale.min().item()) if scale.numel() else 0.0,
                "scale_max": float(scale.max().item()) if scale.numel() else 0.0,
                "max_err": float(flat.max().item()) if numel else 0.0,
                "mean_err": float(flat.mean().item()) if numel else 0.0,
                "sample_count": int(sample.numel()),
                "sample_p50_err": _sample_quantile(sample, 0.50),
                "sample_p95_err": _sample_quantile(sample, 0.95),
                "sample_p99_err": _sample_quantile(sample, 0.99),
                "sample_p999_err": _sample_quantile(sample, 0.999),
            }
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with _INT8_ERROR_TRACE_LOCK:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        err_row = {
            "event": "int8_real_kv_error_trace_failure",
            "trace_tag": os.environ.get("LMCACHE_INT8_ERROR_TRACE_TAG", ""),
            "pid": os.getpid(),
            "time_unix": time.time(),
            "error": repr(exc),
        }
        with _INT8_ERROR_TRACE_LOCK:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(err_row, ensure_ascii=False, sort_keys=True) + "\n")


class FidelityCodec:
    name: str = "unknown"

    def get_base_dtypes(self, dtypes: List[torch.dtype]) -> List[torch.dtype]:
        return list(dtypes)

    def get_base_dtype(self, dtype: torch.dtype) -> torch.dtype:
        return self.get_base_dtypes([dtype])[0]

    def encode(self, allocator: Any, memory_obj: MemoryObj) -> MemoryObj:
        return memory_obj

    def decode(self, allocator: Any, memory_obj: MemoryObj) -> MemoryObj:
        return memory_obj

    def is_encoded_memory_obj(self, memory_obj: MemoryObj) -> bool:
        return False


class FakeBaseCodec(FidelityCodec):
    name = "fake"


class Int8BaseCodec(FidelityCodec):
    name = "int8"

    def __init__(self, scale_dtype: torch.dtype = torch.float32, min_scale: float = 1e-8):
        self.scale_dtype = scale_dtype
        self.min_scale = min_scale

    def get_base_dtypes(self, dtypes: List[torch.dtype]) -> List[torch.dtype]:
        # Boundary encode/decode needs the temporary store buffers to stay in the
        # original floating dtype; compression happens after D2H offload.
        return list(dtypes)

    def encode_base_tensor(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_floating_point(tensor):
            raise TypeError(
                f"Int8BaseCodec only supports floating tensors, got {tensor.dtype}"
            )
        working = tensor.to(torch.float32)
        scale = working.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(scale / 127.0, min=self.min_scale)
        quantized = torch.round(working / scale).clamp(-128, 127).to(torch.int8)
        stored_scale = scale.to(self.scale_dtype)
        _maybe_trace_int8_error(tensor, quantized, stored_scale)
        return quantized, stored_scale

    def decode_base_tensor(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return (quantized.to(torch.float32) * scale.to(torch.float32)).to(output_dtype)

    def is_encoded_memory_obj(self, memory_obj: MemoryObj) -> bool:
        extra = getattr(memory_obj.metadata, "extra", None) or {}
        return (
            extra.get(_CODEC_NAME_KEY) == self.name
            and extra.get(_CODEC_VERSION_KEY) == 1
        )

    def encode(self, allocator: Any, memory_obj: MemoryObj) -> MemoryObj:
        if self.is_encoded_memory_obj(memory_obj):
            return memory_obj

        original_shapes = memory_obj.get_shapes()
        original_dtypes = memory_obj.get_dtypes()
        encoded_tensors: list[torch.Tensor] = []
        encoded_shapes: list[torch.Size] = []
        encoded_dtypes: list[torch.dtype] = []

        for group_idx in range(len(original_shapes)):
            group_tensor = memory_obj.get_tensor(group_idx)
            if group_tensor is None:
                raise ValueError(f"Missing tensor group {group_idx} while encoding")

            quantized, stored_scale = self.encode_base_tensor(group_tensor)
            encoded_tensors.extend([quantized, stored_scale])
            encoded_shapes.extend([quantized.shape, stored_scale.shape])
            encoded_dtypes.extend([quantized.dtype, stored_scale.dtype])

        encoded_memory_obj = allocator.allocate(
            encoded_shapes,
            encoded_dtypes,
            fmt=memory_obj.metadata.fmt,
        )
        if encoded_memory_obj is None:
            raise RuntimeError("Failed to allocate memory for Int8BaseCodec encode")

        for group_idx, tensor in enumerate(encoded_tensors):
            dst = encoded_memory_obj.get_tensor(group_idx)
            if dst is None:
                raise ValueError(f"Missing encoded tensor slot {group_idx}")
            dst.copy_(tensor)

        extra = dict(getattr(memory_obj.metadata, "extra", None) or {})
        extra.update(
            {
                _CODEC_NAME_KEY: self.name,
                _CODEC_VERSION_KEY: 1,
                _GROUP_COUNT_KEY: len(original_shapes),
                _ORIGINAL_SHAPES_KEY: [list(shape) for shape in original_shapes],
                _ORIGINAL_DTYPES_KEY: [_dtype_to_str(dtype) for dtype in original_dtypes],
            }
        )
        encoded_memory_obj.metadata.extra = extra
        encoded_memory_obj.metadata.cached_positions = memory_obj.metadata.cached_positions
        return encoded_memory_obj

    def decode(self, allocator: Any, memory_obj: MemoryObj) -> MemoryObj:
        if not self.is_encoded_memory_obj(memory_obj):
            return memory_obj

        extra = dict(memory_obj.metadata.extra or {})
        group_count = int(extra[_GROUP_COUNT_KEY])
        original_shapes = [torch.Size(shape) for shape in extra[_ORIGINAL_SHAPES_KEY]]
        original_dtypes = [_dtype_from_str(dtype) for dtype in extra[_ORIGINAL_DTYPES_KEY]]

        decoded_memory_obj = allocator.allocate(
            original_shapes,
            original_dtypes,
            fmt=memory_obj.metadata.fmt,
        )
        if decoded_memory_obj is None:
            raise RuntimeError("Failed to allocate memory for Int8BaseCodec decode")

        for group_idx in range(group_count):
            quantized = memory_obj.get_tensor(group_idx * 2)
            scale = memory_obj.get_tensor(group_idx * 2 + 1)
            dst = decoded_memory_obj.get_tensor(group_idx)
            if quantized is None or scale is None or dst is None:
                raise ValueError(f"Missing tensor group {group_idx} while decoding")
            dst.copy_(self.decode_base_tensor(quantized, scale, dst.dtype))

        for key in (
            _CODEC_NAME_KEY,
            _CODEC_VERSION_KEY,
            _GROUP_COUNT_KEY,
            _ORIGINAL_SHAPES_KEY,
            _ORIGINAL_DTYPES_KEY,
        ):
            extra.pop(key, None)
        decoded_memory_obj.metadata.extra = extra or None
        decoded_memory_obj.metadata.cached_positions = memory_obj.metadata.cached_positions
        return decoded_memory_obj


def create_fidelity_codec(name: str) -> FidelityCodec:
    codec_name = (name or "fake").lower()
    if codec_name == "fake":
        return FakeBaseCodec()
    if codec_name == "int8":
        return Int8BaseCodec()
    raise ValueError(f"Unsupported base codec: {name}")
