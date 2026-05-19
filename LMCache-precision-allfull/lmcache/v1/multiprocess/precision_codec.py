# SPDX-License-Identifier: Apache-2.0
"""Precision/base codec helpers for LMCache MPConnector experiments.

This module is intentionally scoped to the multiprocess server path used by
LMCacheMPConnector. The regular v1 CacheEngine fidelity codec is not on this
path, so MP mode needs its own boundary encode/decode helpers.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

MP_FULL_TIER = "full"
MP_BASE_TIER = "base"

_PRECISION_POLICY_ENV = "LMCACHE_PRECISION_POLICY"
_ENABLE_FIDELITY_ENV = "LMCACHE_ENABLE_FIDELITY_CACHE"
_BASE_CODEC_ENV = "LMCACHE_BASE_CODEC"
_MIXED_SINK_CHUNKS_ENV = "LMCACHE_MIXED_SINK_CHUNKS"
_MIXED_RECENT_CHUNKS_ENV = "LMCACHE_MIXED_RECENT_CHUNKS"
_PRECISION_THRESHOLD_ENV = "LMCACHE_PRECISION_THRESHOLD"
_INT8_QUANT_EPS = 1e-8


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return max(0, int(raw))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return float(raw)


def precision_policy_from_env() -> str:
    return os.environ.get(_PRECISION_POLICY_ENV, "all-full").strip() or "all-full"


def base_codec_from_env() -> str:
    return os.environ.get(_BASE_CODEC_ENV, "fake").strip().lower() or "fake"


def is_mp_int8_base_enabled() -> bool:
    return (
        precision_policy_from_env() == "all-base"
        and _env_bool(_ENABLE_FIDELITY_ENV)
        and base_codec_from_env() == "int8"
    )


def is_mp_mixed_3span_enabled() -> bool:
    return (
        precision_policy_from_env() == "mixed-3span"
        and _env_bool(_ENABLE_FIDELITY_ENV)
        and base_codec_from_env() == "int8"
    )


def is_mp_int8_base_tier_enabled(tier: str) -> bool:
    return (
        tier == MP_BASE_TIER
        and _env_bool(_ENABLE_FIDELITY_ENV)
        and base_codec_from_env() == "int8"
    )


def namespace_model_name_for_precision(model_name: str, tier: str | None = None) -> str:
    """Return the MP ObjectKey model namespace for a precision tier."""

    if tier is None:
        tier = MP_BASE_TIER if precision_policy_from_env() == "all-base" else MP_FULL_TIER
    if tier == MP_FULL_TIER:
        return model_name
    if tier == MP_BASE_TIER:
        codec = base_codec_from_env()
        return f"{model_name}#precision=base#codec={codec}"
    raise ValueError(f"Unsupported MP precision tier: {tier}")


def build_mixed_3span_tiers(
    total_chunks: int,
    sink_chunks: int | None = None,
    recent_chunks: int | None = None,
) -> tuple[str, ...]:
    """Return hard-coded Phase 6A [full-sink, base, full-recent] tiers."""

    if total_chunks < 0:
        raise ValueError(f"total_chunks must be non-negative, got {total_chunks}")
    if total_chunks == 0:
        return ()
    sink = _env_int(_MIXED_SINK_CHUNKS_ENV, 1) if sink_chunks is None else sink_chunks
    recent = (
        _env_int(_MIXED_RECENT_CHUNKS_ENV, 2)
        if recent_chunks is None
        else recent_chunks
    )
    sink_end = min(max(sink, 0), total_chunks)
    recent_start = max(sink_end, total_chunks - max(recent, 0))
    tiers = [MP_BASE_TIER] * total_chunks
    for idx in range(0, sink_end):
        tiers[idx] = MP_FULL_TIER
    for idx in range(recent_start, total_chunks):
        tiers[idx] = MP_FULL_TIER
    return tuple(tiers)


def _is_standard_threshold_span_shape(tiers: tuple[str, ...]) -> bool:
    spans = group_precision_tier_spans(tiers)
    if not spans:
        return True
    if len(spans) == 1:
        return spans[0][2] == MP_FULL_TIER
    if len(spans) != 3:
        return False
    return (
        spans[0][0] == 0
        and spans[0][2] == MP_FULL_TIER
        and spans[1][2] == MP_BASE_TIER
        and spans[2][2] == MP_FULL_TIER
        and spans[2][1] == len(tiers)
    )


def normalize_threshold_mixed_tiers(tiers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(tiers)
    if _is_standard_threshold_span_shape(normalized):
        return normalized
    return (MP_FULL_TIER,) * len(normalized)


def build_threshold_mixed_tiers(
    total_chunks: int,
    threshold: float | None = None,
    sink_chunks: int | None = None,
) -> tuple[str, ...]:
    """Build Phase 6B tiers from sink and linear recency scores.

    score = max(sink_score, recent_score), where
    recent_score = (chunk_index + 1) / total_chunks and sink_score=1.0 for
    the first sink chunk(s). Chunks with score >= threshold use full KV.
    Non-standard span shapes conservatively fall back to all-full.
    """

    if total_chunks < 0:
        raise ValueError(f"total_chunks must be non-negative, got {total_chunks}")
    if total_chunks == 0:
        return ()
    threshold_value = (
        _env_float(_PRECISION_THRESHOLD_ENV, 0.5)
        if threshold is None
        else threshold
    )
    sink = _env_int(_MIXED_SINK_CHUNKS_ENV, 1) if sink_chunks is None else sink_chunks
    tiers: list[str] = []
    for chunk_idx in range(total_chunks):
        recent_score = (chunk_idx + 1) / total_chunks
        sink_score = 1.0 if chunk_idx < max(sink, 0) else 0.0
        score = max(recent_score, sink_score)
        tiers.append(MP_FULL_TIER if score >= threshold_value else MP_BASE_TIER)
    return normalize_threshold_mixed_tiers(tuple(tiers))


def precision_tiers_for_range(
    total_chunks: int,
    start_chunk: int,
    end_chunk: int,
) -> tuple[str, ...]:
    """Return precision tiers for chunk range [start_chunk, end_chunk)."""

    if start_chunk < 0 or end_chunk < start_chunk or end_chunk > total_chunks:
        raise ValueError(
            "Invalid chunk range: "
            f"start={start_chunk}, end={end_chunk}, total={total_chunks}"
        )
    policy = precision_policy_from_env()
    if policy == "all-base":
        tiers = (MP_BASE_TIER,) * total_chunks
    elif policy == "mixed-3span":
        tiers = build_mixed_3span_tiers(total_chunks)
    elif policy == "threshold-mixed":
        tiers = build_threshold_mixed_tiers(total_chunks)
    else:
        tiers = (MP_FULL_TIER,) * total_chunks
    return tiers[start_chunk:end_chunk]


def group_precision_tier_spans(tiers: tuple[str, ...] | list[str]) -> tuple[tuple[int, int, str], ...]:
    """Group contiguous chunk tiers into (start, end, tier) spans."""

    if not tiers:
        return ()
    spans: list[tuple[int, int, str]] = []
    start = 0
    current = tiers[0]
    for idx, tier in enumerate(tiers[1:], start=1):
        if tier == current:
            continue
        spans.append((start, idx, current))
        start = idx
        current = tier
    spans.append((start, len(tiers), current))
    return tuple(spans)


def build_mp_base_layout_desc(layout_desc):
    """Build INT8+scale layout for MP base KV storage."""

    shapes: list[torch.Size] = []
    dtypes: list[torch.dtype] = []
    for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
        if not dtype.is_floating_point:
            raise TypeError(f"MP int8 base only supports floating dtype, got {dtype}")
        if len(shape) == 0:
            raise ValueError("MP int8 base requires tensor shape with at least one dim")
        scale_shape = torch.Size([*shape[:-1], 1])
        shapes.extend([shape, scale_shape])
        dtypes.extend([torch.int8, torch.float32])
    return layout_desc.__class__(shapes=shapes, dtypes=dtypes)


def _layout_size_bytes(layout_desc) -> int:
    return sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
    )


def make_cpu_memory_obj(
    layout_desc,
    fmt: MemoryFormat,
) -> TensorMemoryObj:
    raw_size = _layout_size_bytes(layout_desc)
    raw_data = torch.empty(raw_size, dtype=torch.uint8, device="cpu")
    return TensorMemoryObj(
        raw_data=raw_data,
        metadata=MemoryObjMetadata(
            shape=layout_desc.shapes[0],
            dtype=layout_desc.dtypes[0],
            address=0,
            phy_size=raw_size,
            ref_count=1,
            pin_count=0,
            fmt=fmt,
            shapes=layout_desc.shapes,
            dtypes=layout_desc.dtypes,
        ),
        parent_allocator=None,
    )


def _encode_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"MP int8 base only supports floating tensors, got {tensor.dtype}")
    working = tensor.to(torch.float32) if tensor.device.type == "cpu" else tensor
    scale = working.abs().amax(dim=-1, keepdim=True).clamp_min(_INT8_QUANT_EPS) / 127.0
    quantized = torch.round(working / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


def encode_tensor_to_base_memory_obj(
    src: torch.Tensor,
    encoded_obj: MemoryObj,
    group_idx: int = 0,
) -> None:
    q_dst = encoded_obj.get_tensor(group_idx * 2)
    scale_dst = encoded_obj.get_tensor(group_idx * 2 + 1)
    if q_dst is None or scale_dst is None:
        raise ValueError(f"Missing MP base tensor group {group_idx}")
    quantized, scale = _encode_tensor(src)
    q_dst.copy_(quantized, non_blocking=quantized.device.type == "cuda")
    scale_dst.copy_(scale, non_blocking=scale.device.type == "cuda")


def encode_memory_obj_to_base(full_obj: MemoryObj, encoded_obj: MemoryObj) -> None:
    full_shapes = full_obj.get_shapes()
    if len(encoded_obj.get_shapes()) != len(full_shapes) * 2:
        raise ValueError("Encoded MP base object must have quantized+scale groups")
    for group_idx in range(len(full_shapes)):
        src = full_obj.get_tensor(group_idx)
        if src is None:
            raise ValueError(f"Missing MP full tensor group {group_idx}")
        encode_tensor_to_base_memory_obj(src, encoded_obj, group_idx)


def decode_base_memory_obj_to_tensor(
    encoded_obj: MemoryObj,
    dst: torch.Tensor,
    group_idx: int = 0,
) -> torch.Tensor:
    quantized = encoded_obj.get_tensor(group_idx * 2)
    scale = encoded_obj.get_tensor(group_idx * 2 + 1)
    if quantized is None or scale is None:
        raise ValueError(f"Missing MP base tensor group {group_idx}")
    if dst.device.type == "cuda":
        dst.copy_(quantized, non_blocking=True)
        scale_on_dst = scale.to(device=dst.device, dtype=dst.dtype, non_blocking=True)
        dst.mul_(scale_on_dst)
    else:
        decoded = (quantized.to(torch.float32) * scale.to(torch.float32)).to(dst.dtype)
        dst.copy_(decoded)
    return dst


def decode_base_memory_obj(encoded_obj: MemoryObj, decoded_obj: MemoryObj) -> None:
    decoded_shapes = decoded_obj.get_shapes()
    if len(encoded_obj.get_shapes()) != len(decoded_shapes) * 2:
        raise ValueError("Encoded MP base object must have quantized+scale groups")
    for group_idx in range(len(decoded_shapes)):
        dst = decoded_obj.get_tensor(group_idx)
        if dst is None:
            raise ValueError(f"Missing MP decoded tensor group {group_idx}")
        decode_base_memory_obj_to_tensor(encoded_obj, dst, group_idx)


def quantization_error_summary(
    reference: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, float]:
    ref = reference.detach().to(torch.float32).flatten()
    rec = reconstructed.detach().to(torch.float32).flatten()
    err = (ref - rec).abs()
    cosine = F.cosine_similarity(ref, rec, dim=0, eps=1e-12)
    return {
        "max_abs_error": float(err.max().item()),
        "mean_abs_error": float(err.mean().item()),
        "cosine_similarity": float(cosine.item()),
    }
