# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import IntEnum
import subprocess
from typing import Iterable

import torch

from lmcache import non_cuda_equivalents as _nce

PYTHON_FALLBACK = True


class TransferDirection(IntEnum):
    H2D = 0
    D2H = 1


class GPUKVFormat(IntEnum):
    NB_NL_TWO_BS_NH_HS = 0
    NL_X_TWO_NB_BS_NH_HS = 1
    NL_X_NB_TWO_BS_NH_HS = 2
    NL_X_NB_BS_HS = 3
    TWO_X_NL_X_NBBS_NH_HS = 4
    NL_X_NBBS_ONE_HS = 5


alloc_pinned_ptr = _nce.alloc_pinned_ptr
free_pinned_ptr = _nce.free_pinned_ptr
alloc_pinned_numa_ptr = _nce.alloc_pinned_numa_ptr
free_pinned_numa_ptr = _nce.free_pinned_numa_ptr
alloc_shm_pinned_ptr = _nce.alloc_shm_pinned_ptr
free_shm_pinned_ptr = _nce.free_shm_pinned_ptr


def alloc_numa_ptr(size: int, numa_id: int = 0) -> int:
    return _nce.alloc_pinned_numa_ptr(size, numa_id)


def free_numa_ptr(ptr: int, size: int | None = None) -> None:
    _nce.free_pinned_numa_ptr(ptr, size)


def get_gpu_pci_bus_id(device_index: int) -> str:
    cmd = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=pci.bus_id",
        "--format=csv,noheader",
    ]
    return subprocess.check_output(cmd, text=True).strip()


def _as_direction(direction: TransferDirection | int) -> TransferDirection:
    return direction if isinstance(direction, TransferDirection) else TransferDirection(direction)


def _as_format(gpu_kv_format: GPUKVFormat | int) -> GPUKVFormat:
    return gpu_kv_format if isinstance(gpu_kv_format, GPUKVFormat) else GPUKVFormat(gpu_kv_format)


def _copy_tensor(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.device == src.device:
        dst.copy_(src)
    else:
        dst.copy_(src.to(dst.device), non_blocking=True)


def _normalize_slot_mapping(slot_mapping: torch.Tensor, skip_prefix_n_tokens: int = 0) -> torch.Tensor:
    if skip_prefix_n_tokens:
        slot_mapping = slot_mapping[skip_prefix_n_tokens:]
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(dtype=torch.int64)
    return slot_mapping


def _layer_refs(key_value_refs) -> list[torch.Tensor]:
    if isinstance(key_value_refs, list):
        return key_value_refs
    if isinstance(key_value_refs, tuple):
        return list(key_value_refs)
    raise TypeError(
        "Python fallback expects real kv-cache tensors instead of raw pointer arrays; "
        f"got {type(key_value_refs)!r}."
    )


def _block_indices(slot_mapping: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0 for paged vLLM formats")
    return torch.div(slot_mapping, block_size, rounding_mode="floor"), torch.remainder(slot_mapping, block_size)


def _gather_non_mla_layer(
    kv_layer: torch.Tensor,
    block_idx: torch.Tensor,
    block_offset: torch.Tensor,
    gpu_kv_format: GPUKVFormat,
) -> torch.Tensor:
    if gpu_kv_format == GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        selected = kv_layer[:, block_idx, block_offset, :, :]
    elif gpu_kv_format == GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        selected = kv_layer[block_idx, :, block_offset, :, :].permute(1, 0, 2, 3)
    else:
        raise NotImplementedError(f"Unsupported non-MLA format in fallback: {gpu_kv_format}")
    return selected.reshape(2, block_idx.numel(), -1)


def _scatter_non_mla_layer(
    kv_layer: torch.Tensor,
    values: torch.Tensor,
    block_idx: torch.Tensor,
    block_offset: torch.Tensor,
    gpu_kv_format: GPUKVFormat,
) -> None:
    num_heads = kv_layer.shape[-2]
    head_size = kv_layer.shape[-1]
    values = values.reshape(2, block_idx.numel(), num_heads, head_size)
    if values.device != kv_layer.device or values.dtype != kv_layer.dtype:
        values = values.to(device=kv_layer.device, dtype=kv_layer.dtype, non_blocking=True)
    if gpu_kv_format == GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        kv_layer[:, block_idx, block_offset, :, :] = values
    elif gpu_kv_format == GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        kv_layer[block_idx, :, block_offset, :, :] = values.permute(1, 0, 2, 3)
    else:
        raise NotImplementedError(f"Unsupported non-MLA format in fallback: {gpu_kv_format}")


def multi_layer_kv_transfer(
    key_value: torch.Tensor,
    key_value_ptrs,
    slot_mapping: torch.Tensor,
    paged_memory_device,
    page_buffer_size: int,
    direction: TransferDirection | int,
    gpu_kv_format: GPUKVFormat | int,
    block_size: int = 0,
    skip_prefix_n_tokens: int = 0,
) -> None:
    del paged_memory_device, page_buffer_size
    direction = _as_direction(direction)
    gpu_kv_format = _as_format(gpu_kv_format)

    mem_offset = skip_prefix_n_tokens if direction == TransferDirection.H2D else 0
    slot_mapping = _normalize_slot_mapping(slot_mapping, skip_prefix_n_tokens if direction == TransferDirection.H2D else 0)
    if slot_mapping.numel() == 0:
        return

    if gpu_kv_format == GPUKVFormat.NL_X_NB_BS_HS:
        kv_layers = _layer_refs(key_value_ptrs)
        block_idx, block_offset = _block_indices(slot_mapping, block_size)
        for layer_id, kv_layer in enumerate(kv_layers):
            mem_view = key_value[0, layer_id, mem_offset : mem_offset + slot_mapping.numel()]
            if direction == TransferDirection.D2H:
                gathered = kv_layer[block_idx, block_offset, :]
                _copy_tensor(mem_view, gathered)
            else:
                values = mem_view if mem_view.device == kv_layer.device else mem_view.to(kv_layer.device, non_blocking=True)
                kv_layer[block_idx, block_offset, :] = values
        return

    if gpu_kv_format == GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        block_idx, block_offset = _block_indices(slot_mapping, block_size)
        if direction == TransferDirection.D2H:
            selected = key_value_ptrs[block_idx, :, :, block_offset, :, :].permute(2, 1, 0, 3, 4)
            _copy_tensor(key_value[:, :, mem_offset : mem_offset + slot_mapping.numel()], selected.reshape(2, key_value.shape[1], slot_mapping.numel(), -1))
        else:
            num_heads = key_value_ptrs.shape[-2]
            head_size = key_value_ptrs.shape[-1]
            values = key_value[:, :, mem_offset : mem_offset + slot_mapping.numel()].reshape(2, key_value.shape[1], slot_mapping.numel(), num_heads, head_size)
            if values.device != key_value_ptrs.device or values.dtype != key_value_ptrs.dtype:
                values = values.to(device=key_value_ptrs.device, dtype=key_value_ptrs.dtype, non_blocking=True)
            key_value_ptrs[block_idx, :, :, block_offset, :, :] = values.permute(2, 1, 0, 3, 4)
        return

    kv_layers = _layer_refs(key_value_ptrs)
    block_idx, block_offset = _block_indices(slot_mapping, block_size)
    for layer_id, kv_layer in enumerate(kv_layers):
        mem_view = key_value[:, layer_id, mem_offset : mem_offset + slot_mapping.numel()]
        if direction == TransferDirection.D2H:
            gathered = _gather_non_mla_layer(kv_layer, block_idx, block_offset, gpu_kv_format)
            _copy_tensor(mem_view, gathered)
        else:
            _scatter_non_mla_layer(kv_layer, mem_view, block_idx, block_offset, gpu_kv_format)


def single_layer_kv_transfer(
    lmc_key_value_cache: torch.Tensor,
    vllm_key_value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    direction: TransferDirection | int,
    gpu_kv_format: GPUKVFormat | int,
    token_major: bool = False,
) -> None:
    direction = _as_direction(direction)
    gpu_kv_format = _as_format(gpu_kv_format)
    slot_mapping = _normalize_slot_mapping(slot_mapping)
    if slot_mapping.numel() == 0:
        return

    mem_view = lmc_key_value_cache.permute(1, 0, 2) if token_major else lmc_key_value_cache

    if gpu_kv_format == GPUKVFormat.NL_X_NB_BS_HS:
        block_size = vllm_key_value_cache.shape[1]
        block_idx, block_offset = _block_indices(slot_mapping, block_size)
        if direction == TransferDirection.D2H:
            _copy_tensor(mem_view[0], vllm_key_value_cache[block_idx, block_offset, :])
        else:
            values = mem_view[0]
            if values.device != vllm_key_value_cache.device or values.dtype != vllm_key_value_cache.dtype:
                values = values.to(device=vllm_key_value_cache.device, dtype=vllm_key_value_cache.dtype, non_blocking=True)
            vllm_key_value_cache[block_idx, block_offset, :] = values
        return

    block_size = vllm_key_value_cache.shape[2] if gpu_kv_format == GPUKVFormat.NL_X_TWO_NB_BS_NH_HS else vllm_key_value_cache.shape[2]
    block_idx, block_offset = _block_indices(slot_mapping, block_size)
    if direction == TransferDirection.D2H:
        gathered = _gather_non_mla_layer(vllm_key_value_cache, block_idx, block_offset, gpu_kv_format)
        _copy_tensor(mem_view, gathered)
    else:
        _scatter_non_mla_layer(vllm_key_value_cache, mem_view, block_idx, block_offset, gpu_kv_format)


def single_layer_kv_transfer_sgl(*args, **kwargs):
    raise NotImplementedError("Python fallback currently targets the vLLM connector path only")


def multi_layer_kv_transfer_unilateral(*args, **kwargs):
    raise NotImplementedError("Python fallback currently targets the vLLM connector path only")


def rotary_embedding_k_fused(*args, **kwargs):
    raise NotImplementedError("rotary_embedding_k_fused is not implemented in Python fallback")


def encode_fast_new(*args, **kwargs):
    raise NotImplementedError("CacheGen CUDA ops are not implemented in Python fallback")


def decode_fast_new(*args, **kwargs):
    raise NotImplementedError("CacheGen CUDA ops are not implemented in Python fallback")


def decode_fast_prefsum(*args, **kwargs):
    raise NotImplementedError("CacheGen CUDA ops are not implemented in Python fallback")


def calculate_cdf(*args, **kwargs):
    raise NotImplementedError("CacheGen CUDA ops are not implemented in Python fallback")


def lmcache_memcpy_async(*args, **kwargs):
    raise NotImplementedError(
        "lmcache_memcpy_async is unavailable in Python fallback; use the standard allocator path"
    )
