#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch

from lmcache.v1.fidelity.codec import Int8BaseCodec
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat


@dataclass
class BenchmarkResult:
    mode: str
    path: str
    context_len: int
    object_count: int
    shape: list[int]
    dtype: str
    encode_ms_total: float
    decode_ms_total: float
    encode_ms_per_object: float
    decode_ms_per_object: float


def time_fn(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return sum(times) / len(times)


def tensor_mode_bench(codec: Int8BaseCodec, tensor: torch.Tensor, repeat: int, warmup: int, iters: int):
    quantized_ref, scale_ref = codec.encode_base_tensor(tensor)

    def encode_only():
        for _ in range(repeat):
            codec.encode_base_tensor(tensor)

    def decode_only():
        for _ in range(repeat):
            codec.decode_base_tensor(quantized_ref, scale_ref, tensor.dtype)

    return time_fn(encode_only, warmup, iters), time_fn(decode_only, warmup, iters)


def memory_obj_mode_bench(codec: Int8BaseCodec, shape: list[int], dtype: torch.dtype, fmt: MemoryFormat, repeat: int, warmup: int, iters: int):
    allocator = AdHocMemoryAllocator(device="cpu")
    sample = allocator.allocate(torch.Size(shape), dtype, fmt=fmt)
    assert sample is not None
    sample_tensor = sample.tensor
    assert sample_tensor is not None
    sample_tensor.copy_(torch.randn(shape, dtype=torch.float32).to(dtype))
    encoded_ref = codec.encode(allocator, sample)

    def encode_only():
        for _ in range(repeat):
            encoded = codec.encode(allocator, sample)
            if encoded is not sample:
                encoded.ref_count_down()

    def decode_only():
        for _ in range(repeat):
            decoded = codec.decode(allocator, encoded_ref)
            if decoded is not encoded_ref:
                decoded.ref_count_down()

    encode_ms = time_fn(encode_only, warmup, iters)
    decode_ms = time_fn(decode_only, warmup, iters)

    if encoded_ref is not sample:
        encoded_ref.ref_count_down()
    sample.ref_count_down()
    return encode_ms, decode_ms


def make_results(context_len: int, chunk_size: int, num_layers: int, hidden_dim: int, dtype: torch.dtype, warmup: int, iters: int) -> list[BenchmarkResult]:
    codec = Int8BaseCodec()
    chunk_count = context_len // chunk_size
    o1_shape = [2, num_layers, chunk_size, hidden_dim]
    o2_shape = [chunk_size, 2, hidden_dim]
    o1_tensor = torch.randn(o1_shape, dtype=torch.float32).to(dtype)
    o2_tensor = torch.randn(o2_shape, dtype=torch.float32).to(dtype)

    configs = [
        ("tensor_only", "o1_non_layerwise", o1_shape, chunk_count, MemoryFormat.KV_2LTD, o1_tensor),
        ("tensor_only", "o2_layerwise", o2_shape, chunk_count * num_layers, MemoryFormat.KV_T2D, o2_tensor),
        ("memory_obj_boundary", "o1_non_layerwise", o1_shape, chunk_count, MemoryFormat.KV_2LTD, None),
        ("memory_obj_boundary", "o2_layerwise", o2_shape, chunk_count * num_layers, MemoryFormat.KV_T2D, None),
    ]

    results: list[BenchmarkResult] = []
    for mode, path, shape, repeat, fmt, tensor in configs:
        if mode == "tensor_only":
            encode_ms, decode_ms = tensor_mode_bench(codec, tensor, repeat, warmup, iters)
        else:
            encode_ms, decode_ms = memory_obj_mode_bench(codec, shape, dtype, fmt, repeat, warmup, iters)
        results.append(
            BenchmarkResult(
                mode=mode,
                path=path,
                context_len=context_len,
                object_count=repeat,
                shape=shape,
                dtype=str(dtype),
                encode_ms_total=round(encode_ms, 3),
                decode_ms_total=round(decode_ms, 3),
                encode_ms_per_object=round(encode_ms / repeat, 6),
                decode_ms_per_object=round(decode_ms / repeat, 6),
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-lens", nargs="+", type=int, default=[16384, 32512])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1234)

    all_results = []
    for context_len in args.context_lens:
        if context_len % args.chunk_size != 0:
            raise ValueError(f"context_len={context_len} must be divisible by chunk_size={args.chunk_size}")
        all_results.extend(
            make_results(
                context_len=context_len,
                chunk_size=args.chunk_size,
                num_layers=args.num_layers,
                hidden_dim=args.hidden_dim,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    payload = {
        "event": "int8_codec_cost_benchmark",
        "dtype": str(dtype),
        "chunk_size": args.chunk_size,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": [asdict(r) for r in all_results],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
