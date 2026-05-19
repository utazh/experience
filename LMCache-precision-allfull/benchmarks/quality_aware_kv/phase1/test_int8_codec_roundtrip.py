from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import torch

from lmcache.v1.fidelity.codec import Int8BaseCodec
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat


@dataclass
class CaseResult:
    name: str
    shape: list[int]
    dtype: str
    original_bytes: int
    encoded_bytes: int
    compression_ratio: float
    max_err: float
    mean_err: float


def run_case(name: str, shape: list[int], dtype: torch.dtype, fmt: MemoryFormat) -> CaseResult:
    allocator = AdHocMemoryAllocator(device="cpu")
    codec = Int8BaseCodec()
    memory_obj = allocator.allocate(torch.Size(shape), dtype, fmt=fmt)
    assert memory_obj is not None
    tensor = memory_obj.tensor
    assert tensor is not None

    # Use bounded random values so the error target is meaningful and stable.
    source = torch.randn(shape, dtype=torch.float32).mul_(0.7)
    tensor.copy_(source.to(dtype))

    encoded = codec.encode(allocator, memory_obj)
    decoded = codec.decode(allocator, encoded)
    decoded_tensor = decoded.tensor
    assert decoded_tensor is not None

    original = tensor.to(torch.float32)
    reconstructed = decoded_tensor.to(torch.float32)
    abs_err = (original - reconstructed).abs()

    result = CaseResult(
        name=name,
        shape=shape,
        dtype=str(dtype),
        original_bytes=memory_obj.get_size(),
        encoded_bytes=encoded.get_size(),
        compression_ratio=encoded.get_size() / memory_obj.get_size(),
        max_err=float(abs_err.max().item()),
        mean_err=float(abs_err.mean().item()),
    )

    if decoded is not encoded:
        decoded.ref_count_down()
    if encoded is not memory_obj:
        encoded.ref_count_down()
    memory_obj.ref_count_down()
    return result


def main() -> int:
    torch.manual_seed(1234)
    cases = [
        run_case(
            name="kv_2ltd_cpu",
            shape=[2, 8, 128, 256],
            dtype=torch.bfloat16,
            fmt=MemoryFormat.KV_2LTD,
        ),
        run_case(
            name="kv_t2d_cpu",
            shape=[256, 2, 256],
            dtype=torch.bfloat16,
            fmt=MemoryFormat.KV_T2D,
        ),
    ]

    max_err = max(case.max_err for case in cases)
    max_ratio = max(case.compression_ratio for case in cases)
    payload = {
        "codec": "int8_per_channel_symmetric",
        "seed": 1234,
        "threshold_max_err": 0.02,
        "threshold_max_ratio": 0.55,
        "overall_max_err": max_err,
        "overall_max_ratio": max_ratio,
        "passed": max_err < 0.02 and max_ratio < 0.55,
        "cases": [asdict(case) for case in cases],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
