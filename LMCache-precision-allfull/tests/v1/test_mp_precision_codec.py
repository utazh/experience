import pytest
import torch

from lmcache.v1.distributed.api import MemoryLayoutDesc, ipc_key_to_object_keys
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.precision_codec import (
    build_mp_base_layout_desc,
    decode_base_memory_obj,
    decode_base_memory_obj_to_tensor,
    encode_memory_obj_to_base,
    encode_tensor_to_base_memory_obj,
    is_mp_int8_base_enabled,
    make_cpu_memory_obj,
    quantization_error_summary,
    namespace_model_name_for_precision,
)


def test_all_full_mp_object_key_keeps_model_name(monkeypatch):
    monkeypatch.delenv("LMCACHE_PRECISION_POLICY", raising=False)
    key = IPCCacheEngineKey.from_token_ids(
        model_name="qwen2-1.5b",
        world_size=1,
        worker_id=0,
        token_ids=list(range(256)),
        start=0,
        end=256,
        request_id="r0",
    )

    object_keys = ipc_key_to_object_keys(key, [b"hash0"])

    assert object_keys[0].model_name == "qwen2-1.5b"


def test_all_base_mp_object_key_uses_precision_namespace(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "all-base")
    monkeypatch.setenv("LMCACHE_ENABLE_FIDELITY_CACHE", "True")
    monkeypatch.setenv("LMCACHE_BASE_CODEC", "int8")
    key = IPCCacheEngineKey.from_token_ids(
        model_name="qwen2-1.5b",
        world_size=1,
        worker_id=0,
        token_ids=list(range(256)),
        start=0,
        end=256,
        request_id="r0",
    )

    object_keys = ipc_key_to_object_keys(key, [b"hash0"])

    assert namespace_model_name_for_precision("qwen2-1.5b") == object_keys[0].model_name
    assert object_keys[0].model_name != "qwen2-1.5b"
    assert "precision=base" in object_keys[0].model_name
    assert "codec=int8" in object_keys[0].model_name


def test_mp_int8_base_layout_halves_bf16_payload(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "all-base")
    monkeypatch.setenv("LMCACHE_ENABLE_FIDELITY_CACHE", "True")
    monkeypatch.setenv("LMCACHE_BASE_CODEC", "int8")
    full = MemoryLayoutDesc(
        shapes=[torch.Size([2, 28, 256, 256])],
        dtypes=[torch.bfloat16],
    )

    encoded = build_mp_base_layout_desc(full)

    assert is_mp_int8_base_enabled()
    assert encoded.shapes == [torch.Size([2, 28, 256, 256]), torch.Size([2, 28, 256, 1])]
    assert encoded.dtypes == [torch.int8, torch.float32]
    full_bytes = sum(s.numel() * d.itemsize for s, d in zip(full.shapes, full.dtypes))
    base_bytes = sum(s.numel() * d.itemsize for s, d in zip(encoded.shapes, encoded.dtypes))
    assert base_bytes / full_bytes == 0.5078125


def test_mp_int8_base_roundtrip_cpu_memory_obj(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "all-base")
    monkeypatch.setenv("LMCACHE_ENABLE_FIDELITY_CACHE", "True")
    monkeypatch.setenv("LMCACHE_BASE_CODEC", "int8")
    full_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 16, 32])],
        dtypes=[torch.bfloat16],
    )
    base_layout = build_mp_base_layout_desc(full_layout)
    full_obj = make_cpu_memory_obj(full_layout, MemoryFormat.KV_2LTD)
    encoded_obj = make_cpu_memory_obj(base_layout, MemoryFormat.KV_2LTD)
    decoded_obj = make_cpu_memory_obj(full_layout, MemoryFormat.KV_2LTD)
    source = torch.randn(full_layout.shapes[0], dtype=torch.bfloat16)
    full_obj.get_tensor(0).copy_(source)

    encode_memory_obj_to_base(full_obj, encoded_obj)
    decode_base_memory_obj(encoded_obj, decoded_obj)

    assert encoded_obj.get_size() < full_obj.get_size()
    err = (source.float() - decoded_obj.get_tensor(0).float()).abs()
    assert float(err.max()) <= 0.02
    assert float(err.mean()) <= 0.01

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mp_int8_gpu_decode_writes_existing_destination(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "all-base")
    monkeypatch.setenv("LMCACHE_ENABLE_FIDELITY_CACHE", "True")
    monkeypatch.setenv("LMCACHE_BASE_CODEC", "int8")
    full_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 3, 32, 64])],
        dtypes=[torch.bfloat16],
    )
    base_layout = build_mp_base_layout_desc(full_layout)
    full_obj = make_cpu_memory_obj(full_layout, MemoryFormat.KV_2LTD)
    encoded_obj = make_cpu_memory_obj(base_layout, MemoryFormat.KV_2LTD)
    source = torch.randn(full_layout.shapes[0], dtype=torch.bfloat16)
    full_obj.get_tensor(0).copy_(source)
    encode_memory_obj_to_base(full_obj, encoded_obj)
    dst = torch.empty(full_layout.shapes[0], dtype=torch.bfloat16, device="cuda")

    returned = decode_base_memory_obj_to_tensor(encoded_obj, dst)
    torch.cuda.synchronize()

    assert returned is dst
    assert dst.device.type == "cuda"
    summary = quantization_error_summary(source, dst.cpu())
    assert summary["cosine_similarity"] >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mp_int8_gpu_encode_roundtrip_passes_quality_gate(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "all-base")
    monkeypatch.setenv("LMCACHE_ENABLE_FIDELITY_CACHE", "True")
    monkeypatch.setenv("LMCACHE_BASE_CODEC", "int8")
    full_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 3, 32, 64])],
        dtypes=[torch.bfloat16],
    )
    base_layout = build_mp_base_layout_desc(full_layout)
    encoded_obj = make_cpu_memory_obj(base_layout, MemoryFormat.KV_2LTD)
    source = torch.randn(full_layout.shapes[0], dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(source)

    encode_tensor_to_base_memory_obj(source, encoded_obj)
    decode_base_memory_obj_to_tensor(encoded_obj, dst)
    torch.cuda.synchronize()

    assert encoded_obj.get_tensor(0).device.type == "cpu"
    assert encoded_obj.get_tensor(0).dtype == torch.int8
    summary = quantization_error_summary(source.cpu(), dst.cpu())
    assert summary["cosine_similarity"] >= 0.999

