import pytest
import torch

from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.precision_codec import (
    MP_BASE_TIER,
    MP_FULL_TIER,
    build_mixed_3span_tiers,
    build_threshold_mixed_tiers,
    group_precision_tier_spans,
    normalize_threshold_mixed_tiers,
    precision_tiers_for_range,
    namespace_model_name_for_precision,
)


def _key(num_chunks: int, worker_id=0):
    num_tokens = num_chunks * 256
    return IPCCacheEngineKey.from_token_ids(
        model_name="qwen2-1.5b",
        world_size=1,
        worker_id=worker_id,
        token_ids=list(range(num_tokens)),
        start=0,
        end=num_tokens,
        request_id="mixed-r0",
    )


def test_mixed_3span_policy_for_eight_chunks():
    tiers = build_mixed_3span_tiers(total_chunks=8, sink_chunks=1, recent_chunks=2)

    assert tiers == (
        MP_FULL_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_FULL_TIER,
        MP_FULL_TIER,
    )
    assert group_precision_tier_spans(tiers) == (
        (0, 1, MP_FULL_TIER),
        (1, 6, MP_BASE_TIER),
        (6, 8, MP_FULL_TIER),
    )


def test_mixed_object_keys_use_per_chunk_precision_namespace(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "mixed-3span")
    monkeypatch.setenv("LMCACHE_ENABLE_FIDELITY_CACHE", "True")
    monkeypatch.setenv("LMCACHE_BASE_CODEC", "int8")
    key = _key(num_chunks=8)
    hashes = [f"hash{i}".encode() for i in range(8)]
    tiers = build_mixed_3span_tiers(total_chunks=8, sink_chunks=1, recent_chunks=2)

    object_keys = ipc_key_to_object_keys(key, hashes, precision_tiers=tiers)

    model_names = [k.model_name for k in object_keys]
    assert model_names[0] == "qwen2-1.5b"
    assert model_names[1:6] == [
        namespace_model_name_for_precision("qwen2-1.5b", MP_BASE_TIER)
    ] * 5
    assert model_names[6:] == ["qwen2-1.5b", "qwen2-1.5b"]


def test_mixed_world_size_expansion_keeps_chunk_tier_order(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "mixed-3span")
    monkeypatch.setenv("LMCACHE_ENABLE_FIDELITY_CACHE", "True")
    monkeypatch.setenv("LMCACHE_BASE_CODEC", "int8")
    key = IPCCacheEngineKey.from_token_ids(
        model_name="qwen2-1.5b",
        world_size=2,
        worker_id=None,
        token_ids=list(range(4 * 256)),
        start=0,
        end=4 * 256,
        request_id="mixed-r1",
    )
    hashes = [f"hash{i}".encode() for i in range(4)]
    tiers = (MP_FULL_TIER, MP_BASE_TIER, MP_BASE_TIER, MP_FULL_TIER)

    object_keys = ipc_key_to_object_keys(key, hashes, precision_tiers=tiers)

    assert len(object_keys) == 8
    assert [k.model_name for k in object_keys[:2]] == ["qwen2-1.5b"] * 2
    assert all("precision=base" in k.model_name for k in object_keys[2:6])
    assert [k.model_name for k in object_keys[6:]] == ["qwen2-1.5b"] * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mixed_slot_order_cosine_validation_with_known_chunks(monkeypatch):
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.multiprocess.precision_codec import (
        build_mp_base_layout_desc,
        decode_base_memory_obj_to_tensor,
        encode_tensor_to_base_memory_obj,
        make_cpu_memory_obj,
        quantization_error_summary,
    )

    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "mixed-3span")
    monkeypatch.setenv("LMCACHE_ENABLE_FIDELITY_CACHE", "True")
    monkeypatch.setenv("LMCACHE_BASE_CODEC", "int8")
    total_chunks = 8
    chunk_size = 32
    shape = torch.Size([2, 2, total_chunks * chunk_size, 64])
    source = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    restored = torch.empty_like(source)
    tiers = build_mixed_3span_tiers(total_chunks, sink_chunks=1, recent_chunks=2)

    for chunk_idx, tier in enumerate(tiers):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        src_chunk = source[:, :, start:end, :]
        dst_chunk = restored[:, :, start:end, :]
        if tier == MP_FULL_TIER:
            dst_chunk.copy_(src_chunk)
        else:
            layout = MemoryLayoutDesc(shapes=[src_chunk.shape], dtypes=[src_chunk.dtype])
            encoded = make_cpu_memory_obj(
                build_mp_base_layout_desc(layout), MemoryFormat.KV_2LTD
            )
            encode_tensor_to_base_memory_obj(src_chunk, encoded)
            decode_base_memory_obj_to_tensor(encoded, dst_chunk)
    torch.cuda.synchronize()

    for chunk_idx, tier in enumerate(tiers):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        summary = quantization_error_summary(
            source[:, :, start:end, :].cpu(), restored[:, :, start:end, :].cpu()
        )
        if tier == MP_FULL_TIER:
            assert summary["cosine_similarity"] >= 0.999999
            assert summary["max_abs_error"] == 0.0
        else:
            assert summary["cosine_similarity"] >= 0.999


def test_threshold_mixed_full_chunks_are_monotonic_non_increasing():
    counts = []
    for threshold in (0.25, 0.50, 0.75):
        tiers = build_threshold_mixed_tiers(
            total_chunks=8,
            threshold=threshold,
            sink_chunks=1,
        )
        counts.append(tiers.count(MP_FULL_TIER))

    assert counts == [8, 6, 4]
    assert counts[0] >= counts[1] >= counts[2]


def test_threshold_mixed_sink_override_survives_high_threshold():
    tiers = build_threshold_mixed_tiers(
        total_chunks=8,
        threshold=0.9,
        sink_chunks=1,
    )

    assert tiers == (
        MP_FULL_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_FULL_TIER,
    )
    assert group_precision_tier_spans(tiers) == (
        (0, 1, MP_FULL_TIER),
        (1, 7, MP_BASE_TIER),
        (7, 8, MP_FULL_TIER),
    )


def test_threshold_mixed_invalid_span_shape_falls_back_to_all_full():
    tiers = normalize_threshold_mixed_tiers(
        (MP_BASE_TIER, MP_FULL_TIER, MP_BASE_TIER, MP_FULL_TIER)
    )

    assert tiers == (MP_FULL_TIER,) * 4


def test_threshold_mixed_policy_uses_env_threshold(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "threshold-mixed")
    monkeypatch.setenv("LMCACHE_PRECISION_THRESHOLD", "0.75")
    monkeypatch.setenv("LMCACHE_ENABLE_FIDELITY_CACHE", "True")
    monkeypatch.setenv("LMCACHE_BASE_CODEC", "int8")

    tiers = precision_tiers_for_range(total_chunks=8, start_chunk=0, end_chunk=8)

    assert tiers == (
        MP_FULL_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_BASE_TIER,
        MP_FULL_TIER,
        MP_FULL_TIER,
        MP_FULL_TIER,
    )

