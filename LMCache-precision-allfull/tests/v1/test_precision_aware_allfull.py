from lmcache.integration.vllm.precision_aware import (
    PrecisionLoadPlanner,
    PrecisionTier,
    normalize_precision_request_configs,
)


def test_default_policy_is_all_full_without_precision_tag():
    planner = PrecisionLoadPlanner()

    plan = planner.plan(request_configs=None, num_chunks=3)
    normalized = normalize_precision_request_configs(None, plan.default_tier)

    assert plan.policy == "all-full"
    assert plan.default_tier is PrecisionTier.FULL
    assert plan.spans == ()
    assert normalized == {"lmcache.precision_policy": "all-full"}
    assert "lmcache.tag.precision" not in normalized


def test_existing_tags_survive_all_full_normalization():
    request_configs = {
        "lmcache.tag.tenant": "research",
        "lmcache.skip_save": False,
    }
    planner = PrecisionLoadPlanner()

    plan = planner.plan(request_configs=request_configs, num_chunks=2)
    normalized = normalize_precision_request_configs(request_configs, plan.default_tier)

    assert plan.default_tier is PrecisionTier.FULL
    assert normalized["lmcache.tag.tenant"] == "research"
    assert normalized["lmcache.skip_save"] is False
    assert normalized["lmcache.precision_policy"] == "all-full"
    assert "lmcache.tag.precision" not in normalized


def test_all_base_adds_precision_tag_for_future_path():
    planner = PrecisionLoadPlanner()

    plan = planner.plan(
        request_configs={"lmcache.precision_policy": "all-base"},
        num_chunks=2,
    )
    normalized = normalize_precision_request_configs({}, plan.default_tier)

    assert plan.policy == "all-base"
    assert plan.default_tier is PrecisionTier.BASE
    assert normalized["lmcache.precision_policy"] == "all-base"
    assert normalized["lmcache.tag.precision"] == "base"


def test_vllm_adapter_extracts_default_all_full_policy():
    from types import SimpleNamespace

    from lmcache.integration.vllm.vllm_v1_adapter import extract_request_configs

    request_configs = extract_request_configs(SimpleNamespace(extra_args=None))

    assert request_configs == {"lmcache.precision_policy": "all-full"}
    assert "lmcache.tag.precision" not in request_configs


def test_vllm_adapter_preserves_user_configs_with_all_full_policy():
    from types import SimpleNamespace

    from lmcache.integration.vllm.vllm_v1_adapter import extract_request_configs

    sampling_params = SimpleNamespace(
        extra_args={
            "kv_transfer_params": {
                "lmcache.tag.tenant": "research",
                "lmcache.skip_save": False,
            }
        }
    )

    request_configs = extract_request_configs(sampling_params)

    assert request_configs["lmcache.precision_policy"] == "all-full"
    assert request_configs["lmcache.tag.tenant"] == "research"
    assert request_configs["lmcache.skip_save"] is False
    assert "lmcache.tag.precision" not in request_configs


def test_env_policy_defaults_to_all_base_when_request_policy_missing(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "all-base")
    planner = PrecisionLoadPlanner()

    plan = planner.plan(request_configs=None, num_chunks=2)
    normalized = normalize_precision_request_configs(None, plan.default_tier)

    assert plan.policy == "all-base"
    assert plan.default_tier is PrecisionTier.BASE
    assert normalized["lmcache.precision_policy"] == "all-base"
    assert normalized["lmcache.tag.precision"] == "base"


def test_explicit_request_policy_overrides_env_policy(monkeypatch):
    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "all-base")
    planner = PrecisionLoadPlanner()

    plan = planner.plan(
        request_configs={"lmcache.precision_policy": "all-full"},
        num_chunks=2,
    )
    normalized = normalize_precision_request_configs(
        {"lmcache.precision_policy": "all-full"},
        plan.default_tier,
    )

    assert plan.policy == "all-full"
    assert plan.default_tier is PrecisionTier.FULL
    assert normalized == {"lmcache.precision_policy": "all-full"}
    assert "lmcache.tag.precision" not in normalized


def test_base_precision_tag_changes_cache_engine_key_namespace():
    import torch

    from lmcache.utils import CacheEngineKey

    full_key = CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=12345,
        dtype=torch.bfloat16,
        request_configs={"lmcache.precision_policy": "all-full"},
    )
    base_key = CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=12345,
        dtype=torch.bfloat16,
        request_configs={
            "lmcache.precision_policy": "all-base",
            "lmcache.tag.precision": "base",
        },
    )

    assert full_key != base_key
    assert "precision%base" not in full_key.to_string()
    assert "precision%base" in base_key.to_string()


def test_vllm_adapter_uses_env_policy_when_no_request_configs(monkeypatch):
    from types import SimpleNamespace

    from lmcache.integration.vllm.vllm_v1_adapter import extract_request_configs

    monkeypatch.setenv("LMCACHE_PRECISION_POLICY", "all-base")

    request_configs = extract_request_configs(SimpleNamespace(extra_args=None))

    assert request_configs["lmcache.precision_policy"] == "all-base"
    assert request_configs["lmcache.tag.precision"] == "base"

