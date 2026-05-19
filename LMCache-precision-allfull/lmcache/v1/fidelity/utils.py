
from __future__ import annotations

from typing import Any, Optional, Tuple

from lmcache.v1.fidelity.policy import (
    STATE_STORE_V0_ENABLE_PROMOTION_KEY,
    STATE_STORE_V0_PREFIX_ID_KEY,
    STATE_STORE_V0_SEQUENCE_INDEX_KEY,
    STATE_STORE_V0_STATE_KEY,
    decide_fidelity,
)
from lmcache.v1.fidelity.types import FidelityDecision, FidelityLevel

REQUEST_FIDELITY_KEY = "lmcache.fidelity"
REQUEST_FIDELITY_POLICY_KEY = "lmcache.fidelity_policy"
REQUEST_FIDELITY_REASON_KEY = "lmcache.fidelity_reason"
REQUEST_FIDELITY_TAG_KEY = "lmcache.tag.fidelity"


def normalize_request_configs(
    config: Any,
    request_configs: Optional[dict],
    context_len: int,
) -> Tuple[Optional[dict], Optional[FidelityDecision]]:
    if request_configs is not None and len(request_configs) != 0:
        normalized = dict(request_configs)
    else:
        normalized = {}

    if not getattr(config, "enable_fidelity_cache", False):
        return request_configs, None

    decision = decide_fidelity(config, normalized, context_len)
    normalized[REQUEST_FIDELITY_KEY] = decision.level.value
    normalized[REQUEST_FIDELITY_POLICY_KEY] = decision.policy
    normalized[REQUEST_FIDELITY_REASON_KEY] = decision.reason
    normalized[REQUEST_FIDELITY_TAG_KEY] = decision.level.value
    return normalized, decision


def request_configs_for_level(request_configs: Optional[dict], level: FidelityLevel) -> dict:
    normalized = dict(request_configs or {})
    normalized[REQUEST_FIDELITY_KEY] = level.value
    normalized[REQUEST_FIDELITY_TAG_KEY] = level.value
    return normalized


def get_fidelity_level(request_configs: Optional[dict]) -> Optional[FidelityLevel]:
    if not request_configs:
        return None
    level = request_configs.get(REQUEST_FIDELITY_KEY)
    if level not in (FidelityLevel.FULL.value, FidelityLevel.BASE.value):
        return None
    return FidelityLevel(level)
