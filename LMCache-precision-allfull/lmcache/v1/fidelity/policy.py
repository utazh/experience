
from __future__ import annotations

from typing import Any, Optional

from lmcache.v1.fidelity.types import FidelityDecision, FidelityLevel

STATE_STORE_V0_STATE_KEY = "lmcache.state_store_v0.state"
STATE_STORE_V0_PREFIX_ID_KEY = "lmcache.state_store_v0.prefix_id"
STATE_STORE_V0_SEQUENCE_INDEX_KEY = "lmcache.state_store_v0.sequence_index"
STATE_STORE_V0_ENABLE_PROMOTION_KEY = "lmcache.state_store_v0.enable_promotion"

_STATE_STORE_V0_DECISIONS = {
    "FULL_READY": (FidelityLevel.FULL, "full_ready_hit"),
    "PROMOTING": (FidelityLevel.BASE, "promoting_read_base_no_duplicate"),
    "BASE_READY": (FidelityLevel.BASE, "base_ready_hit"),
    "FULL_EVICTED": (FidelityLevel.FULL, "full_evicted_reload_full_from_slow"),
    "MISS": (FidelityLevel.BASE, "miss_choose_base"),
}


def _decide_state_store_v0(request_configs: dict) -> Optional[FidelityDecision]:
    raw_state = request_configs.get(STATE_STORE_V0_STATE_KEY)
    if raw_state is None:
        raw_state = request_configs.get("lmcache.prefix_state")
    if raw_state is None:
        return None

    state = str(raw_state).strip().upper()
    if state not in _STATE_STORE_V0_DECISIONS:
        raise ValueError(f"Unsupported state_store_v0 state: {raw_state!r}")

    level, reason = _STATE_STORE_V0_DECISIONS[state]
    return FidelityDecision(
        level=level,
        policy="auto_state_store_v0",
        reason=reason,
    )


def decide_fidelity(
    config: Any,
    request_configs: Optional[dict],
    context_len: int,
) -> FidelityDecision:
    request_configs = request_configs or {}
    explicit = request_configs.get("lmcache.fidelity") or request_configs.get("fidelity")
    if explicit in (FidelityLevel.FULL.value, FidelityLevel.BASE.value):
        level = FidelityLevel(explicit)
        return FidelityDecision(
            level=level,
            policy="force_" + level.value,
            reason=f"explicit_{level.value}",
        )

    default_fidelity = getattr(config, "default_fidelity", "full")
    if default_fidelity in (FidelityLevel.FULL.value, FidelityLevel.BASE.value):
        level = FidelityLevel(default_fidelity)
        return FidelityDecision(
            level=level,
            policy="default_" + level.value,
            reason="config_default",
        )

    if default_fidelity != "auto":
        raise ValueError(f"Unsupported default_fidelity: {default_fidelity}")

    state_decision = _decide_state_store_v0(request_configs)
    if state_decision is not None:
        return state_decision

    recent_kv_stall_ms = float(
        request_configs.get(
            "lmcache.recent_kv_stall_ms",
            request_configs.get("recent_kv_stall_ms", 0.0),
        ) or 0.0
    )
    context_threshold = int(getattr(config, "fidelity_auto_context_threshold", 16384))
    kv_stall_threshold = float(
        getattr(config, "fidelity_auto_kv_stall_threshold_ms", 200.0)
    )
    if context_len >= context_threshold and recent_kv_stall_ms >= kv_stall_threshold:
        return FidelityDecision(
            level=FidelityLevel.BASE,
            policy="auto",
            reason="context_and_kv_stall_threshold",
        )
    if context_len >= context_threshold and kv_stall_threshold <= 0.0:
        return FidelityDecision(
            level=FidelityLevel.BASE,
            policy="auto",
            reason="context_threshold_only",
        )
    return FidelityDecision(
        level=FidelityLevel.FULL,
        policy="auto",
        reason="below_auto_threshold",
    )
