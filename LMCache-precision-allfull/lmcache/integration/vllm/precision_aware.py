# SPDX-License-Identifier: Apache-2.0
"""Precision-aware KV fetch planning helpers for vLLM integration.

The first runnable phase is intentionally conservative: the default
``all-full`` policy does not add a precision tag, so it uses the same
LMCache key namespace as the existing full-precision path.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


_PRECISION_POLICY_KEY = "lmcache.precision_policy"
_PRECISION_TAG_KEY = "lmcache.tag.precision"
_PRECISION_POLICY_ENV = "LMCACHE_PRECISION_POLICY"


class PrecisionTier(str, Enum):
    """KV precision tier selected for a chunk or request."""

    FULL = "full"
    BASE = "base"


@dataclass(frozen=True)
class PrecisionSpan:
    """A contiguous chunk range sharing the same precision tier."""

    start_chunk: int
    end_chunk: int
    tier: PrecisionTier


@dataclass(frozen=True)
class PrecisionPlan:
    """Precision fetch plan for a request."""

    default_tier: PrecisionTier
    spans: tuple[PrecisionSpan, ...] = ()
    policy: str = "all-full"
    score_threshold: Optional[float] = None


def _default_policy() -> str:
    return os.environ.get(_PRECISION_POLICY_ENV) or "all-full"


def _coerce_policy(request_configs: Optional[dict]) -> str:
    if not request_configs:
        return _default_policy()
    policy = request_configs.get(_PRECISION_POLICY_KEY)
    if policy in (None, ""):
        return _default_policy()
    return str(policy)


def normalize_precision_request_configs(
    request_configs: Optional[dict],
    tier: PrecisionTier,
) -> dict:
    """Return request configs for the selected precision namespace.

    ``FULL`` deliberately does not add ``lmcache.tag.precision`` so the
    first all-full framework remains key-compatible with ordinary LMCache.
    ``BASE`` adds a tag because LMCache includes ``lmcache.tag.*`` values in
    ``CacheEngineKey`` identity.
    """

    normalized = dict(request_configs or {})
    if tier is PrecisionTier.FULL:
        normalized.pop(_PRECISION_TAG_KEY, None)
        normalized[_PRECISION_POLICY_KEY] = "all-full"
        return normalized

    if tier is PrecisionTier.BASE:
        normalized[_PRECISION_TAG_KEY] = "base"
        normalized[_PRECISION_POLICY_KEY] = "all-base"
        return normalized

    raise ValueError(f"Unsupported precision tier: {tier}")


class PrecisionLoadPlanner:
    """Build the first-phase request-level precision plan."""

    def plan(
        self,
        request_configs: Optional[dict],
        num_chunks: int,
    ) -> PrecisionPlan:
        if num_chunks < 0:
            raise ValueError(f"num_chunks must be non-negative, got {num_chunks}")

        policy = _coerce_policy(request_configs)
        if policy == "all-full":
            return PrecisionPlan(
                default_tier=PrecisionTier.FULL,
                policy="all-full",
            )
        if policy == "all-base":
            return PrecisionPlan(
                default_tier=PrecisionTier.BASE,
                policy="all-base",
            )
        if policy in {"mixed-3span", "threshold-mixed"}:
            return PrecisionPlan(
                default_tier=PrecisionTier.FULL,
                policy=policy,
            )
        raise ValueError(f"Unsupported precision policy: {policy}")
