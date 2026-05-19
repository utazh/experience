
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FidelityLevel(str, Enum):
    FULL = "full"
    BASE = "base"


class FidelityState(str, Enum):
    MISS = "MISS"
    BASE_READY = "BASE_READY"
    PROMOTING = "PROMOTING"
    FULL_READY = "FULL_READY"
    FULL_EVICTED = "FULL_EVICTED"


@dataclass(frozen=True)
class FidelityDecision:
    level: FidelityLevel
    policy: str
    reason: str
