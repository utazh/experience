from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from lmcache.v1.fidelity.types import FidelityState


@dataclass(frozen=True)
class FidelityStateRecord:
    state: FidelityState
    updated_at: float
    reason: str


class FidelityStateStore:
    """Thread-safe in-memory prefix fidelity state store."""

    def __init__(self) -> None:
        self._states: dict[str, FidelityStateRecord] = {}
        self._lock = threading.Lock()

    def get(self, prefix_id: str) -> FidelityState:
        with self._lock:
            record = self._states.get(prefix_id)
            return FidelityState.MISS if record is None else record.state

    def get_record(self, prefix_id: str) -> FidelityStateRecord | None:
        with self._lock:
            return self._states.get(prefix_id)

    def set(self, prefix_id: str, state: FidelityState, *, reason: str = "set") -> FidelityState:
        with self._lock:
            previous = self._states.get(prefix_id)
            self._states[prefix_id] = FidelityStateRecord(state=state, updated_at=time.time(), reason=reason)
            return FidelityState.MISS if previous is None else previous.state

    def transition(self, prefix_id: str, from_state: FidelityState, to_state: FidelityState, *, reason: str = "transition") -> bool:
        with self._lock:
            current = self._states.get(prefix_id)
            current_state = FidelityState.MISS if current is None else current.state
            if current_state != from_state:
                return False
            self._states[prefix_id] = FidelityStateRecord(state=to_state, updated_at=time.time(), reason=reason)
            return True

    def snapshot(self) -> dict[str, FidelityStateRecord]:
        with self._lock:
            return dict(self._states)


__all__ = ["FidelityStateRecord", "FidelityStateStore"]
