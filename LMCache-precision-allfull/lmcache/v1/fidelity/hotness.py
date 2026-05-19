from __future__ import annotations

import threading


class HotnessTracker:
    """Thread-safe per-prefix access counter for request-driven promotion tests."""

    def __init__(self, min_access_count: int = 2) -> None:
        self.min_access_count = max(1, int(min_access_count))
        self._access_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def record_access(self, prefix_id: str) -> tuple[int, bool]:
        with self._lock:
            count = self._access_counts.get(prefix_id, 0) + 1
            self._access_counts[prefix_id] = count
            return count, count >= self.min_access_count

    def get(self, prefix_id: str) -> int:
        with self._lock:
            return self._access_counts.get(prefix_id, 0)

    def reset(self, prefix_id: str) -> None:
        with self._lock:
            self._access_counts.pop(prefix_id, None)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._access_counts)


__all__ = ["HotnessTracker"]
