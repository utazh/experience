from lmcache.v1.fidelity.hotness import HotnessTracker
from lmcache.v1.fidelity.state_store import FidelityStateStore
from lmcache.v1.fidelity.types import FidelityState


def test_fidelity_state_store_transition_is_atomic() -> None:
    store = FidelityStateStore()
    prefix_id = "prefix-a"

    assert store.get(prefix_id) == FidelityState.MISS
    assert store.transition(
        prefix_id,
        FidelityState.MISS,
        FidelityState.BASE_READY,
        reason="prime",
    )
    assert store.get(prefix_id) == FidelityState.BASE_READY
    assert not store.transition(
        prefix_id,
        FidelityState.MISS,
        FidelityState.FULL_READY,
        reason="stale",
    )
    assert store.get(prefix_id) == FidelityState.BASE_READY


def test_hotness_tracker_threshold_and_reset() -> None:
    tracker = HotnessTracker(min_access_count=2)

    assert tracker.record_access("prefix-a") == (1, False)
    assert tracker.record_access("prefix-a") == (2, True)
    tracker.reset("prefix-a")
    assert tracker.get("prefix-a") == 0
