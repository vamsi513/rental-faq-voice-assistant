"""Session registry: thread mapping, serialisation, duplicate turns, cleanup."""
from __future__ import annotations

import asyncio

import pytest

from app.sessions import DEDUPE_WINDOW_SECONDS, MAX_FINGERPRINTS_PER_CALL, SessionRegistry

pytestmark = pytest.mark.asyncio


async def test_one_call_maps_to_one_thread():
    registry = SessionRegistry()
    first = await registry.get("call-abc")
    second = await registry.get("call-abc")
    assert first is second
    assert first.thread_id == "vapi-call:call-abc"


async def test_separate_calls_get_separate_threads():
    registry = SessionRegistry()
    a = await registry.get("call-a")
    b = await registry.get("call-b")
    assert a.thread_id != b.thread_id


async def test_fingerprint_is_stable_for_a_retry_and_changes_with_position():
    registry = SessionRegistry()
    # A Vapi retry: same call, same text, same caller-side position.
    retry = registry.fingerprint("c", "What is the pet fee?", 0)
    assert registry.fingerprint("c", "What is the pet fee?", 0) == retry
    # Same question later in the call is a genuine new turn.
    assert registry.fingerprint("c", "What is the pet fee?", 2) != retry
    # Same question on another call is a different turn.
    assert registry.fingerprint("other", "What is the pet fee?", 0) != retry


async def test_whitespace_and_case_do_not_defeat_the_duplicate_check():
    registry = SessionRegistry()
    assert registry.fingerprint("c", "What  is   the pet fee?", 0) == \
           registry.fingerprint("c", "what is the pet fee?", 0)


async def test_duplicate_is_replayed_not_recomputed():
    registry = SessionRegistry()
    session = await registry.get("c")
    fp = registry.fingerprint("c", "q", 0)
    assert registry.replay(session, fp) is None
    registry.remember(session, fp, "the answer")
    assert registry.replay(session, fp) == "the answer"


async def test_replay_expires_outside_the_window():
    registry = SessionRegistry()
    session = await registry.get("c")
    fp = registry.fingerprint("c", "q", 0)
    registry.remember(session, fp, "old")
    # Backdate past the window.
    stamp, answer = session.answered[fp]
    session.answered[fp] = (stamp - DEDUPE_WINDOW_SECONDS - 1, answer)
    assert registry.replay(session, fp) is None


async def test_fingerprint_memory_is_bounded():
    registry = SessionRegistry()
    session = await registry.get("c")
    for i in range(MAX_FINGERPRINTS_PER_CALL * 3):
        registry.remember(session, registry.fingerprint("c", f"q{i}", i), "a")
    assert len(session.answered) <= MAX_FINGERPRINTS_PER_CALL


async def test_per_call_lock_serialises_overlapping_turns():
    """Two concurrent turns on one call must not interleave, or they would read the
    same checkpoint and write back conflicting versions."""
    registry = SessionRegistry()
    session = await registry.get("c")
    order: list[str] = []

    async def turn(name: str):
        async with session.lock:
            order.append(f"{name}-start")
            await asyncio.sleep(0.02)
            order.append(f"{name}-end")

    await asyncio.gather(turn("A"), turn("B"))
    # Whichever ran first must finish before the other starts.
    assert order in (
        ["A-start", "A-end", "B-start", "B-end"],
        ["B-start", "B-end", "A-start", "A-end"],
    )


async def test_different_calls_are_not_serialised_against_each_other():
    registry = SessionRegistry()
    a, b = await registry.get("a"), await registry.get("b")
    async with a.lock:
        # b must still be acquirable while a is held.
        assert not b.lock.locked()
        async with b.lock:
            assert a.lock.locked() and b.lock.locked()


async def test_idle_sessions_are_swept_but_busy_ones_are_kept():
    registry = SessionRegistry(idle_seconds=0)
    await registry.get("idle")
    busy = await registry.get("busy")
    async with busy.lock:
        swept = await registry.sweep()
    assert swept == 1
    assert "idle" not in [s["call_id"] for s in registry.stats()["sessions"]]


async def test_stats_reports_thread_mapping():
    registry = SessionRegistry()
    await registry.get("call-1")
    stats = registry.stats()
    assert stats["active_sessions"] == 1
    assert stats["sessions"][0]["thread_id"] == "vapi-call:call-1"
