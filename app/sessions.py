"""Call -> thread mapping, per-call serialisation, and duplicate-turn handling.

What this is: a correctness guard around the checkpointer, plus bookkeeping.
What this is not: authentication. A `call_id` says "these turns belong to one phone
call". It says nothing about who is on the line. Anyone who can reach the endpoint can
present any call id, so nothing here is treated as proof of identity -- and the agent
only ever reads from one public resume, so there is nothing per-caller to protect.

Three jobs:

1. Thread mapping. One Vapi call id -> one LangGraph thread id. Turns in a call reuse
   the thread (so the agent remembers "that project"); separate calls get separate
   threads and cannot see each other's history.

2. Serialisation. Vapi can have a request in flight when the caller interrupts and a
   new turn starts. Two concurrent runs on one thread would read the same checkpoint and
   write back conflicting versions -- last writer wins and a turn silently vanishes. An
   asyncio lock per call id makes turns on one thread strictly sequential. Different
   calls still run fully in parallel.

3. Duplicate turns. A retried or replayed request would otherwise append the same turn
   twice. Each turn is fingerprinted; a repeat inside the dedupe window replays the
   stored answer instead of re-running the graph.

This registry is in-process. One worker only -- see README "Scaling" for what moves
where if that stops being true.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("agent.sessions")

DEDUPE_WINDOW_SECONDS = 120
MAX_FINGERPRINTS_PER_CALL = 32


@dataclass
class Session:
    call_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    turns: int = 0
    # fingerprint -> (timestamp, answer) for replaying duplicates
    answered: dict[str, tuple[float, str]] = field(default_factory=dict)

    @property
    def thread_id(self) -> str:
        """The LangGraph thread id. Prefixed so a checkpoint row is obviously ours."""
        return f"vapi-call:{self.call_id}"


class SessionRegistry:
    def __init__(self, idle_seconds: int = 1800):
        self._sessions: dict[str, Session] = {}
        self._idle_seconds = idle_seconds
        self._guard = asyncio.Lock()

    async def get(self, call_id: str) -> Session:
        async with self._guard:
            session = self._sessions.get(call_id)
            if session is None:
                session = Session(call_id=call_id)
                self._sessions[call_id] = session
                log.info("session opened call_id=%s thread=%s", call_id, session.thread_id)
            session.last_seen = time.time()
            return session

    @staticmethod
    def fingerprint(call_id: str, question: str, position: int) -> str:
        """Identify a turn by call, position and content.

        `position` must be the CALLER's view of how far the conversation has got -- for
        Vapi, the number of messages it sent in the request body. Using our own
        checkpoint length instead would break the only case this exists for: when Vapi
        retries a request it has not had a response to, its view has not advanced but
        ours already has, so the two would fingerprint differently and the turn would
        be appended twice.

        Position is in the key so that genuinely asking the same question again later
        in a call is treated as a new turn, not a duplicate.
        """
        raw = f"{call_id}|{position}|{' '.join(question.lower().split())}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def replay(self, session: Session, fingerprint: str) -> str | None:
        entry = session.answered.get(fingerprint)
        if entry is None:
            return None
        seen_at, answer = entry
        if time.time() - seen_at > DEDUPE_WINDOW_SECONDS:
            session.answered.pop(fingerprint, None)
            return None
        log.info("duplicate turn replayed call_id=%s fp=%s", session.call_id, fingerprint)
        return answer

    def remember(self, session: Session, fingerprint: str, answer: str) -> None:
        session.answered[fingerprint] = (time.time(), answer)
        session.turns += 1
        if len(session.answered) > MAX_FINGERPRINTS_PER_CALL:
            oldest = sorted(session.answered.items(), key=lambda kv: kv[1][0])
            for key, _ in oldest[: len(oldest) - MAX_FINGERPRINTS_PER_CALL]:
                session.answered.pop(key, None)

    async def drop(self, call_id: str) -> bool:
        """Forget a session immediately, used when a WebSocket closes.

        Frees the lock and the dedupe table. Checkpoint rows are untouched: the
        conversation stays readable through /debug/thread, and a reconnect with the
        same id would resume it. Without this, every socket that ever connected would
        sit in the registry until the idle sweeper ran.
        """
        async with self._guard:
            existed = self._sessions.pop(call_id, None) is not None
        if existed:
            log.info("session dropped call_id=%s", call_id)
        return existed

    async def sweep(self) -> int:
        """Drop idle sessions. Checkpoint rows are untouched -- only locks and
        fingerprints are freed, so a late turn on a swept call still has its history."""
        cutoff = time.time() - self._idle_seconds
        async with self._guard:
            stale = [
                cid for cid, s in self._sessions.items()
                if s.last_seen < cutoff and not s.lock.locked()
            ]
            for cid in stale:
                self._sessions.pop(cid, None)
        if stale:
            log.info("swept %d idle session(s)", len(stale))
        return len(stale)

    def stats(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "sessions": [
                {
                    "call_id": s.call_id,
                    "thread_id": s.thread_id,
                    "turns": s.turns,
                    "age_seconds": round(time.time() - s.created_at, 1),
                    "idle_seconds": round(time.time() - s.last_seen, 1),
                }
                for s in self._sessions.values()
            ],
        }
