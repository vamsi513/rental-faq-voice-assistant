"""WebSocket chat at /ws/chat.

A second way into the same LangGraph workflow, alongside the Vapi custom-LLM endpoint.
It exists because the HTTP path cannot stream: Vapi wants one finished answer, so
/chat/completions sends a single SSE chunk. A browser can do better, and this does --
each token is forwarded as the model produces it.

Protocol. Every frame is a JSON object with a `type`.

  client -> server
    {"type":"auth","secret":"..."}      first frame, required when SERVER_SECRET is set
    {"type":"user","text":"..."}        one caller turn
    {"type":"ping"}                     optional keepalive

  server -> client
    {"type":"ready","call_id":"..."}    auth accepted; this is the session id
    {"type":"delta","text":"..."}       a piece of the answer, as it is generated
    {"type":"reset"}                    discard what was streamed; a correction follows
    {"type":"ticket","status":"submitted"|"failed","ticket_id":...,"error":...}
    {"type":"done","route":"...","latency_seconds":n}
    {"type":"error","message":"..."}    recoverable; the socket stays open
    {"type":"pong"}

Close codes: 1008 for a failed or missing auth, and for a frame over the size limit.

Honest notes:
  * Only the grounded-answer path streams token by token. Refusals, emergency
    instructions and maintenance replies are fixed or short strings produced with no
    model call, so they arrive as one delta. The `done` event names the route.
  * A post-generation check can reject an answer that has already been streamed. When
    that happens the server sends `reset` and then the replacement as fresh deltas. A
    client that ignores `reset` will show a retracted answer.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

log = logging.getLogger("agent.ws")

# Only ids this server hands out can be resumed, so a client cannot probe for or
# invent thread names.
_RESUMABLE = re.compile(r"^ws-[0-9a-f]{12}$")

# Generous for typed questions, small enough that a hostile client cannot make the
# server buffer megabytes before the first check runs.
MAX_MESSAGE_BYTES = 8 * 1024
AUTH_TIMEOUT_SECONDS = 15
CLOSE_POLICY_VIOLATION = 1008
CLOSE_TRY_AGAIN_LATER = 1013

# Every turn costs two or three model calls, so an unthrottled client can spend the
# operator's quota as fast as it can type. A sliding window per connection, refused
# with an error frame rather than a close: a human who types quickly should be told
# to wait, not disconnected.
MAX_TURNS_PER_WINDOW = 12
RATE_WINDOW_SECONDS = 60.0

# Backpressure has two parts. Turns on one connection are already strictly
# sequential, because _serve_turns awaits each one before reading the next frame, so
# a client cannot make the server run two graphs at once on one socket. What that
# does NOT bound is the number of sockets, so this caps them globally.
MAX_CONNECTIONS = 32


class _Closed(Exception):
    """The peer went away. Raised from the token sink so generation unwinds."""


class _RateLimiter:
    """Sliding window of turn timestamps for one connection."""

    def __init__(self, limit: int | None = None, window: float | None = None):
        # Read the module constants at construction, not as default arguments.
        # Defaults bind once at class definition, so a changed constant would
        # silently have no effect on new connections.
        self._limit = MAX_TURNS_PER_WINDOW if limit is None else limit
        self._window = RATE_WINDOW_SECONDS if window is None else window
        self._hits: list[float] = []

    def check(self) -> float:
        """Return 0.0 if a turn is allowed now, else seconds until one is."""
        now = time.monotonic()
        self._hits = [h for h in self._hits if now - h < self._window]
        if len(self._hits) < self._limit:
            self._hits.append(now)
            return 0.0
        return round(self._window - (now - self._hits[0]), 1)


# Live connection count, for the global cap. Single process, so a plain int is
# enough; a second worker would need this in Redis alongside the session registry.
_connections = 0


async def chat_socket(websocket: WebSocket, state: dict, settings: Any) -> None:
    """Serve one connection. `state` is the app's STATE dict."""
    global _connections

    await websocket.accept()
    if _connections >= MAX_CONNECTIONS:
        await websocket.send_json({"type": "error", "message": "server at capacity"})
        await _safe_close(websocket, CLOSE_TRY_AGAIN_LATER)
        return
    _connections += 1

    call_id = f"ws-{uuid.uuid4().hex[:12]}"
    registry = state.get("registry")

    try:
        # With no secret there is nothing to authenticate, and blocking for an auth
        # frame would hang a client that correctly never sends one. Such a client
        # resumes with a `resume` frame instead, handled in the turn loop.
        if settings.server_secret:
            resumed = await _authenticate(websocket, settings)
            if resumed is None:
                return
        else:
            resumed = ""
        if resumed:
            call_id = resumed
        await websocket.send_json({
            "type": "ready", "call_id": call_id, "resumed": bool(resumed),
        })
        await _serve_turns(websocket, state, settings, call_id)
    except WebSocketDisconnect:
        log.info("ws disconnected call_id=%s", call_id)
    except _Closed:
        log.info("ws closed mid-stream call_id=%s", call_id)
    except Exception:
        log.exception("ws failed call_id=%s", call_id)
        await _safe_close(websocket, 1011)
    finally:
        _connections -= 1
        # Drop the session so a connection cannot leak a lock or a dedupe table.
        # Checkpoint rows are deliberately left alone, which is what makes a resume
        # possible: only in-process state is freed.
        if registry is not None:
            await registry.drop(call_id)


async def _authenticate(websocket: WebSocket, settings: Any) -> str | None:
    """Validate the first frame.

    Only called when a secret is configured. Returns None if the connection was
    rejected, an empty string for a fresh session, or a call id to resume. The
    secret never travels in the URL, where it would land in browser history, Referer
    headers and every proxy log on the way.

    A resume is accepted on the caller's word alone. With a single shared secret
    there is nothing else to check it against: anyone holding the secret can resume
    any thread. That is fine for one operator's prototype and is stated in the
    README, but it is not multi-tenant and must not be presented as such.
    """
    try:
        raw = await websocket.receive_text()
    except WebSocketDisconnect:
        return None
    if len(raw.encode()) > MAX_MESSAGE_BYTES:
        await _safe_close(websocket, CLOSE_POLICY_VIOLATION)
        return None
    try:
        message = json.loads(raw)
    except ValueError:
        message = {}
    if not isinstance(message, dict):
        message = {}

    supplied = message.get("secret")
    if message.get("type") != "auth" or supplied != settings.server_secret:
        await _safe_close(websocket, CLOSE_POLICY_VIOLATION)
        return None

    resume = message.get("resume")
    if resume is None:
        return ""
    if not isinstance(resume, str) or not _RESUMABLE.match(resume):
        await websocket.send_json({
            "type": "error", "message": "resume must be a call id this server issued",
        })
        return ""
    return resume


async def _serve_turns(websocket: WebSocket, state: dict, settings: Any, call_id: str) -> None:
    # A resumed thread already has turns in its checkpoint. Position feeds the
    # duplicate check, so starting from zero on a resume would make the first new
    # turn look like a replay of the first original one.
    position = await _turns_so_far(state, call_id)
    limiter = _RateLimiter()
    turns_taken = 0
    while True:
        raw = await websocket.receive_text()
        if len(raw.encode()) > MAX_MESSAGE_BYTES:
            await _safe_close(websocket, CLOSE_POLICY_VIOLATION)
            return
        try:
            message = json.loads(raw)
        except ValueError:
            await websocket.send_json({"type": "error", "message": "frame was not JSON"})
            continue
        if not isinstance(message, dict):
            await websocket.send_json({"type": "error", "message": "frame was not an object"})
            continue

        kind = message.get("type")
        if kind == "ping":
            await websocket.send_json({"type": "pong"})
            continue
        if kind == "resume":
            # Allowed only before the first turn: switching threads mid conversation
            # would silently discard what the caller already said.
            wanted = message.get("call_id")
            if turns_taken:
                await websocket.send_json({
                    "type": "error", "message": "resume must come before the first turn",
                })
                continue
            if not isinstance(wanted, str) or not _RESUMABLE.match(wanted):
                await websocket.send_json({
                    "type": "error", "message": "resume must be a call id this server issued",
                })
                continue
            call_id = wanted
            position = await _turns_so_far(state, call_id)
            await websocket.send_json({
                "type": "ready", "call_id": call_id, "resumed": True, "turns": position,
            })
            continue
        if kind != "user":
            await websocket.send_json({"type": "error", "message": f"unknown type {kind!r}"})
            continue

        text = (message.get("text") or "").strip()
        if not text:
            await websocket.send_json({"type": "error", "message": "empty text"})
            continue

        retry_after = limiter.check()
        if retry_after:
            # Refused, not disconnected: a human typing quickly should be told to
            # wait. Every turn costs model calls, so this is a spend guard first.
            await websocket.send_json({
                "type": "error",
                "message": f"rate limit reached, try again in {retry_after} seconds",
                "retry_after_seconds": retry_after,
            })
            continue

        await _handle_turn(websocket, call_id, text, position)
        position += 1
        turns_taken += 1


async def _turns_so_far(state: dict, call_id: str) -> int:
    """How many turns the checkpoint already holds for this thread.

    Zero for a fresh id, and zero if the state cannot be read, which is the safe
    direction: a wrong-but-low position can only make a genuine turn look new, never
    make a new turn look like a duplicate.
    """
    graph = state.get("graph")
    if graph is None:
        return 0
    try:
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": f"vapi-call:{call_id}"}}
        )
        return len((snapshot.values or {}).get("messages", []) or []) // 2
    except Exception:
        log.exception("could not read prior turns for %s", call_id)
        return 0


async def _handle_turn(websocket: WebSocket, call_id: str, text: str, position: int) -> None:
    """Run one turn, streaming it to the client."""
    from app.graph import TOKEN_SINK
    from app.server import run_turn

    streamed: list[str] = []

    async def on_token(piece: str) -> None:
        # Sending is also how a disconnect is noticed: the send raises, _Closed unwinds
        # out of the model stream, and generation stops. There is no separate
        # cancellation path to keep in sync with this one.
        if websocket.client_state is not WebSocketState.CONNECTED:
            raise _Closed
        try:
            await websocket.send_json({"type": "delta", "text": piece})
        except (WebSocketDisconnect, RuntimeError) as exc:
            raise _Closed from exc
        streamed.append(piece)

    started = time.perf_counter()
    token = TOKEN_SINK.set(on_token)
    try:
        result = await run_turn(call_id, text, position=position)
    finally:
        TOKEN_SINK.reset(token)

    answer = result.get("answer", "")
    # Routes that make no model call produce their text whole; send it as one delta so
    # the client does not have to special-case them.
    if not streamed:
        if answer:
            await websocket.send_json({"type": "delta", "text": answer})
    elif "".join(streamed) != answer:
        # A post-check replaced what was generated. Retract and resend.
        await websocket.send_json({"type": "reset"})
        await websocket.send_json({"type": "delta", "text": answer})

    maintenance = result.get("maintenance") or {}
    if maintenance.get("status") in ("submitted", "failed"):
        await websocket.send_json({
            "type": "ticket",
            "status": maintenance["status"],
            "ticket_id": maintenance.get("ticket_id"),
            "error": maintenance.get("error"),
        })

    await websocket.send_json({
        "type": "done",
        "route": result.get("route"),
        "latency_seconds": round(time.perf_counter() - started, 3),
    })


async def _safe_close(websocket: WebSocket, code: int) -> None:
    if websocket.client_state is WebSocketState.CONNECTED:
        try:
            await websocket.close(code=code)
        except RuntimeError:
            pass
