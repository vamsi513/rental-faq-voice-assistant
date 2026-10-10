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
import time
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

log = logging.getLogger("agent.ws")

# Generous for typed questions, small enough that a hostile client cannot make the
# server buffer megabytes before the first check runs.
MAX_MESSAGE_BYTES = 8 * 1024
AUTH_TIMEOUT_SECONDS = 15
CLOSE_POLICY_VIOLATION = 1008


class _Closed(Exception):
    """The peer went away. Raised from the token sink so generation unwinds."""


async def chat_socket(websocket: WebSocket, state: dict, settings: Any) -> None:
    """Serve one connection. `state` is the app's STATE dict."""
    await websocket.accept()
    call_id = f"ws-{uuid.uuid4().hex[:12]}"
    registry = state.get("registry")

    try:
        if not await _authenticate(websocket, settings):
            return
        await websocket.send_json({"type": "ready", "call_id": call_id})
        await _serve_turns(websocket, state, settings, call_id)
    except WebSocketDisconnect:
        log.info("ws disconnected call_id=%s", call_id)
    except _Closed:
        log.info("ws closed mid-stream call_id=%s", call_id)
    except Exception:
        log.exception("ws failed call_id=%s", call_id)
        await _safe_close(websocket, 1011)
    finally:
        # Drop the session so a connection cannot leak a lock or a dedupe table.
        # Checkpoint rows are deliberately left alone; only in-process state is freed.
        if registry is not None:
            await registry.drop(call_id)


async def _authenticate(websocket: WebSocket, settings: Any) -> bool:
    """First frame must be a valid auth. The secret never travels in the URL, where it
    would land in browser history, Referer headers and every proxy log on the way."""
    if not settings.server_secret:
        return True
    try:
        raw = await websocket.receive_text()
    except WebSocketDisconnect:
        return False
    if len(raw.encode()) > MAX_MESSAGE_BYTES:
        await _safe_close(websocket, CLOSE_POLICY_VIOLATION)
        return False
    try:
        message = json.loads(raw)
    except ValueError:
        message = {}
    supplied = message.get("secret") if isinstance(message, dict) else None
    if message.get("type") != "auth" or supplied != settings.server_secret:
        await _safe_close(websocket, CLOSE_POLICY_VIOLATION)
        return False
    return True


async def _serve_turns(websocket: WebSocket, state: dict, settings: Any, call_id: str) -> None:
    position = 0
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
        if kind != "user":
            await websocket.send_json({"type": "error", "message": f"unknown type {kind!r}"})
            continue

        text = (message.get("text") or "").strip()
        if not text:
            await websocket.send_json({"type": "error", "message": "empty text"})
            continue

        await _handle_turn(websocket, call_id, text, position)
        position += 1


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
