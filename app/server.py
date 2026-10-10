"""FastAPI server.

The Vapi integration point is POST /chat/completions.

NOTE on "OpenAI-compatible": this is the OpenAI Chat Completions wire format -- the
request/response JSON shape -- not an OpenAPI specification. The two are unrelated. Vapi's "Custom LLM" provider
speaks the OpenAI chat-completions protocol: it POSTs an OpenAI-shaped body (plus a
`call` object carrying the call id) and expects either a JSON completion or an SSE
stream of `chat.completion.chunk` objects terminated by `data: [DONE]`. Replying on
that path is what makes the agent's words reach the caller -- Vapi feeds the text
straight into text-to-speech. A webhook that only logs events would not speak.

Verified against Vapi's docs and its own example server (VapiAI/server-example-
serverless-vercel reads exactly `model, messages, max_tokens, temperature, stream,
call` off the body).

Other routes exist for the demo and for debugging, not for Vapi:
  GET  /health      readiness plus the resolved config
  GET  /debug/last  evidence and routing for the last turn of a call
  GET  /debug/chunks  the whole indexed corpus
  POST /text        same graph, text in/text out -- how the tests and eval drive it
  GET  /            the browser call page
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import GREETING, PROPERTY_NAME, PROTOTYPE_NOTICE, RENTAL_CORPUS_DIR, ROOT, settings
from app.graph import build_graph, new_turn_id
from app.sessions import SessionRegistry

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
)
log = logging.getLogger("agent.server")

app = FastAPI(title="Rental FAQ voice assistant (prototype)", version="2.0.0")

# Populated in the lifespan handler.
STATE: dict[str, Any] = {"graph": None, "retriever": None, "registry": None, "ready": False}
# Last completed turn per call, for the debug panel. Bounded; see _remember_debug.
LAST_TURN: dict[str, dict] = {}
MAX_DEBUG_CALLS = 50


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    import httpx
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.embeddings import CachedEmbedder, HashingEmbedder, OpenAIEmbedder
    from app.retriever import PolicyRetriever

    if settings.openai_api_key:
        embedder = CachedEmbedder(
            OpenAIEmbedder(settings.openai_api_key, settings.embedding_model, settings.embedding_dim),
            ROOT / "data/emb_cache.json",
        )
        llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            timeout=12,      # a voice caller will not wait longer than this
            max_retries=1,   # one retry; a second would push past the caller's patience
        )
    else:
        # Lets the server boot for UI work without a key. Answers will be useless.
        log.warning("OPENAI_API_KEY missing -- offline stand-in embedder, no generation")
        embedder, llm = HashingEmbedder(), None

    retriever = PolicyRetriever.from_corpus(
        RENTAL_CORPUS_DIR, embedder,
        hybrid=settings.hybrid_retrieval, dense_weight=settings.dense_weight,
    )
    registry = SessionRegistry(idle_seconds=settings.session_idle_seconds)

    # AsyncSqliteSaver.from_conn_string is an async context manager; hold it open for
    # the process lifetime via an exit stack rather than per request.
    async with contextlib.AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(settings.checkpoint_db)
        )
        await checkpointer.setup()
        # One shared client: connection reuse matters when a voice caller is waiting.
        http_client = await stack.enter_async_context(
            httpx.AsyncClient(timeout=settings.n8n_timeout_seconds)
        )
        STATE.update(
            graph=build_graph(retriever, llm, settings, checkpointer, http_client) if llm else None,
            retriever=retriever,
            registry=registry,
            checkpointer=checkpointer,
            ready=llm is not None,
        )
        sweeper = asyncio.create_task(_sweep_loop(registry))
        log.info(
            "ready: %d chunks from %s, embedder=%s, model=%s, n8n=%s",
            len(retriever.chunks), RENTAL_CORPUS_DIR.name, embedder.name,
            settings.chat_model, "configured" if settings.n8n_configured else "NOT configured",
        )
        try:
            yield
        finally:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper


async def _sweep_loop(registry: SessionRegistry) -> None:
    while True:
        await asyncio.sleep(60)
        with contextlib.suppress(Exception):
            await registry.sweep()


app.router.lifespan_context = lifespan


# --------------------------------------------------------------------------
# Core turn handler -- shared by the Vapi route and the text route
# --------------------------------------------------------------------------
def _remember_debug(call_id: str, payload: dict) -> None:
    LAST_TURN[call_id] = payload
    if len(LAST_TURN) > MAX_DEBUG_CALLS:
        for key in list(LAST_TURN)[: len(LAST_TURN) - MAX_DEBUG_CALLS]:
            LAST_TURN.pop(key, None)


async def run_turn(call_id: str, question: str, position: int | None = None) -> dict:
    """One caller turn through the graph. Returns the answer plus debug detail.

    `position` is the caller's own view of conversation length, used only for duplicate
    detection. Vapi supplies it implicitly as the length of the `messages` it sends.
    When it is None we fall back to our checkpoint length, which cannot detect a retry
    -- see SessionRegistry.fingerprint.
    """
    graph, registry = STATE["graph"], STATE["registry"]
    if graph is None:
        raise HTTPException(503, "Agent not ready: OPENAI_API_KEY is not configured.")

    session = await registry.get(call_id)

    # Serialise turns within a call; different calls proceed in parallel.
    async with session.lock:
        config = {"configurable": {"thread_id": session.thread_id}}
        existing = await graph.aget_state(config)
        turn_index = len((existing.values or {}).get("messages", []) or []) // 2

        fingerprint = registry.fingerprint(
            call_id, question, position if position is not None else turn_index
        )
        replayed = registry.replay(session, fingerprint)
        if replayed is not None:
            return {"answer": replayed, "duplicate": True, "call_id": call_id,
                    "thread_id": session.thread_id}

        turn_id = new_turn_id()
        started = time.perf_counter()
        result = await graph.ainvoke(
            {"question": question, "call_id": call_id, "turn_id": turn_id}, config
        )
        elapsed = time.perf_counter() - started
        answer = result.get("answer", "")
        registry.remember(session, fingerprint, answer)

        debug = {
            "call_id": call_id,
            "thread_id": session.thread_id,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "question": question,
            "resolved_question": result.get("resolved_question"),
            "route": result.get("route"),
            "route_reason": result.get("route_reason"),
            "flags": result.get("flags", []),
            "top_dense_score": round(result.get("top_dense_score", 0.0), 4),
            "evidence": result.get("evidence", []),
            "answer": answer,
            # Status only -- never the webhook URL or the shared secret.
            "maintenance": {
                k: v for k, v in (result.get("maintenance") or {}).items()
                if k in ("status", "request_id", "unit", "issue", "ticket_id", "error")
            },
            "latency_seconds": round(elapsed, 3),
            "history_messages": len(result.get("messages", [])),
            "completed_at": time.time(),
        }
        _remember_debug(call_id, debug)
        log.info(
            "turn call=%s idx=%d route=%s score=%.3f %.2fs q=%r",
            call_id, turn_index, debug["route"], debug["top_dense_score"], elapsed, question[:60],
        )
        return {"answer": answer, "duplicate": False, **debug}


# --------------------------------------------------------------------------
# Vapi custom-LLM endpoint
# --------------------------------------------------------------------------
def _extract_question(body: dict) -> str:
    """Last user message. Vapi sends `messages`; it falls back to `call.messages`."""
    messages = body.get("messages") or (body.get("call") or {}).get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            # Tolerate the content-parts form some clients send.
            if isinstance(content, list):
                text = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                ).strip()
                if text:
                    return text
    return ""


def _resolve_call_id(body: dict, request: Request) -> str:
    """Vapi's call id is the session identity. Fall back only so local curl works."""
    call = body.get("call") or {}
    for candidate in (
        call.get("id"),
        body.get("callId"),
        (body.get("metadata") or {}).get("call_id"),
        request.headers.get("x-call-id"),
    ):
        if candidate:
            return str(candidate)
    # No id: treat as a one-off turn rather than silently sharing one global thread.
    return f"anon-{uuid.uuid4().hex[:12]}"


def _sse_chunks(answer: str, model: str) -> Any:
    """OpenAI chat.completion.chunk stream, which is what Vapi consumes.

    HONEST DESCRIPTION: this is NOT token-by-token streaming. The graph produces one
    finished answer, which is emitted as a single content chunk followed by a stop
    chunk and `[DONE]`. The framing is SSE and Vapi consumes it happily, but no tokens
    arrive incrementally. Real streaming would not help here anyway -- the route
    (answer, refusal, or maintenance confirmation) is only known once the node
    finishes, so there is nothing to stream before then.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def envelope(delta: dict, finish: str | None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield envelope({"role": "assistant", "content": answer}, None)
    yield envelope({}, "stop")
    yield "data: [DONE]\n\n"


def require_secret(x_vapi_secret: str | None = Header(default=None)) -> None:
    """Shared auth for every route that spends money or returns data.

    Originally only the Vapi endpoint checked this, which left /text able to run model
    requests on the operator's OpenAI key, and /debug/* able to return resume content,
    to anyone who found the tunnel URL. The secret is optional (unset = open) so local
    development needs no config, but setting it must protect everything, not one route.
    """
    if settings.server_secret and x_vapi_secret != settings.server_secret:
        raise HTTPException(401, "Bad or missing X-Vapi-Secret")


@app.post("/chat/completions")
async def chat_completions(
    request: Request,
    x_vapi_secret: str | None = Header(default=None),
) -> Any:
    require_secret(x_vapi_secret)

    body = await request.json()
    question = _extract_question(body)
    call_id = _resolve_call_id(body, request)
    model = body.get("model") or settings.chat_model

    if not question:
        # No caller turn yet: Vapi is opening the conversation. Greet.
        answer = GREETING
    else:
        try:
            # Vapi's own view of conversation length. Identical on a retry, which is
            # what lets the duplicate check fire.
            incoming = body.get("messages") or (body.get("call") or {}).get("messages") or []
            result = await run_turn(call_id, question, position=len(incoming))
            answer = result["answer"]
        except HTTPException:
            raise
        except Exception:
            log.exception("turn failed call=%s", call_id)
            answer = "Sorry, something went wrong on my end. Could you ask that again?"

    if body.get("stream"):
        return StreamingResponse(
            _sse_chunks(answer, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


# Vapi installs vary in whether the configured URL already ends in /chat/completions.
# Accepting the alias costs nothing and removes a whole class of setup mistake.
app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])


# --------------------------------------------------------------------------
# Demo + debug routes
# --------------------------------------------------------------------------
class TextTurn(BaseModel):
    call_id: str = Field(..., description="Any stable id. Acts as the session key.")
    message: str
    position: int | None = Field(
        default=None,
        description="Caller-side conversation length. Send the same value twice to "
                    "simulate a Vapi retry and exercise duplicate detection.",
    )


@app.post("/text")
async def text_turn(turn: TextTurn, _: None = Depends(require_secret)) -> dict:
    """Same graph, no voice. What the tests and the eval harness drive.

    Behind the same secret as the Vapi route: it runs identical model requests, so
    leaving it open would let anyone who found the URL spend the operator's quota.
    """
    return await run_turn(turn.call_id, turn.message, position=turn.position)


@app.get("/health")
async def health() -> dict:
    """Public readiness. Deliberately minimal.

    This is the one unauthenticated route, because the demo page reads it before the
    viewer has supplied a secret and an uptime check needs it. So it carries only what
    that banner shows -- no session list, no tuning constants, no filesystem paths.

    It previously returned every active call id and thread id, which are session
    identifiers, plus the absolute path of the checkpoint database, which leaks the
    operator's username and directory layout. Both now live behind /debug/sessions.
    """
    retriever = STATE["retriever"]
    return {
        "ready": STATE["ready"],
        "chunks": len(retriever.chunks) if retriever else 0,
        "embedding_model": retriever.embedder.name if retriever else None,
        "embedding_dim": retriever.embedder.dim if retriever else None,
        "chat_model": settings.chat_model,
        "checkpointer": "AsyncSqliteSaver",
        "property": PROPERTY_NAME,
        "notice": PROTOTYPE_NOTICE,
        # Whether the maintenance workflow can reach n8n at all. The URL itself is not
        # published -- only whether one is set.
        "n8n_configured": settings.n8n_configured,
    }


@app.get("/debug/sessions")
async def debug_sessions(_: None = Depends(require_secret)) -> dict:
    """Live sessions and resolved tuning config. Authenticated: call ids and thread ids
    identify conversations, and the checkpoint path identifies the operator's machine."""
    registry = STATE["registry"]
    return {
        "config": {
            "hybrid_retrieval": settings.hybrid_retrieval,
            "dense_weight": settings.dense_weight,
            "min_similarity": settings.min_similarity,
            "high_confidence_similarity": settings.high_confidence_similarity,
            "top_k": settings.top_k,
            "max_history_turns": settings.max_history_turns,
            "session_idle_seconds": settings.session_idle_seconds,
            "n8n_configured": settings.n8n_configured,
            "n8n_timeout_seconds": settings.n8n_timeout_seconds,
            # Name only -- the full path would expose the operator's home directory.
            "checkpoint_db": Path(settings.checkpoint_db).name,
        },
        **(registry.stats() if registry else {}),
    }


@app.get("/debug/last")
async def debug_last(call_id: str, _: None = Depends(require_secret)) -> dict:
    if not settings.debug_panel:
        raise HTTPException(404, "Debug panel disabled")
    # call_id is required, and there is deliberately no "most recent turn" fallback and
    # no way to list all calls. An earlier version had both so the voice demo's debug
    # panel could work before the browser learned Vapi's call id -- but that let one
    # caller read another caller's question and retrieved resume passages. The page now
    # takes the id from vapi.start()'s return value instead.
    return LAST_TURN.get(call_id, {"note": f"no completed turn for call_id {call_id}"})


@app.get("/debug/chunks")
async def debug_chunks(_: None = Depends(require_secret)) -> dict:
    if not settings.debug_panel:
        raise HTTPException(404, "Debug panel disabled")
    retriever = STATE["retriever"]
    return {
        "embedding_model": retriever.embedder.name,
        "embedding_dim": retriever.embedder.dim,
        "count": len(retriever.chunks),
        "chunks": [
            {"chunk_id": c.chunk_id, "citation": c.citation, "chars": len(c.text), "text": c.text}
            for c in retriever.chunks
        ],
    }


@app.get("/debug/thread")
async def debug_thread(call_id: str, _: None = Depends(require_secret)) -> dict:
    """Read the checkpoint back for a call -- shows persistence is real."""
    if not settings.debug_panel:
        raise HTTPException(404, "Debug panel disabled")
    graph = STATE["graph"]
    if graph is None:
        raise HTTPException(503, "Agent not ready")
    snapshot = await graph.aget_state({"configurable": {"thread_id": f"vapi-call:{call_id}"}})
    values = snapshot.values or {}
    return {
        "thread_id": f"vapi-call:{call_id}",
        "messages": [
            {"role": type(m).__name__.replace("Message", "").lower(), "content": m.content}
            for m in values.get("messages", [])
        ],
        "last_route": values.get("route"),
        "checkpoint_id": getattr(snapshot.config, "get", lambda *_: None)("configurable", {}).get("checkpoint_id")
        if snapshot.config else None,
    }


@app.get("/vapi-config")
async def vapi_config(_: None = Depends(require_secret)) -> dict:
    """Public key and assistant id, so the page can pre-fill itself.

    Behind the secret even though a Vapi public key is designed for client code: the
    key plus the assistant id is enough for a stranger to start calls on the operator's
    Vapi account and spend their credits. On a public tunnel that is a real cost, so
    the convenience of pre-filling is gated the same as everything else.
    """
    return {
        "publicKey": settings.vapi_public_key,
        "assistantId": settings.vapi_assistant_id,
        "configured": bool(settings.vapi_public_key and settings.vapi_assistant_id),
    }


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """Streaming chat over a socket. Protocol and limits are documented in app/ws.py.

    Same graph, same checkpointer and the same per-connection thread isolation as the
    Vapi path -- the difference is that this one can stream tokens, which the SSE
    endpoint cannot because Vapi wants one finished answer.
    """
    from app.ws import chat_socket

    await chat_socket(websocket, STATE, settings)


@app.get("/greeting")
async def greeting() -> dict:
    return {"greeting": GREETING}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "web/index.html")


def main() -> None:
    import uvicorn

    uvicorn.run("app.server:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
