"""The /ws/chat protocol, driven through FastAPI's TestClient.

The graph is stubbed so these test the socket, not the model: streaming is simulated by
a stub that calls the token sink exactly as the real generate node does. The one place a
real component is used is the n8n client, which runs against httpx.MockTransport.
"""
from __future__ import annotations

import contextlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import server
from app.sessions import SessionRegistry
from app.ws import CLOSE_POLICY_VIOLATION, MAX_MESSAGE_BYTES

SECRET = "ws-test-secret"


class StreamingGraph:
    """Stands in for the compiled graph.

    Streams `answer` through the token sink one word at a time, which is what the real
    generate node does with the model's chunks. Records every thread it was invoked
    with, so isolation can be asserted.
    """

    def __init__(self, answer: str = "The pet fee is two fifty.", route: str = "property_question"):
        self.answer, self.route = answer, route
        self.threads: list[str] = []
        self.questions: list[str] = []
        self.final_answer: str | None = None   # set to differ from streamed text

    async def aget_state(self, config):
        class Snapshot:
            values: dict = {}
            config = None
        return Snapshot()

    async def ainvoke(self, payload, config):
        from app.graph import TOKEN_SINK

        self.threads.append(config["configurable"]["thread_id"])
        self.questions.append(payload["question"])
        sink = TOKEN_SINK.get()
        if sink is not None:
            for word in self.answer.split(" "):
                await sink(word + " ")
        return {
            "answer": self.final_answer if self.final_answer is not None else self.answer,
            "route": self.route, "evidence": [], "top_dense_score": 0.5,
            "messages": [], "maintenance": {},
        }


class _StubRetriever:
    chunks: list = []

    class embedder:
        name = "stub"
        dim = 3


@pytest.fixture
def ws_client(monkeypatch):
    """App with no lifespan and a secret configured."""
    from dataclasses import replace

    @contextlib.asynccontextmanager
    async def no_startup(_):
        yield

    monkeypatch.setattr(server.app.router, "lifespan_context", no_startup)
    monkeypatch.setattr(server, "settings", replace(server.settings, server_secret=SECRET))
    graph = StreamingGraph()
    registry = SessionRegistry()
    server.STATE.update(graph=graph, registry=registry, ready=True,
                        retriever=_StubRetriever(), checkpointer=None)
    server.LAST_TURN.clear()
    with TestClient(server.app) as client:
        client.graph = graph
        client.registry = registry
        yield client


def _collect(ws) -> tuple[str, list[dict]]:
    """Read frames until `done`, returning the reassembled text and every frame."""
    text, frames = "", []
    while True:
        frame = ws.receive_json()
        frames.append(frame)
        if frame["type"] == "delta":
            text += frame["text"]
        elif frame["type"] == "reset":
            text = ""
        elif frame["type"] == "done":
            return text, frames


# --- auth --------------------------------------------------------------------

def test_missing_auth_frame_closes_1008(ws_client):
    with pytest.raises(WebSocketDisconnect) as exc, ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "user", "text": "What is the pet fee?"})
        ws.receive_json()
    assert exc.value.code == CLOSE_POLICY_VIOLATION


def test_wrong_secret_closes_1008(ws_client):
    with pytest.raises(WebSocketDisconnect) as exc, ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": "wrong"})
        ws.receive_json()
    assert exc.value.code == CLOSE_POLICY_VIOLATION


def test_malformed_auth_frame_closes_1008(ws_client):
    with pytest.raises(WebSocketDisconnect) as exc, ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_text("not json at all")
        ws.receive_json()
    assert exc.value.code == CLOSE_POLICY_VIOLATION


def test_correct_secret_is_accepted(ws_client):
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ready = ws.receive_json()
    assert ready["type"] == "ready" and ready["call_id"].startswith("ws-")


def test_secret_is_never_accepted_from_the_url(ws_client):
    """A query string lands in history, Referer headers and proxy logs."""
    with pytest.raises(WebSocketDisconnect) as exc, \
            ws_client.websocket_connect(f"/ws/chat?secret={SECRET}") as ws:
        ws.send_json({"type": "user", "text": "hi"})
        ws.receive_json()
    assert exc.value.code == CLOSE_POLICY_VIOLATION


# --- streaming ---------------------------------------------------------------

def test_deltas_reassemble_into_the_full_answer(ws_client):
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json({"type": "user", "text": "What is the pet fee?"})
        text, frames = _collect(ws)
    assert text.strip() == "The pet fee is two fifty."
    assert sum(1 for f in frames if f["type"] == "delta") > 1, "arrived in one piece, not streamed"
    assert frames[-1]["type"] == "done" and frames[-1]["route"] == "property_question"


def test_done_carries_the_route_and_latency(ws_client):
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json({"type": "user", "text": "What is the pet fee?"})
        _, frames = _collect(ws)
    done = frames[-1]
    assert done["route"] and isinstance(done["latency_seconds"], (int, float))


def test_a_retracted_answer_sends_reset_then_the_replacement(ws_client):
    """A post-check can reject text already streamed. The client must be told."""
    ws_client.graph.final_answer = "I don't have that in the property documents."
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json({"type": "user", "text": "Are there units available?"})
        text, frames = _collect(ws)
    assert any(f["type"] == "reset" for f in frames)
    assert text == "I don't have that in the property documents."


def test_multiple_turns_reuse_one_thread(ws_client):
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        for q in ("What is the pet fee?", "Is that refundable?"):
            ws.send_json({"type": "user", "text": q})
            _collect(ws)
    assert len(set(ws_client.graph.threads)) == 1
    assert len(ws_client.graph.threads) == 2


def test_two_sockets_are_isolated(ws_client):
    with ws_client.websocket_connect("/ws/chat") as a:
        a.send_json({"type": "auth", "secret": SECRET})
        ready_a = a.receive_json()
        a.send_json({"type": "user", "text": "What is the pet fee?"})
        _collect(a)
        with ws_client.websocket_connect("/ws/chat") as b:
            b.send_json({"type": "auth", "secret": SECRET})
            ready_b = b.receive_json()
            b.send_json({"type": "user", "text": "Is that refundable?"})
            _collect(b)
    assert ready_a["call_id"] != ready_b["call_id"]
    assert len(set(ws_client.graph.threads)) == 2, "two sockets shared a LangGraph thread"


# --- housekeeping ------------------------------------------------------------

def test_disconnect_leaves_no_session_behind(ws_client):
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json({"type": "user", "text": "What is the pet fee?"})
        _collect(ws)
    assert ws_client.registry.stats()["active_sessions"] == 0


def test_disconnect_mid_stream_stops_generation_and_cleans_up(ws_client):
    """The client vanishes while tokens are in flight. The send fails, generation
    unwinds, and the session is dropped -- no lock or dedupe table left behind."""
    sent: list[str] = []

    class SlowGraph(StreamingGraph):
        async def ainvoke(self, payload, config):
            from app.graph import TOKEN_SINK
            self.threads.append(config["configurable"]["thread_id"])
            sink = TOKEN_SINK.get()
            for i in range(200):
                await sink(f"word{i} ")
                sent.append(f"word{i}")
            return {"answer": " ".join(sent), "route": "property_question",
                    "evidence": [], "messages": [], "maintenance": {}}

    ws_client.graph = SlowGraph()
    server.STATE["graph"] = ws_client.graph
    with contextlib.suppress(Exception), ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json({"type": "user", "text": "tell me everything"})
        ws.receive_json()          # one delta, then walk away
        ws.close()
    assert len(sent) < 200, "generation kept running after the client disconnected"
    assert ws_client.registry.stats()["active_sessions"] == 0


def test_oversized_frame_is_rejected(ws_client):
    with pytest.raises(WebSocketDisconnect) as exc, ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json({"type": "user", "text": "x" * (MAX_MESSAGE_BYTES + 100)})
        ws.receive_json()
    assert exc.value.code == CLOSE_POLICY_VIOLATION


def test_oversized_auth_frame_is_rejected(ws_client):
    with pytest.raises(WebSocketDisconnect) as exc, ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"type": "auth", "secret": "x" * (MAX_MESSAGE_BYTES + 100)}))
        ws.receive_json()
    assert exc.value.code == CLOSE_POLICY_VIOLATION


@pytest.mark.parametrize("frame,expected", [
    ({"type": "ping"}, "pong"),
    ({"type": "nonsense"}, "error"),
    ({"type": "user", "text": "   "}, "error"),
])
def test_control_and_bad_frames_do_not_close_the_socket(ws_client, frame, expected):
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json(frame)
        assert ws.receive_json()["type"] == expected
        # still usable
        ws.send_json({"type": "user", "text": "What is the pet fee?"})
        assert _collect(ws)[0].strip()


def test_non_json_frame_is_reported_without_closing(ws_client):
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_text("{not json")
        assert ws.receive_json()["type"] == "error"


def test_no_auth_required_when_no_secret_is_configured(monkeypatch):
    """Local development must not need configuration."""
    from dataclasses import replace

    @contextlib.asynccontextmanager
    async def no_startup(_):
        yield

    monkeypatch.setattr(server.app.router, "lifespan_context", no_startup)
    monkeypatch.setattr(server, "settings", replace(server.settings, server_secret=""))
    server.STATE.update(graph=StreamingGraph(), registry=SessionRegistry(), ready=True,
                        retriever=_StubRetriever(), checkpointer=None)
    with TestClient(server.app) as client, client.websocket_connect("/ws/chat") as ws:
        assert ws.receive_json()["type"] == "ready"


# --- maintenance over the socket ---------------------------------------------

class MaintenanceGraph:
    """Runs the REAL maintenance node logic against a mocked n8n."""

    def __init__(self, http_client, llm_replies):
        from dataclasses import replace

        from langgraph.checkpoint.memory import InMemorySaver

        from app.config import RENTAL_CORPUS_DIR
        from app.config import settings as base
        from app.embeddings import HashingEmbedder
        from app.graph import build_graph
        from app.retriever import PolicyRetriever
        from tests.conftest import ScriptedLLM

        self._graph = build_graph(
            PolicyRetriever.from_corpus(RENTAL_CORPUS_DIR, HashingEmbedder()),
            ScriptedLLM(llm_replies),
            replace(base, n8n_webhook_url="https://n8n.test/w", server_secret=SECRET),
            InMemorySaver(), http_client,
        )

    async def aget_state(self, config):
        return await self._graph.aget_state(config)

    async def ainvoke(self, payload, config):
        return await self._graph.ainvoke(payload, config)


def _n8n(seen: list, status: int = 200, timeout: bool = False) -> httpx.AsyncClient:
    """A mock-transport client. Not closed: MockTransport opens no sockets, and these
    tests are synchronous so there is no loop to await aclose() on."""
    def handler(request: httpx.Request) -> httpx.Response:
        if timeout:
            raise httpx.ReadTimeout("slow", request=request)
        body = json.loads(request.read().decode())
        seen.append(body)
        if status >= 300:
            return httpx.Response(status)
        return httpx.Response(200, json={"status": "accepted", "ticket_id": "DEMO-WS1",
                                         "request_id": body["request_id"]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _maintenance_client(monkeypatch, http_client, replies):
    from dataclasses import replace

    @contextlib.asynccontextmanager
    async def no_startup(_):
        yield

    monkeypatch.setattr(server.app.router, "lifespan_context", no_startup)
    monkeypatch.setattr(server, "settings", replace(server.settings, server_secret=SECRET))
    server.STATE.update(graph=MaintenanceGraph(http_client, replies),
                        registry=SessionRegistry(), ready=True,
                        retriever=_StubRetriever(), checkpointer=None)
    return TestClient(server.app)


def test_maintenance_submit_over_the_socket_creates_exactly_one_ticket(monkeypatch):
    seen: list = []
    client = _maintenance_client(monkeypatch, _n8n(seen), [
        "MAINTENANCE_REQUEST", "UNIT: 2B\nISSUE: leaking kitchen faucet",
    ])
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json({"type": "user", "text": "my kitchen faucet is leaking in unit 2B"})
        _, frames = _collect(ws)
        assert not any(f["type"] == "ticket" for f in frames), "submitted before confirmation"

        ws.send_json({"type": "user", "text": "yes"})
        _, frames = _collect(ws)
        ticket = next(f for f in frames if f["type"] == "ticket")
        assert ticket["status"] == "submitted" and ticket["ticket_id"] == "DEMO-WS1"

        ws.send_json({"type": "user", "text": "yes"})      # duplicate confirmation
        _collect(ws)
    assert len(seen) == 1, "a repeated yes created a second ticket"


def test_n8n_timeout_pushes_failed_never_submitted(monkeypatch):
    seen: list = []
    client = _maintenance_client(monkeypatch, _n8n(seen, timeout=True), [
        "MAINTENANCE_REQUEST", "UNIT: 2B\nISSUE: leaking kitchen faucet",
    ])
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json({"type": "user", "text": "my kitchen faucet is leaking in unit 2B"})
        _collect(ws)
        ws.send_json({"type": "user", "text": "yes"})
        text, frames = _collect(ws)
    ticket = next(f for f in frames if f["type"] == "ticket")
    assert ticket["status"] == "failed"
    assert ticket["ticket_id"] is None
    assert "couldn't confirm" in text.lower()
    assert "DEMO" not in text


def test_n8n_error_response_pushes_failed(monkeypatch):
    seen: list = []
    client = _maintenance_client(monkeypatch, _n8n(seen, status=500), [
        "MAINTENANCE_REQUEST", "UNIT: 2B\nISSUE: leaking kitchen faucet",
    ])
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "secret": SECRET})
        ws.receive_json()
        ws.send_json({"type": "user", "text": "my kitchen faucet is leaking in unit 2B"})
        _collect(ws)
        ws.send_json({"type": "user", "text": "yes"})
        _, frames = _collect(ws)
    assert next(f for f in frames if f["type"] == "ticket")["status"] == "failed"
