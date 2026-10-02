"""The Vapi custom-LLM contract, exercised against the real app.

Vapi's "Custom LLM" provider speaks OpenAI chat-completions. These tests pin the parts
of that contract a working voice integration depends on: where the caller's turn is
read from, where the session identity comes from, and the exact SSE framing Vapi
consumes. A webhook that only logged events would pass none of this.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import server
from app.config import GREETING


class StubGraph:
    """Stands in for the compiled graph so these tests cover the HTTP layer only."""

    def __init__(self):
        self.seen: list[tuple[str, str]] = []

    async def aget_state(self, config):
        class Snapshot:
            values: dict = {}
            config = None
        return Snapshot()

    async def ainvoke(self, payload, config):
        self.seen.append((config["configurable"]["thread_id"], payload["question"]))
        return {"answer": f"answered: {payload['question']}", "route": "property_question",
                "evidence": [], "top_dense_score": 0.5, "messages": [], "maintenance": {}}


@pytest.fixture
def client(monkeypatch):
    """Replaces the real lifespan: these tests cover the HTTP layer, so booting the
    embedder, the SQLite checkpointer and a live model would only add minutes and
    network flakiness."""
    import contextlib

    from app.sessions import SessionRegistry

    @contextlib.asynccontextmanager
    async def no_startup(_):
        yield

    monkeypatch.setattr(server.app.router, "lifespan_context", no_startup)
    # Force the secret off by default. Otherwise these tests pass or fail depending on
    # whether the developer's local .env happens to set SERVER_SECRET -- which is
    # exactly how they broke once a real Vapi tunnel was configured.
    from dataclasses import replace

    monkeypatch.setattr(server, "settings", replace(server.settings, server_secret=""))
    graph = StubGraph()
    server.STATE.update(graph=graph, registry=SessionRegistry(), ready=True,
                        retriever=_StubRetriever(), checkpointer=None)
    server.LAST_TURN.clear()
    with TestClient(server.app) as c:
        c.graph = graph
        yield c


class _StubRetriever:
    chunks: list = []

    class embedder:
        name = "stub"
        dim = 3


def vapi_body(**overrides):
    body = {
        "model": "gpt-4o-mini",
        "stream": True,
        "max_tokens": 200,
        "temperature": 0.2,
        "call": {"id": "vapi-call-1"},
        "messages": [
            {"role": "system", "content": "(vapi prompt)"},
            {"role": "user", "content": "What is the pet fee?"},
        ],
    }
    body.update(overrides)
    return body


# --- the protocol ----------------------------------------------------------

def test_streaming_response_is_openai_sse_terminated_with_done(client):
    response = client.post("/chat/completions", json=vapi_body())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"

    first = json.loads(lines[0][6:])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["content"].startswith("answered:")
    last_chunk = json.loads(lines[-2][6:])
    assert last_chunk["choices"][0]["finish_reason"] == "stop"


def test_non_streaming_returns_a_chat_completion(client):
    response = client.post("/chat/completions", json=vapi_body(stream=False))
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_call_id_becomes_the_thread_id(client):
    client.post("/chat/completions", json=vapi_body(stream=False))
    assert client.graph.seen[0][0] == "vapi-call:vapi-call-1"


def test_separate_calls_get_separate_threads(client):
    client.post("/chat/completions", json=vapi_body(stream=False))
    client.post("/chat/completions", json=vapi_body(
        stream=False, call={"id": "vapi-call-2"},
        messages=[{"role": "user", "content": "What are the office hours?"}]))
    threads = {thread for thread, _ in client.graph.seen}
    assert threads == {"vapi-call:vapi-call-1", "vapi-call:vapi-call-2"}


def test_last_user_message_is_the_turn(client):
    """Vapi sends the whole transcript; only the newest caller turn is the question."""
    client.post("/chat/completions", json=vapi_body(stream=False, messages=[
        {"role": "user", "content": "older question"},
        {"role": "assistant", "content": "older answer"},
        {"role": "user", "content": "the newest question"},
    ]))
    assert client.graph.seen[-1][1] == "the newest question"


def test_falls_back_to_call_messages(client):
    """Vapi omits the top-level messages array in some configurations."""
    body = vapi_body(stream=False)
    body["call"]["messages"] = body.pop("messages")
    client.post("/chat/completions", json=body)
    assert client.graph.seen[-1][1] == "What is the pet fee?"


def test_content_parts_form_is_tolerated(client):
    client.post("/chat/completions", json=vapi_body(stream=False, messages=[
        {"role": "user", "content": [{"type": "text", "text": "parts form question"}]},
    ]))
    assert client.graph.seen[-1][1] == "parts form question"


def test_no_user_turn_yields_the_greeting(client):
    """Vapi opening the conversation. The greeting must not go through the graph."""
    response = client.post("/chat/completions", json=vapi_body(
        stream=False, messages=[{"role": "system", "content": "(vapi prompt)"}]))
    assert response.json()["choices"][0]["message"]["content"] == GREETING
    assert client.graph.seen == []


def test_missing_call_id_does_not_share_a_global_thread(client):
    body = vapi_body(stream=False)
    body.pop("call")
    client.post("/chat/completions", json=body)
    client.post("/chat/completions", json=body)
    threads = {thread for thread, _ in client.graph.seen}
    assert len(threads) == 2, "anonymous turns must not be merged into one conversation"


def test_duplicate_request_is_not_appended_twice(client):
    body = vapi_body(stream=False)
    client.post("/chat/completions", json=body)
    client.post("/chat/completions", json=body)   # identical retry
    assert len(client.graph.seen) == 1


def test_v1_alias_is_accepted(client):
    assert client.post("/v1/chat/completions", json=vapi_body(stream=False)).status_code == 200


def test_shared_secret_is_enforced_when_configured(client, monkeypatch):
    from dataclasses import replace

    # Settings is frozen on purpose, so swap the whole object the handler reads.
    monkeypatch.setattr(server, "settings", replace(server.settings, server_secret="s3cret"))
    assert client.post("/chat/completions", json=vapi_body(stream=False)).status_code == 401
    ok = client.post("/chat/completions", json=vapi_body(stream=False),
                     headers={"X-Vapi-Secret": "s3cret"})
    assert ok.status_code == 200


def test_graph_failure_still_speaks_to_the_caller(client):
    """A 500 would leave the caller in silence; Vapi needs speakable text."""
    class Broken(StubGraph):
        async def ainvoke(self, payload, config):
            raise RuntimeError("boom")

    server.STATE["graph"] = Broken()
    response = client.post("/chat/completions", json=vapi_body(stream=False))
    assert response.status_code == 200
    assert "went wrong" in response.json()["choices"][0]["message"]["content"]


# --- debug surface ---------------------------------------------------------

def test_debug_last_exposes_evidence_per_call(client):
    client.post("/chat/completions", json=vapi_body(stream=False))
    debug = client.get("/debug/last", params={"call_id": "vapi-call-1"}).json()
    assert debug["thread_id"] == "vapi-call:vapi-call-1"
    assert debug["route"] == "property_question"


def test_health_reports_the_live_configuration(client):
    health = client.get("/health").json()
    assert health["ready"] is True
    assert health["checkpointer"] == "AsyncSqliteSaver"


# --- the shared secret must cover every costly or revealing route ---------------
# Found in review: only /chat/completions checked it, so anyone who found the public
# tunnel could run model requests on the operator's OpenAI key via /text, read resume
# content via /debug/*, and lift the Vapi key pair via /vapi-config.

@pytest.fixture
def secured(client, monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(server, "settings", replace(server.settings, server_secret="s3cret"))
    return client


@pytest.mark.parametrize("method,path,payload", [
    ("post", "/chat/completions", {"model": "m", "stream": False,
                                   "messages": [{"role": "user", "content": "hi"}]}),
    ("post", "/text", {"call_id": "c", "message": "hi"}),
    ("get", "/debug/last?call_id=c", None),
    ("get", "/debug/chunks", None),
    ("get", "/debug/thread?call_id=c", None),
    ("get", "/vapi-config", None),
])
def test_every_sensitive_route_requires_the_secret(secured, method, path, payload):
    call = getattr(secured, method)
    response = call(path, json=payload) if payload is not None else call(path)
    assert response.status_code == 401, f"{method.upper()} {path} is unprotected"


@pytest.mark.parametrize("method,path,payload", [
    ("post", "/text", {"call_id": "c", "message": "hi"}),
    ("get", "/debug/chunks", None),
    ("get", "/vapi-config", None),
])
def test_sensitive_routes_work_with_the_secret(secured, method, path, payload):
    call = getattr(secured, method)
    headers = {"X-Vapi-Secret": "s3cret"}
    response = call(path, json=payload, headers=headers) if payload is not None \
        else call(path, headers=headers)
    assert response.status_code == 200


def test_text_endpoint_is_open_when_no_secret_is_configured(client):
    """Local development must not need configuration."""
    assert client.post("/text", json={"call_id": "c", "message": "hi"}).status_code == 200


# --- debug must not cross sessions ---------------------------------------------
# Found in review: the page fell back to "the most recent turn across all calls", which
# could show one caller another caller's question and retrieved passages.

def test_debug_last_requires_a_call_id(client):
    assert client.get("/debug/last").status_code == 422


def test_debug_last_cannot_read_another_call(client):
    client.post("/chat/completions", json=vapi_body(stream=False))   # call vapi-call-1
    other = client.get("/debug/last", params={"call_id": "someone-elses-call"}).json()
    assert "note" in other, "a caller must not see another call's turn"
    assert "question" not in other


def test_there_is_no_endpoint_listing_all_calls(client):
    """The all-calls listing and the 'latest' fallback were both removed."""
    client.post("/chat/completions", json=vapi_body(stream=False))
    body = client.get("/debug/last", params={"call_id": "x", "latest": "1"}).json()
    assert "calls" not in body
    assert "question" not in body


def test_debug_payload_carries_raw_text_for_the_client_to_escape(client):
    """The server returns the caller's turn verbatim -- it does not sanitise it.

    That is deliberate: escaping belongs at the render site, and the page writes every
    one of these values with textContent. This test documents the contract so nobody
    later assumes the server scrubbed it. The matching client-side guarantee was
    verified in a browser: the payload below renders as literal text, injecting no
    nodes and running no script.
    """
    payload = '<img src=x onerror="alert(1)">'
    client.post("/chat/completions", json=vapi_body(stream=False, messages=[
        {"role": "user", "content": payload}]))
    debug = client.get("/debug/last", params={"call_id": "vapi-call-1"}).json()
    assert debug["question"] == payload


# --- /health must not leak session or host information --------------------------
# Found in review: /health is the one unauthenticated route, and it was returning every
# active call id and thread id (session identifiers) plus the absolute path of the
# checkpoint database (the operator's username and directory layout).

def test_health_is_public(secured):
    """Still reachable without a key -- the demo page and uptime checks need it."""
    assert secured.get("/health").status_code == 200


@pytest.mark.parametrize("leaky_field", [
    "sessions", "active_sessions", "checkpoint_db",
    "min_similarity", "dense_weight", "top_k", "max_history_turns",
])
def test_health_does_not_expose_sessions_or_internals(client, leaky_field):
    client.post("/chat/completions", json=vapi_body(stream=False))
    assert leaky_field not in client.get("/health").json()


def test_health_still_reports_what_the_page_needs(client):
    body = client.get("/health").json()
    for field in ("ready", "chunks", "embedding_model", "embedding_dim", "chat_model"):
        assert field in body


def test_health_body_contains_no_filesystem_path(client):
    assert "/" not in str(client.get("/health").json().get("checkpointer", ""))
    assert not any(
        isinstance(v, str) and v.startswith("/") for v in client.get("/health").json().values()
    )


def test_session_detail_moved_behind_the_secret(secured):
    assert secured.get("/debug/sessions").status_code == 401
    body = secured.get("/debug/sessions", headers={"X-Vapi-Secret": "s3cret"}).json()
    assert "sessions" in body and "config" in body
    # Even here the checkpoint path is reduced to a filename.
    assert "/" not in body["config"]["checkpoint_db"]


# --- the prototype must not leak the n8n credential or URL ---------------------

def test_health_does_not_publish_the_webhook_url(client):
    body = client.get("/health").json()
    # Whether a workflow is wired up is useful; where it lives is not public.
    assert "n8n_configured" in body
    assert not any("webhook" in str(v).lower() or "http" in str(v) for v in body.values())


def test_debug_payload_exposes_maintenance_status_but_no_secret(client):
    client.post("/chat/completions", json=vapi_body(stream=False))
    debug = client.get("/debug/last", params={"call_id": "vapi-call-1"}).json()
    assert "maintenance" in debug
    blob = str(debug).lower()
    assert "secret" not in blob and "webhook" not in blob


def test_prototype_notice_is_served(client):
    body = client.get("/health").json()
    assert "prototype" in body["notice"].lower()
    assert body["property"] == "Maple Court Apartments"
