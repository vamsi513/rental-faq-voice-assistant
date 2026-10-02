"""The maintenance request state machine and the n8n client.

All MOCKED -- no network. The live FastAPI-to-n8n check is a separate, manual step
documented in the README; nothing in this file proves n8n works.
"""
from __future__ import annotations

import httpx
import pytest

from app.maintenance import (
    EMERGENCY_REPLY,
    MaintenanceRequest,
    is_confirmation,
    is_decline,
    is_emergency,
    submit_to_n8n,
)

pytestmark = pytest.mark.asyncio


# --- the deterministic decision to submit ------------------------------------

@pytest.mark.parametrize("text", [
    "yes", "yes please", "yep", "sure", "ok", "confirm", "go ahead", "do it",
    "submit it", "send it", "yes please submit it", "that's right", "sounds good",
    "go ahead and submit",
])
async def test_explicit_confirmations_are_recognised(text):
    assert is_confirmation(text)


@pytest.mark.parametrize("text", [
    "no", "nope", "cancel", "cancel that", "never mind", "don't", "stop",
    "actually no", "no not yet", "forget it",
])
async def test_declines_are_not_treated_as_confirmation(text):
    assert not is_confirmation(text)
    assert is_decline(text)


@pytest.mark.parametrize("text", [
    "it's unit 2B", "the kitchen faucet", "what is the pet fee",
    "maybe", "I think so", "can you repeat that", "hold on",
])
async def test_ambiguous_turns_do_not_confirm(text):
    """Anything short of an unmistakable yes must NOT submit. A maybe is not a yes."""
    assert not is_confirmation(text)


@pytest.mark.parametrize("text", [
    "my kitchen is flooding", "I smell gas", "there's a fire",
    "the power is out in the whole unit", "sewage is backing up",
    "there's no heat", "a pipe burst", "I think there's an electrical problem",
])
async def test_emergencies_are_detected(text):
    assert is_emergency(text)


@pytest.mark.parametrize("text", [
    "a dripping faucet", "the cabinet handle is loose", "my dishwasher won't drain",
    "the bathroom fan is rattling", "a light bulb is out",
])
async def test_routine_problems_are_not_emergencies(text):
    assert not is_emergency(text)


async def test_emergency_reply_matches_the_documented_instructions():
    """The agent must not improvise safety advice: 911 first, then the documented line."""
    assert "911" in EMERGENCY_REPLY
    assert "555-0199" in EMERGENCY_REPLY


# --- the n8n client ----------------------------------------------------------

def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ready() -> MaintenanceRequest:
    r = MaintenanceRequest()
    r.start()
    r.unit, r.issue = "2B", "leaking kitchen faucet"
    return r


async def test_successful_submission_returns_the_ticket():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        seen["secret"] = request.headers.get("X-Prototype-Secret")
        import json
        body = json.loads(seen["body"])
        return httpx.Response(200, json={
            "status": "accepted", "ticket_id": "DEMO-123", "request_id": body["request_id"]})

    request = _ready()
    async with _client(handler) as c:
        result = await submit_to_n8n(
            request, "call-1", webhook_url="https://n8n.test/webhook/maintenance",
            secret="s3cret", header_name="X-Prototype-Secret", timeout=5, client=c)
    assert result.ok and result.ticket_id == "DEMO-123"
    # The payload shape the n8n workflow validates.
    import json
    body = json.loads(seen["body"])
    assert set(body) == {"request_id", "call_id", "unit", "issue", "source"}
    assert body["unit"] == "2B" and body["source"] == "rental-voice-prototype"
    assert seen["secret"] == "s3cret"


async def test_timeout_is_reported_as_unconfirmed_not_success():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    async with _client(handler) as c:
        result = await submit_to_n8n(
            _ready(), "call-1", webhook_url="https://n8n.test/w", secret="",
            header_name="X", timeout=1, client=c)
    assert not result.ok and result.error == "timeout"
    assert result.ticket_id is None


@pytest.mark.parametrize("status", [400, 401, 404, 500, 502])
async def test_non_2xx_is_a_failure(status):
    async with _client(lambda r: httpx.Response(status, text="nope")) as c:
        result = await submit_to_n8n(
            _ready(), "call-1", webhook_url="https://n8n.test/w", secret="",
            header_name="X", timeout=5, client=c)
    assert not result.ok and str(status) in (result.error or "")


async def test_non_json_response_is_a_failure():
    async with _client(lambda r: httpx.Response(200, text="<html>hi</html>")) as c:
        result = await submit_to_n8n(
            _ready(), "call-1", webhook_url="https://n8n.test/w", secret="",
            header_name="X", timeout=5, client=c)
    assert not result.ok and "JSON" in (result.error or "")


@pytest.mark.parametrize("body", [
    {},                                                  # nothing
    {"status": "accepted"},                              # no ticket
    {"ticket_id": "DEMO-1"},                             # no status
    {"status": "rejected", "ticket_id": "DEMO-1"},       # refused
    [1, 2, 3],                                           # not an object
])
async def test_malformed_success_bodies_are_failures(body):
    """A 200 is not enough. Announcing success on a body we did not understand is the
    exact failure this is here to prevent."""
    async with _client(lambda r: httpx.Response(200, json=body)) as c:
        result = await submit_to_n8n(
            _ready(), "call-1", webhook_url="https://n8n.test/w", secret="",
            header_name="X", timeout=5, client=c)
    assert not result.ok


async def test_mismatched_request_id_is_rejected():
    """A reply echoing a different request belongs to someone else's submission."""
    async with _client(lambda r: httpx.Response(200, json={
        "status": "accepted", "ticket_id": "DEMO-9", "request_id": "req-somebody-else"})) as c:
        result = await submit_to_n8n(
            _ready(), "call-1", webhook_url="https://n8n.test/w", secret="",
            header_name="X", timeout=5, client=c)
    assert not result.ok and "did not match" in (result.error or "")


async def test_unconfigured_webhook_fails_closed():
    result = await submit_to_n8n(
        _ready(), "call-1", webhook_url="", secret="", header_name="X", timeout=5)
    assert not result.ok and "not configured" in (result.error or "")


async def test_secret_header_is_omitted_when_unset():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["has_header"] = "X-Prototype-Secret" in request.headers
        return httpx.Response(200, json={"status": "accepted", "ticket_id": "T"})

    async with _client(handler) as c:
        await submit_to_n8n(_ready(), "c", webhook_url="https://n8n.test/w", secret="",
                            header_name="X-Prototype-Secret", timeout=5, client=c)
    assert seen["has_header"] is False


# --- request identity --------------------------------------------------------

async def test_request_id_is_stable_across_a_pending_request():
    r = MaintenanceRequest()
    r.start()
    first = r.request_id
    r.unit = "2B"
    r.issue = "updated issue"
    assert r.request_id == first, "correcting a detail must not mint a new id"


async def test_starting_a_new_request_clears_the_previous_one():
    r = MaintenanceRequest()
    r.start()
    r.unit, r.issue, r.ticket_id = "2B", "old", "DEMO-1"
    r.start()
    assert r.unit is None and r.issue is None and r.ticket_id is None


async def test_already_submitted_tracks_by_request_id():
    r = MaintenanceRequest()
    r.start()
    assert not r.already_submitted
    r.submitted_ids = [r.request_id]
    assert r.already_submitted


async def test_state_round_trips_through_the_checkpoint_shape():
    r = _ready()
    r.status = "awaiting_confirmation"
    restored = MaintenanceRequest.from_state(r.to_state())
    assert restored == r


async def test_from_state_ignores_unknown_keys():
    """Checkpoints outlive code changes; an old row must not crash the graph."""
    restored = MaintenanceRequest.from_state({"status": "collecting", "legacy_field": 1})
    assert restored.status == "collecting"
