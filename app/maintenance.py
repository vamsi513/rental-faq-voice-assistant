"""Maintenance requests: collect, confirm, submit to n8n.

Deliberately boring and deterministic. Submitting is the only thing this prototype does
that leaves the process, so the decision to submit is made by explicit pattern matching
on the caller's words, not by a model's judgement.

The flow, as a state machine on `MaintenanceRequest.status`:

    idle -> collecting           caller reported a problem, something is still missing
    collecting -> awaiting       unit and issue known; the summary has been read back
    awaiting -> submitted        caller explicitly confirmed AND n8n accepted
    awaiting -> failed           caller confirmed but n8n timed out or refused
    awaiting -> collecting       caller corrected a detail; re-confirm before sending
    awaiting -> cancelled        caller declined

Honest limits, all documented in the README:
  * A timeout is NOT retried. n8n may already have created the ticket, so a retry could
    duplicate it. The caller is told it could not be confirmed.
  * `request_id` is stable for one pending request, so repeated confirmations and
    duplicate Vapi deliveries submit once. It is per-session state in SQLite, so two
    workers would each keep their own.
  * "Accepted" means the demo workflow returned 2xx with a ticket id. Nobody is
    dispatched. No person is notified.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import httpx

log = logging.getLogger("agent.maintenance")

Status = Literal["idle", "collecting", "awaiting_confirmation", "submitted", "failed", "cancelled"]

# Submit only on an unmistakable yes. Anything ambiguous re-asks rather than guessing.
_CONFIRM = re.compile(
    r"^(yes|yeah|yep|yup|sure|ok|okay|correct|right|confirm(ed)?|do it|go ahead|"
    r"please do|submit( it)?|send( it)?|that'?s right|sounds good|affirmative)\b"
    r"|^(yes|yeah|sure|ok|okay)[\s,.!]*(please|go ahead|submit|send|do it)\b"
    r"|\b(submit|send) (it|the request|that)\b"
    r"|\b(go ahead and (submit|send))\b",
    re.I,
)
_DECLINE = re.compile(
    # A leading hedge is common in speech: "actually no", "um, no", "well no".
    r"^(actually|um+|uh+|well|hmm+|sorry|wait)[\s,]*"
    r"(no|nope|nah|cancel|stop|never ?mind|forget it|don'?t|do not|hold off)\b"
    r"|^(no|nope|nah|cancel|stop|never ?mind|forget it|don'?t|do not|hold off)\b"
    r"|\b(cancel|scrap) (it|that|the request)\b"
    r"|\bnot yet\b",
    re.I,
)

# Anything here means: stop the routine workflow and read the emergency instructions.
# Keyword matching is crude and will miss paraphrases -- it is a floor, not a ceiling.
_EMERGENCY = re.compile(
    r"\b(fire|smoke|burning|flames?)\b"
    r"|\bgas\s*(leak|smell|odou?r)\b|\bsmell(s|ing)?\s+gas\b"
    r"|\b(flood(ing|ed)?|water (is )?pouring|gushing|sewage|sewer backup)\b"
    # "burst pipe" and "a pipe burst" are the same emergency said two ways.
    r"|\b(burst|broken|cracked)\s+pipe\b|\bpipe\s+(burst|broke|is leaking badly)\b"
    r"|\bno (heat|power|electricity)\b|\bpower('s| is)? out\b"
    r"|\b(carbon monoxide|co alarm)\b"
    r"|\b(break[- ]?in|broken (door|window) lock|can'?t lock)\b"
    r"|\b(electrical|sparking|sparks|exposed wire)\b"
    r"|\b(emergency|urgent|right now|immediately)\b.{0,30}\b(leak|water|fire|gas|heat)\b",
    re.I,
)

# The documented emergency instruction. Verbatim from data/rental/08-maintenance.md, so
# the agent cannot improvise safety advice.
EMERGENCY_REPLY = (
    "That sounds like an emergency, so please don't use the routine request. "
    "If there's a fire, a gas smell, or anyone is in danger, call 911 first. "
    "Then call the 24-hour emergency line at 555-0199 and press 1. "
    "They handle flooding, active leaks, no heat, total power loss, broken exterior "
    "locks and sewage backups at any hour."
)


# A caller amending a pending request rather than answering yes or no.
_CORRECTION = re.compile(
    r"^(actually|wait|sorry|no wait|hold on|scratch that|correction)\b"
    r"|\b(make it|change it to|it'?s actually|should be|i meant)\b"
    r"|^(unit|apartment|apt)\s*\w+"
    r"|\bit'?s (unit|apartment|apt)\b",
    re.I,
)


def is_correction(text: str) -> bool:
    """True when the turn amends a pending request instead of answering it."""
    return bool(_CORRECTION.search(" ".join((text or "").split())))


def answers_pending_confirmation(text: str) -> bool:
    """Is this turn a reply to "should I submit this?" at all?

    Without this check a pending request captured EVERY later turn, so a caller who
    asked about office hours while a request was waiting got the maintenance node
    instead of an answer. A pending request should wait patiently, not interrupt.
    """
    return is_confirmation(text) or is_decline(text) or is_correction(text)


def is_emergency(text: str) -> bool:
    return bool(_EMERGENCY.search(text or ""))


def is_confirmation(text: str) -> bool:
    cleaned = " ".join((text or "").lower().split())
    return bool(_CONFIRM.search(cleaned)) and not _DECLINE.search(cleaned)


def is_decline(text: str) -> bool:
    return bool(_DECLINE.search(" ".join((text or "").lower().split())))


@dataclass
class MaintenanceRequest:
    """One pending request. Lives in graph state, so it persists across turns."""

    status: Status = "idle"
    request_id: str = ""
    unit: str | None = None
    issue: str | None = None
    ticket_id: str | None = None
    error: str | None = None
    # request_ids already sent, so a repeated confirmation cannot submit twice.
    submitted_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_state(cls, raw: dict | None) -> MaintenanceRequest:
        if not raw:
            return cls()
        known = {f for f in cls().__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_state(self) -> dict:
        return asdict(self)

    def start(self) -> None:
        """Begin a new request with a fresh id. The id is what makes submission
        idempotent, so it is minted once here and not regenerated on re-confirmation."""
        self.request_id = f"req-{uuid.uuid4().hex[:12]}"
        self.status = "collecting"
        self.unit = self.issue = self.ticket_id = self.error = None

    @property
    def ready(self) -> bool:
        return bool(self.unit and self.issue)

    @property
    def already_submitted(self) -> bool:
        return bool(self.request_id) and self.request_id in self.submitted_ids

    def summary(self) -> str:
        return (
            f"I have a demo maintenance request for unit {self.unit}: {self.issue}. "
            "Should I submit this demo maintenance request?"
        )


@dataclass(frozen=True)
class SubmitResult:
    ok: bool
    ticket_id: str | None = None
    error: str | None = None


async def submit_to_n8n(
    request: MaintenanceRequest,
    call_id: str,
    *,
    webhook_url: str,
    secret: str,
    header_name: str,
    timeout: float,
    client: httpx.AsyncClient | None = None,
) -> SubmitResult:
    """POST the request to the n8n webhook and validate what comes back.

    Success is narrow on purpose: 2xx, parseable JSON, status "accepted", a ticket id,
    and a request_id echoing the one sent. Anything else is reported to the caller as
    "couldn't confirm" rather than as success, because the whole point of this call is
    to avoid telling someone their request went through when it may not have.
    """
    if not webhook_url:
        return SubmitResult(False, error="n8n webhook URL is not configured")

    payload = {
        "request_id": request.request_id,
        "call_id": call_id,
        "unit": request.unit,
        "issue": request.issue,
        "source": "rental-voice-prototype",
    }
    headers = {"Content-Type": "application/json"}
    if secret:
        headers[header_name] = secret

    owned = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await client.post(webhook_url, json=payload, headers=headers, timeout=timeout)
    except httpx.TimeoutException:
        # Not retried: n8n may have processed it already.
        log.warning("n8n timeout request_id=%s", request.request_id)
        return SubmitResult(False, error="timeout")
    except httpx.HTTPError as exc:
        log.warning("n8n transport error request_id=%s: %s", request.request_id, exc)
        return SubmitResult(False, error="unreachable")
    finally:
        if owned:
            await client.aclose()

    if response.status_code >= 300:
        log.warning("n8n returned %s for request_id=%s", response.status_code, request.request_id)
        return SubmitResult(False, error=f"http {response.status_code}")

    try:
        body: Any = response.json()
    except ValueError:
        return SubmitResult(False, error="response was not JSON")
    if not isinstance(body, dict):
        return SubmitResult(False, error="response was not a JSON object")

    ticket_id = body.get("ticket_id")
    if body.get("status") != "accepted" or not ticket_id:
        return SubmitResult(False, error="response missing status=accepted or ticket_id")
    # Mismatched echo means the reply belongs to a different request; do not claim it.
    echoed = body.get("request_id")
    if echoed and echoed != request.request_id:
        return SubmitResult(False, error="response request_id did not match")

    return SubmitResult(True, ticket_id=str(ticket_id))
