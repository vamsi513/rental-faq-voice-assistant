"""A deterministic stand-in for the claims field extractor.

Lives in `app/` for the same reason `HashingEmbedder` does: it is an offline double the
tests and the offline evaluation need, and keeping it beside the thing it doubles makes
the pairing obvious. It is never used in a serving path, which only ever builds the real
extractor from the configured chat model.

WHAT THIS IS, PRECISELY: keyword and regex extraction, not a model. It reads the same
prompt the real extractor is sent and returns the same six lines, so the flow under test
is the real flow and only the extractor differs. It is weaker than a model at exactly the
place a model is better, free-form phrasing, so a scenario that passes offline is
evidence about the flow and not about any model's extraction.
"""
from __future__ import annotations

import re

from app.claims import LOSS_TYPES

_TURN = re.compile(r"Caller's latest turn:\s*(.*)\s*$", re.S)
_COLLECTED = re.compile(r"^(POLICY|DATE_OF_LOSS|LOSS_TYPE|DESCRIPTION|INJURED):\s*(.*)$", re.M)

# A policy number as spoken, or something the caller offered as one. Deliberately
# liberal: a malformed number must reach the validator and be refused there, not be
# silently dropped here.
_POLICY = re.compile(
    r"\b(PW[-\s]?\d[\d\s-]{6,}\d)\b"
    r"|policy\s*(?:number|no\.?|#)?\s*(?:is\s*)?([A-Z]{0,3}[-\s]?\d[\d\s-]{2,}\d)",
    re.I,
)

_DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})\b"
    r"|\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
    r"|\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)\b"
    r"|\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:,?\s*\d{4})?)\b"
    r"|\b(yesterday|last night|this morning|today)\b"
    r"|\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)\b",
    re.I,
)

_LOSS_WORDS: dict[str, tuple[str, ...]] = {
    "water": ("water", "flood", "leak", "burst pipe", "pipe burst", "soaked", "damp"),
    "fire": ("fire", "smoke", "burn", "scorch"),
    "theft": ("theft", "stolen", "stole", "burglar", "burgl", "broke in", "broken into", "robbed"),
    "wind": ("wind", "storm", "gale", "blew"),
    "hail": ("hail",),
    "vehicle": ("vehicle", "car hit", "van hit", "truck", "drove into"),
}

_LOSS_ANY = re.compile(
    r"\b(claim|loss|damage|damaged|stolen|theft|burgl\w+|flood\w*|leak\w*|fire|smoke"
    r"|hail|storm|wind|vandal\w+|accident|broke in|broken into)\b",
    re.I,
)

_YES = re.compile(r"\b(yes|yeah|yep|someone was|somebody was|she was|he was|they were)\b", re.I)
_NO_INJURY = re.compile(
    r"\b(no one|nobody|no ?body|nobody was|no injuries|not injured|everyone'?s fine"
    r"|everybody'?s fine|we'?re all fine|all fine|unhurt)\b"
    r"|^\s*no\b",
    re.I,
)

#: Turns that are plainly not about a loss at all.
_OFF_TOPIC = re.compile(
    r"\b(capital of|weather|who won|tell me a joke|what time is it|recipe"
    r"|change my policy|cancel my policy|pay my premium|update my address)\b",
    re.I,
)


def _already(prompt: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in _COLLECTED.finditer(prompt)}


def _latest_turn(prompt: str) -> str:
    match = _TURN.search(prompt)
    return (match.group(1) if match else "").strip()


def _policy(turn: str) -> str | None:
    match = _POLICY.search(turn)
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip() or None


def _date(turn: str) -> str | None:
    match = _DATE.search(turn)
    if not match:
        return None
    return next((g for g in match.groups() if g), None)


def _loss_type(turn: str) -> str | None:
    lowered = turn.lower()
    for name, words in _LOSS_WORDS.items():
        if any(word in lowered for word in words):
            return name
    return None


def _injured(turn: str, asking: bool) -> str | None:
    """Only read an injury answer when there is one to read.

    A bare "yes" is a confirmation in most of this flow, so it is read as an injury
    answer only on the turn that answers "was anyone injured?".
    """
    if _NO_INJURY.search(turn):
        return "no"
    if asking and _YES.search(turn):
        return "yes"
    return None


def extract(system: str, user: str) -> str:
    """Return the six lines the real extractor returns. Signature matches it exactly."""
    turn = _latest_turn(user)
    collected = _already(user)
    asking_injury = all(
        collected.get(name, "NONE") != "NONE"
        for name in ("POLICY", "DATE_OF_LOSS", "LOSS_TYPE", "DESCRIPTION")
    ) and collected.get("INJURED", "NONE") == "NONE"

    policy = _policy(turn)
    date = _date(turn)
    loss_type = _loss_type(turn)
    injured = _injured(turn, asking_injury)

    # A description is the caller's own words, kept only when the turn actually describes
    # something. A bare field answer is not a description. A turn whose only loss word is
    # in the loss-type vocabulary ("a pipe burst") still counts, which an earlier version
    # of this condition missed because it checked only the generic loss words.
    description = None
    if (loss_type or _LOSS_ANY.search(turn)) and len(turn.split()) >= 4:
        description = " ".join(turn.split())[:160]

    off_topic = bool(_OFF_TOPIC.search(turn)) or not any(
        (policy, date, loss_type, injured, description, _LOSS_ANY.search(turn))
    )

    def line(name: str, value: str | None) -> str:
        return f"{name}: {value if value else 'NONE'}"

    return "\n".join(
        [
            line("POLICY", policy),
            line("DATE_OF_LOSS", date),
            line("LOSS_TYPE", loss_type),
            line("DESCRIPTION", description),
            line("INJURED", injured),
            f"OFF_TOPIC: {'yes' if off_topic else 'no'}",
        ]
    )


def loss_type_names() -> tuple[str, ...]:
    """Exposed so a test can assert the fake and the flow agree on the vocabulary."""
    return LOSS_TYPES
