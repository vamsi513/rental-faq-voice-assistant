"""An extracted "nobody was hurt" is only believed when the caller said it.

All MOCKED -- no network, no model. These tests drive the flow with canned extractor
replies standing in for what a model returns, because the thing under test is what the
flow does with a reply, not how any model produces one. PINEWRIGHT MUTUAL DOES NOT EXIST.

The bug these cover was found by running the scenarios against a real model, three times.
Asked to extract fields from "I need to report theft, my bike was stolen from the garage,
policy PW-40281773" and then "October 30", the model answered INJURED: no on the second
turn although the caller had never mentioned injuries. That satisfied the last field, so
the flow skipped "Was anyone injured?" and read back "Anyone injured: no" about something
the caller never said. The prompt now says not to infer no from silence; these tests are
about the guard that does not depend on the model complying.
"""
from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from app.claims import (
    ASK_DATE,
    ASK_INJURED,
    HANDOFF_INJURY,
    ClaimRequest,
    ClaimsIntake,
    is_injury,
    states_injury_outcome,
)

pytestmark = pytest.mark.asyncio

POLICY = "PW-40281773"
THEFT_TURN = "I need to report theft, my bike was stolen from the garage, policy PW-40281773"


def canned(*replies: str) -> Callable[[str, str], str]:
    """An extractor that returns the given replies in order, one per turn."""
    remaining = list(replies)

    def extract(system: str, user: str) -> str:
        assert remaining, "the flow asked for more extractions than the test scripted"
        return remaining.pop(0)

    return extract


def reply(policy="NONE", date="NONE", loss="NONE", description="NONE", injured="NONE",
          off_topic="no") -> str:
    return "\n".join([
        f"POLICY: {policy}",
        f"DATE_OF_LOSS: {date}",
        f"LOSS_TYPE: {loss}",
        f"DESCRIPTION: {description}",
        f"INJURED: {injured}",
        f"OFF_TOPIC: {off_topic}",
    ])


def accepting() -> httpx.AsyncClient:
    """An endpoint that would accept a filing, so a wrong handoff is visible as a
    filing that did not happen rather than as a transport error."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        return httpx.Response(200, json={"status": "accepted", "claim_id": "DEMO-CAB12CD",
                                         "request_id": body["request_id"]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def drive(turns, extractor):
    """One claim through several turns, with an endpoint that would accept a filing."""
    client = accepting()
    intake = ClaimsIntake(extractor, endpoint_url="https://claims.invalid/intake",
                          secret="s", timeout=2.0, client=client)
    request, outcomes = ClaimRequest(), []
    try:
        for text in turns:
            outcome = await intake.handle(request, text, call_id="call-1")
            request = outcome.request
            outcomes.append(outcome)
    finally:
        await client.aclose()
    return request, outcomes


# --- what counts as the caller having said something about injuries -----------

@pytest.mark.parametrize("text", [
    "nobody was hurt", "no one was injured", "we're all fine", "everyone's fine",
    "no", "my wife was injured", "she broke her wrist", "he went to the hospital",
    "someone was hurt",
])
async def test_an_injury_statement_either_way_is_recognised(text):
    assert states_injury_outcome(text)


@pytest.mark.parametrize("text", [
    "October 30", "policy PW-40281773", "a pipe burst in the kitchen",
    "my bike was stolen from the garage", "yes", "water damage to the floor",
])
async def test_a_turn_that_says_nothing_about_injuries_is_not_a_statement(text):
    assert not states_injury_outcome(text)


# --- the guard ----------------------------------------------------------------

async def test_an_unstated_no_is_discarded_and_the_question_is_asked():
    """The recorded real-model failure, reproduced from its transcript."""
    request, outcomes = await drive(
        [THEFT_TURN, "October 30"],
        canned(
            reply(policy=POLICY, loss="theft", description="bike was stolen from the garage"),
            # What the model actually returned: everything carried forward, and a no the
            # caller never said.
            reply(policy=POLICY, date="October 30", loss="theft",
                  description="bike was stolen from the garage", injured="no"),
        ),
    )
    assert outcomes[0].asked_for == "date_of_loss"
    assert outcomes[0].reply == ASK_DATE
    assert outcomes[1].asked_for == "injured"
    assert outcomes[1].reply == ASK_INJURED
    assert request.injured is None
    assert request.status == "collecting"
    # The read back is what made this harmful, so assert it did not happen.
    assert "Anyone injured" not in outcomes[1].reply


async def test_the_discarded_no_does_not_reach_the_read_back():
    """Answering the question the guard forced gets the field set properly."""
    request, outcomes = await drive(
        [THEFT_TURN, "October 30", "no, nobody was hurt"],
        canned(
            reply(policy=POLICY, loss="theft", description="bike was stolen from the garage"),
            reply(policy=POLICY, date="October 30", loss="theft",
                  description="bike was stolen from the garage", injured="no"),
            reply(policy=POLICY, date="October 30", loss="theft",
                  description="bike was stolen from the garage", injured="no"),
        ),
    )
    assert request.injured is False
    assert request.status == "awaiting_confirmation"
    assert "Anyone injured: no" in outcomes[2].reply
    assert not any(o.filed for o in outcomes)


async def test_an_explicit_denial_in_the_callers_words_is_believed():
    """Said in the same breath as the loss, so no extra question is needed."""
    turn = ("My policy is PW-40281773, a pipe burst on October 30 and soaked the kitchen "
            "floor. Nobody was hurt.")
    request, outcomes = await drive(
        [turn],
        canned(reply(policy=POLICY, date="October 30", loss="water",
                     description="a pipe burst and soaked the kitchen floor", injured="no")),
    )
    assert request.injured is False
    assert request.status == "awaiting_confirmation"
    assert "Anyone injured: no" in outcomes[0].reply
    assert outcomes[0].asked_for is None


async def test_an_established_no_is_not_asked_for_a_second_time():
    """The guard must not discard a no it already accepted on an earlier turn."""
    request, outcomes = await drive(
        ["A pipe burst on October 30. Nobody was hurt.", "the policy is PW-40281773"],
        canned(
            reply(date="October 30", loss="water", description="a pipe burst", injured="no"),
            # Carried forward on a turn that says nothing about injuries.
            reply(policy=POLICY, date="October 30", loss="water", description="a pipe burst",
                  injured="no"),
        ),
    )
    assert request.injured is False
    assert outcomes[1].asked_for is None
    assert request.status == "awaiting_confirmation"


async def test_an_injury_stated_as_a_field_still_hands_off():
    """A yes is not second guessed: the caller's turn alone does not read as an injury."""
    turn = "she was shaken up by it"
    assert not is_injury(turn)
    request, outcomes = await drive(
        [THEFT_TURN, "October 30", turn],
        canned(
            reply(policy=POLICY, loss="theft", description="bike was stolen from the garage"),
            reply(policy=POLICY, date="October 30", loss="theft",
                  description="bike was stolen from the garage", injured="no"),
            reply(policy=POLICY, date="October 30", loss="theft",
                  description="bike was stolen from the garage", injured="yes"),
        ),
    )
    assert outcomes[2].handed_off
    assert outcomes[2].refusal == "injury"
    assert outcomes[2].reply == HANDOFF_INJURY
    assert request.status == "handoff"
    assert request.filed_ids == []


async def test_an_injury_stated_as_a_phrase_hands_off_before_extraction():
    """The other of the two paths. The extractor is never consulted."""
    request, outcomes = await drive(["yes, my wife was injured"], canned())
    assert outcomes[0].handed_off
    assert outcomes[0].refusal == "injury"
    assert request.filed_ids == []


async def test_breaking_the_guard_brings_the_bug_back(monkeypatch):
    """Without the guard, the unstated no satisfies the field and skips the question."""
    monkeypatch.setattr("app.claims.states_injury_outcome", lambda text: True)
    request, outcomes = await drive(
        [THEFT_TURN, "October 30"],
        canned(
            reply(policy=POLICY, loss="theft", description="bike was stolen from the garage"),
            reply(policy=POLICY, date="October 30", loss="theft",
                  description="bike was stolen from the garage", injured="no"),
        ),
    )
    assert outcomes[1].asked_for is None
    assert request.injured is False
    assert "Anyone injured: no" in outcomes[1].reply


async def test_the_prompt_tells_the_model_not_to_infer_a_no():
    from app.claims import EXTRACT_SYSTEM

    assert "NONE unless the caller said so themselves" in EXTRACT_SYSTEM
    assert "Do not infer no from" in EXTRACT_SYSTEM
