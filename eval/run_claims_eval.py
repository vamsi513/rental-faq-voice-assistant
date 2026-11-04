"""Scenario evaluation for the synthetic claims intake flow.

PINEWRIGHT MUTUAL DOES NOT EXIST. Every policy number, date and claim number in the
scenarios is invented, the endpoint is an in-process mock, and nothing is filed with any
insurer. The results below are 20 scripted conversations, not a measurement of an
insurance product.

WHAT THIS IS, PRECISELY: rule based scoring, not semantic judgement. No model grades the
output. Each rule is a comparison against the flow's own recorded state, which is the
same choice `eval/run_grounding_eval.py` makes and for the same reason: "did not file" is
the claim that matters, and only the state proves it.

Two modes, and the only difference between them is the field extractor:

  --fake  (default)  the deterministic extractor in app/claims_fake_model.py.
                     No model, no network of any kind. This is what runs in tests.
  --real             the configured chat model does the extraction.

In BOTH modes the claims endpoint is an in-process httpx mock transport, so neither mode
posts anything anywhere. --real spends tokens on extraction and nothing else.

    python eval/run_claims_eval.py                      # offline, no key needed
    python eval/run_claims_eval.py --real               # needs OPENAI_API_KEY
    python eval/run_claims_eval.py --json out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.claims import ClaimRequest, ClaimsIntake  # noqa: E402

SCENARIOS = ROOT / "eval/claims_scenarios.json"
ENDPOINT_URL = "https://claims.invalid/intake"

RULE_NAMES = (
    "fields_extracted",
    "filed_expected_times",
    "no_coverage_promise",
    "handoff_on_injury",
    "turn_expectations",
    "final_status",
)

# An affirmative coverage or payment promise. Phrased to exclude the flow's own
# refusals, which necessarily contain the words: "I can't tell you whether this is
# covered" must not score as a promise, so a negated form is checked for first.
_PROMISE = re.compile(
    r"\b(it|that|this) is covered\b"
    r"|\byou(?:'re| are) covered\b"
    r"|\bwe(?:'ll| will) cover\b"
    r"|\byour claim (?:will be|is) (?:approved|paid|accepted)\b"
    r"|\bdefinitely covered\b"
    r"|\bguarantee(?:d)?\s+(?:that\s+)?(?:it|this|your claim|coverage)\b"
    r"|\byou(?:'ll| will) (?:get|receive)\s+\$?\d",
    re.I,
)
_NEGATED = re.compile(
    r"\b(can'?t|cannot|won'?t|will not|unable to|don'?t)\b.{0,40}\b(tell|say|decide|promise|guarantee)\b"
    r"|\bwhether\b"
    r"|\bdepends on\b",
    re.I,
)


def promises_coverage(reply: str) -> bool:
    """True only for an affirmative promise, not for a refusal that mentions coverage."""
    if _NEGATED.search(reply or ""):
        return False
    return bool(_PROMISE.search(reply or ""))


def mock_endpoint(mode: str, calls: list[dict]) -> httpx.AsyncClient:
    """The claims endpoint, in process. Mirrors scripts/mock_claims.py's modes."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        calls.append(body)
        if mode == "timeout":
            raise httpx.TimeoutException("simulated timeout")
        if mode == "error":
            return httpx.Response(500, text="simulated endpoint failure")
        if mode == "malformed":
            return httpx.Response(200, text="not json at all")
        claim_id = seen.setdefault(body["request_id"], "DEMO-CAB12CD")
        echoed = "clm-does-not-match" if mode == "mismatch" else body["request_id"]
        return httpx.Response(
            200, json={"status": "accepted", "claim_id": claim_id, "request_id": echoed}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def run_scenario(
    scenario: dict[str, Any], extract: Callable[[str, str], str]
) -> dict[str, Any]:
    """Drive one scripted conversation and record what the flow did."""
    calls: list[dict] = []
    client = mock_endpoint(scenario.get("endpoint", "ok"), calls)
    intake = ClaimsIntake(
        extract,
        endpoint_url=ENDPOINT_URL,
        secret="scenario-secret",
        timeout=2.0,
        client=client,
    )
    request = ClaimRequest()
    turns: list[dict[str, Any]] = []
    history: list[str] = []

    try:
        for turn in scenario["turns"]:
            outcome = await intake.handle(
                request, turn["text"], call_id=f"eval-{scenario['id']}",
                history="\n".join(history[-8:]),
            )
            request = outcome.request
            history += [f"caller: {turn['text']}", f"agent: {outcome.reply}"]
            turns.append({
                "text": turn["text"],
                "reply": outcome.reply,
                "status": request.status,
                "filed": outcome.filed,
                "handed_off": outcome.handed_off,
                "refusal": outcome.refusal,
                "asked_for": outcome.asked_for,
            })
    finally:
        await client.aclose()

    return {
        "id": scenario["id"],
        "category": scenario["category"],
        "notes": scenario["notes"],
        "endpoint": scenario.get("endpoint", "ok"),
        "turns": turns,
        "final": request.to_state(),
        "endpoint_calls": len(calls),
    }


def _check_fields(scenario: dict, record: dict) -> tuple[bool | None, str]:
    wanted = (scenario["expect"].get("fields") or {}).copy()
    if not wanted:
        return None, ""
    final = record["final"]
    contains = wanted.pop("description_contains", None)
    for name, value in wanted.items():
        if final.get(name) != value:
            return False, f"{name} was {final.get(name)!r}, expected {value!r}"
    if contains and contains.lower() not in (final.get("description") or "").lower():
        return False, f"description {final.get('description')!r} lacks {contains!r}"
    return True, ""


def _check_filings(scenario: dict, record: dict) -> tuple[bool, str]:
    expected = scenario["expect"]["filings"]
    filed = sum(1 for turn in record["turns"] if turn["filed"])
    if filed != expected:
        return False, f"filed {filed} times, expected {expected}"
    if filed > 1:
        return False, "filed more than once, which this flow must never do"
    wanted_calls = scenario["expect"].get("endpoint_calls")
    if wanted_calls is not None and record["endpoint_calls"] != wanted_calls:
        return False, f"endpoint called {record['endpoint_calls']} times, expected {wanted_calls}"
    return True, ""


def _check_no_promise(record: dict) -> tuple[bool, str]:
    for index, turn in enumerate(record["turns"], 1):
        if promises_coverage(turn["reply"]):
            return False, f"turn {index} promised coverage or a payment: {turn['reply'][:80]!r}"
    return True, ""


def _check_handoff(scenario: dict, record: dict) -> tuple[bool | None, str]:
    if not scenario["expect"].get("handoff"):
        return None, ""
    if not any(turn["handed_off"] for turn in record["turns"]):
        return False, "never handed off"
    if record["final"]["status"] != "handoff":
        return False, f"final status was {record['final']['status']!r}, expected handoff"
    if any(turn["filed"] for turn in record["turns"]):
        return False, "filed something despite handing off"
    return True, ""


def _check_turns(scenario: dict, record: dict) -> tuple[bool | None, str]:
    checked = False
    for index, (expected, actual) in enumerate(
        zip(scenario["turns"], record["turns"], strict=True), 1
    ):
        wanted = expected.get("expect")
        if not wanted:
            continue
        for key, value in wanted.items():
            checked = True
            if actual.get(key) != value:
                return False, f"turn {index} {key} was {actual.get(key)!r}, expected {value!r}"
    return (True, "") if checked else (None, "")


def _check_status(scenario: dict, record: dict) -> tuple[bool | None, str]:
    expected = scenario["expect"].get("status")
    if expected is None:
        return None, ""
    actual = record["final"]["status"]
    return (True, "") if actual == expected else (False, f"status {actual!r}, expected {expected!r}")


def score(scenario: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Apply every rule to one recorded scenario."""
    verdicts: dict[str, str] = {}
    failures: list[str] = []

    for name, (ok, detail) in {
        "fields_extracted": _check_fields(scenario, record),
        "filed_expected_times": _check_filings(scenario, record),
        "no_coverage_promise": _check_no_promise(record),
        "handoff_on_injury": _check_handoff(scenario, record),
        "turn_expectations": _check_turns(scenario, record),
        "final_status": _check_status(scenario, record),
    }.items():
        if ok is None:
            verdicts[name] = "not_applicable"
            continue
        verdicts[name] = "pass" if ok else "fail"
        if not ok:
            failures.append(f"{name}: {detail}")

    return {"rules": verdicts, "failures": failures, "passed": not failures}


def build_extractor(real: bool) -> tuple[Callable[[str, str], str], str]:
    if not real:
        from app.claims_fake_model import extract

        return extract, "fake (app/claims_fake_model.py, no model, no network)"

    from langchain_openai import ChatOpenAI

    from app.config import settings

    if not settings.openai_api_key:
        sys.exit(
            "--real needs OPENAI_API_KEY. Run without --real for the offline mode, "
            "which needs no key."
        )
    llm = ChatOpenAI(
        model=settings.chat_model, api_key=settings.openai_api_key, timeout=12, max_retries=1
    )

    def extract(system: str, user: str) -> str:
        reply = llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=160,
            temperature=0.0,
        )
        return str(reply.content)

    return extract, f"real ({settings.chat_model}); the endpoint is still a mock"


async def main_async(arguments: argparse.Namespace) -> int:
    document = json.loads(arguments.scenarios.read_text())
    scenarios = document["scenarios"]
    if arguments.only:
        wanted = set(arguments.only)
        scenarios = [s for s in scenarios if s["id"] in wanted or s["category"] in wanted]
    if not scenarios:
        sys.exit("no scenarios selected")

    extract, described = build_extractor(arguments.real)
    print(f"extractor : {described}")
    print("endpoint  : in-process mock, nothing leaves this process")
    print(f"scenarios : {len(scenarios)} from {arguments.scenarios.name}")
    print("Pinewright Mutual is fictional; no claim is filed with any insurer.\n")

    records, scores = [], []
    for index, scenario in enumerate(scenarios, 1):
        record = await run_scenario(scenario, extract)
        result = score(scenario, record)
        records.append(record)
        scores.append(result)
        mark = "pass" if result["passed"] else "FAIL"
        print(f"[{index:>3}/{len(scenarios)}] {mark}  {scenario['category']:<20} {scenario['id']}")
        for failure in result["failures"]:
            print(f"            {failure}")

    passed = sum(r["passed"] for r in scores)
    per_rule: dict[str, dict[str, int]] = {
        name: {"pass": 0, "fail": 0, "not_applicable": 0} for name in RULE_NAMES
    }
    per_category: dict[str, dict[str, int]] = {}
    for scenario, result in zip(scenarios, scores, strict=True):
        for name, verdict in result["rules"].items():
            per_rule[name][verdict] += 1
        bucket = per_category.setdefault(scenario["category"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += result["passed"]

    print(f"\n{passed}/{len(scenarios)} scenarios passed")
    print("\nby rule:")
    for name, counts in per_rule.items():
        print(f"  {name:<24} pass={counts['pass']:<3} fail={counts['fail']:<3} n/a={counts['not_applicable']}")
    print("\nby category:")
    for name, counts in sorted(per_category.items()):
        print(f"  {name:<22} {counts['passed']}/{counts['total']}")
    print(
        "\nThese are rule checks on 20 scripted conversations written by the same person "
        "who wrote the flow. They are not a measure of general correctness."
    )

    if arguments.json:
        arguments.json.write_text(json.dumps({
            "extractor": described,
            "passed": passed,
            "total": len(scenarios),
            "per_rule": per_rule,
            "per_category": per_category,
            "results": [{**record, **result} for record, result in zip(records, scores, strict=True)],
        }, indent=2))
        print(f"\nfull results -> {arguments.json}")

    return 0 if passed == len(scenarios) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true",
                        help="Use the configured chat model for extraction. Needs OPENAI_API_KEY.")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS)
    parser.add_argument("--only", action="append", help="Run only this scenario id or category.")
    parser.add_argument("--json", type=Path)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
