"""Deterministic scope and grounding controls.

These run either side of the LLM, because a prompt is a request, not a guarantee.
Three independent mechanisms, each cheap and each with a documented blind spot:

  1. detect_injection   -- flags caller turns that try to rewrite the agent's rules.
                           Pattern-based, so it catches the obvious phrasings and
                           will miss a novel one. It does not block the turn; it
                           downgrades the route to a refusal and gets logged.
  2. neutralise         -- wraps retrieved text before it enters the prompt, so any
                           instruction sitting INSIDE the resume is presented as
                           quoted data. Defence against a poisoned source document.
  3. check_sensitive_claim -- post-generation scan. If the model answered about a
                           topic the documents genuinely do not cover (an individual's
                           balance, application status, unit availability), the answer
                           is discarded and replaced with the missing-info reply,
                           whatever the model said.

Limits, stated plainly: 1 and 3 are keyword-driven and defeatable by paraphrase. They
are a safety net under the router and the grounded prompt, not a substitute. None of
this eliminates hallucination; it narrows the paths to it and makes the common ones
cheap to catch.
"""
from __future__ import annotations

import re

# Caller attempts to override the agent's instructions or extract its prompt.
_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(all\s+|your\s+|the\s+)*(previous|prior|above|earlier)\s+(instruction|prompt|rule|direction)", "override_instructions"),
    (r"disregard\s+(all\s+|your\s+|the\s+)*(previous|prior|above|instruction|rule)", "override_instructions"),
    (r"forget\s+(everything|all|your)\s+(you|instruction|rule|prompt)", "override_instructions"),
    # "on" is optional: speech-to-text drops small words, and a live call produced
    # "from now, from now, you are..." with no "on" at all.
    (r"(you\s+are\s+now|from\s+now\s*(on)?\s*,?\s*you\s+are|act\s+as(\s+if)?|pretend\s+(to\s+be|you))", "role_reassignment"),
    (r"\byou\s+are\s+(a|my|an)\s+(general|generalist|normal|helpful|standard)\b", "role_reassignment"),
    (r"(new|updated|revised)\s+(instruction|system\s+prompt|rule)s?\s*:", "override_instructions"),
    (r"(reveal|show|print|repeat|output|what\s+is)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instruction)", "prompt_extraction"),
    (r"developer\s+mode|jailbreak|dan\s+mode|without\s+any\s+restriction", "jailbreak"),
    (r"\bsudo\b|\badmin\s+override\b|\boverride\s+code\b", "fake_authority"),
)

# Caller asks the agent to assert something the documents do not support.
_FABRICATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(just|simply|go ahead and)\s+(say|claim|make up|invent|guess|tell me)", "assert_unsupported"),
    (r"\b(make\s+up|invent|fabricate|made\s+up)\b", "explicit_fabrication"),
    (r"\b(pretend|act like|imagine)\b.{0,30}\b(rent|fee|deposit|policy|available|approved)\b", "assert_unsupported"),
    (r"\b(can\s+you\s+)?(just\s+)?(waive|drop|lower|change|reduce|remove)\b.{0,25}\b(fee|rent|deposit|charge|policy)\b", "policy_change"),
    (r"\b(say|tell)\s+(me\s+)?(it'?s|the rent is|the fee is)\b.{0,25}\b(free|waived|cheaper|\$?\d)", "assert_unsupported"),
    (r"\bcan\s+you\s+(just\s+)?lie\b|\bmake\s+an\s+exception\b", "explicit_fabrication"),
    (r"\b(approve|accept|guarantee)\s+(me|my application|us)\b", "assert_unsupported"),
)

# Topics the documents provably do not cover. This corpus is one property's published
# policies: it has no account system, no availability feed and no applicant records, so
# a confident answer here is a hallucination by construction.
_SENSITIVE_TOPICS: tuple[tuple[str, str], ...] = (
    (r"\b(my|our|the)\s+(balance|account|ledger|statement)\b|\bhow much do i owe\b|\bdo i owe\b", "account_balance"),
    (r"\b(am i|was i|did i get)\s+(approved|accepted|denied|rejected)\b"
     r"|\b(status of|check)\s+my\s+application\b|\bmy application\b", "application_status"),
    (r"\b(is|are)\s+(there\s+)?(any\s+)?(units?|apartments?|anything)\s+(available|open|vacant)\b"
     r"|\bavailability\b|\bwhen can i move in\b|\bcurrent(ly)? available\b", "availability"),
    (r"\b(my|our)\s+(lease|unit|apartment|rent)\b.{0,30}\b(end|expire|due|start|renew)\b"
     r"|\bwhen does my lease\b|\bwhat is my rent\b|\bmy rent amount\b", "individual_lease"),
    (r"\b(book|schedule|reserve|set up)\b.{0,20}\b(tour|showing|viewing|appointment)\b", "booking"),
    (r"\b(credit score|ssn|social security|bank account|card number|routing number)\b", "sensitive_personal"),
    (r"\b(sue|lawsuit|legal advice|lawyer|attorney|tenant rights law|evict(ion|ed)?)\b", "legal_advice"),
    (r"\b(my neighbou?r|the (person|family) (in|next))\b.{0,30}\b(name|phone|who)\b"
     r"|\bwho lives in\b", "other_residents"),
)


# Speech-to-text disfluencies that break literal pattern matching.
_FILLER = re.compile(r"\b(uh+|um+|er+|ah+|hmm+|like|you know|i mean)\b", re.I)


def normalise_speech(text: str) -> str:
    """Strip the artefacts speech-to-text leaves behind, before pattern matching.

    Found by an actual voice call, not by reasoning about it. A caller saying "ignore
    all previous instructions" was transcribed as "ignore all PREVIOUS PREVIOUS
    instruction", and "from now on you are..." became "from now, FROM NOW, you are".
    Both slipped past patterns that matched the typed versions perfectly. The router
    still refused both, so nothing unsafe happened -- but the deterministic layer was
    silently not doing its job over voice, which is the only channel that matters here.

    Collapses filler words, stutters, and immediately repeated words and word pairs.
    """
    cleaned = " ".join(text.lower().split())
    cleaned = _FILLER.sub(" ", cleaned)
    cleaned = re.sub(r"[.]{2,}|--+", " ", cleaned)        # "you are my... you are"
    cleaned = re.sub(r"[,;]", " ", cleaned)               # "from now, from now"
    cleaned = " ".join(cleaned.split())
    # Collapse an immediately repeated word ("previous previous" -> "previous") and an
    # immediately repeated pair ("from now from now" -> "from now"). Repeat until
    # stable, since a stutter can nest ("i'm i'm i'm").
    for _ in range(3):
        before = cleaned
        # [\w']+ not \w+: "i'm i'm" and "that's that's" are common stutters, and an
        # apostrophe is not a word character.
        cleaned = re.sub(r"(?<!\S)([\w']+)(\s+\1)+(?!\S)", r"\1", cleaned)
        cleaned = re.sub(r"(?<!\S)([\w']+\s+[\w']+)(\s+\1)+(?!\S)", r"\1", cleaned)
        if cleaned == before:
            break
    return cleaned


# A whole turn that is only a greeting, thanks, or an acknowledgement. Matched
# deterministically and only when the turn is short and contains nothing else, so
# "Hi, what's the pet fee?" is NOT caught here and goes to the router.
_PLEASANTRY = re.compile(
    r"^(hi|hey|hello|yo|good\s+(morning|afternoon|evening)|greetings)"
    r"(\s+(there|again))?[\s,.!]*(how\s+are\s+you[\s,.!?]*)?$"
    r"|^(thanks|thank\s+you|ta|cheers)([\s,]+(that'?s\s+)?(helpful|great|perfect|useful|all))?[\s,.!]*$"
    r"|^(ok|okay|okey|got\s+it|i\s+see|sure|right|cool|nice|great|perfect|alright)[\s,.!]*$"
    r"|^(bye|goodbye|see\s+you|that'?s\s+all|that'?s\s+it|no\s+thanks|i'?m\s+good)[\s,.!]*$",
    re.I,
)
MAX_PLEASANTRY_WORDS = 6


def is_pleasantry(text: str) -> bool:
    """True when the whole turn is social, with no question attached.

    Checked before reference resolution because "Hi there" and "Thanks, that's helpful"
    both contain words ("there", "that's") that a reference detector reads as pointing
    at something -- which used to make the agent answer a greeting by asking which
    policy the caller meant.

    NOTE: a bare "yes"/"ok" during a pending maintenance confirmation must NOT land
    here. `sanitize` checks for a pending confirmation BEFORE calling this.
    """
    cleaned = normalise_speech(text)
    if not cleaned or len(cleaned.split()) > MAX_PLEASANTRY_WORDS:
        return False
    return bool(_PLEASANTRY.match(cleaned))


def _matches(text: str, patterns: tuple[tuple[str, str], ...]) -> list[str]:
    normalised = normalise_speech(text)
    return sorted({label for pattern, label in patterns if re.search(pattern, normalised)})


def detect_injection(text: str) -> list[str]:
    """Labels for prompt-injection attempts in a caller turn. Empty means clean."""
    return _matches(text, _INJECTION_PATTERNS)


def detect_fabrication_request(text: str) -> list[str]:
    """Labels for requests to assert something the resume does not support."""
    return _matches(text, _FABRICATION_PATTERNS)


def sensitive_topics(text: str) -> list[str]:
    """Topics the resume provably does not cover."""
    return _matches(text, _SENSITIVE_TOPICS)


def neutralise(passage: str) -> str:
    """Present retrieved text as inert data.

    Defangs the instruction-shaped phrasings an attacker would need if they could edit
    the resume, and fences the passage so the model reads it as a quotation. The fence
    matters more than the substitutions: the prompt tells the model that everything
    inside it is evidence, never direction.
    """
    cleaned = re.sub(
        r"(?i)\b(ignore|disregard|forget)\s+(all\s+|your\s+|the\s+)*(previous|prior|above)\b",
        "[redacted instruction]",
        passage,
    )
    cleaned = re.sub(r"(?i)\b(system\s+prompt|new\s+instructions?)\b", "[redacted]", cleaned)
    return cleaned


def check_sensitive_claim(question: str, answer: str) -> str | None:
    """Return the topic label if an answer strays onto an uncovered topic.

    Fires when the question raises such a topic AND the answer did not decline. A
    hedged answer is allowed through; a substantive one is not.
    """
    topics = sensitive_topics(question)
    if not topics:
        return None
    declined = re.search(
        r"(don't have (that|it) in the (property )?documents"
        r"|documents?\s+do(n't| not)\s+(say|cover|mention|include|list)"
        r"|not\s+in\s+the\s+(property\s+)?documents"
        r"|I\s+(don't|do not|can't|cannot)\s+(have|see|access|look\s+(that|it)?\s*up)"
        r"|I\s+can\s+help\s+with\s+questions\s+about)",
        answer,
        re.I,
    )
    return None if declined else topics[0]
