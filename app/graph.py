r"""The LangGraph workflow.

Flow (one pass per caller turn):

    sanitize -> interpret -> classify -> [retrieve -> generate] -> END
                                      \-> refuse -----------------/

  sanitize   deterministic scope checks on the raw turn; no model call
  interpret  resolves follow-up references into a standalone query (model call,
             only when the turn looks referential)
  classify   routes the turn (model call)
  retrieve   hybrid search over the resume; applies the low similarity backstop
  generate   grounded answer, then the post-generation sensitive-topic check
  refuse     fixed refusal strings, no model call

Nodes are split where the split buys something. `sanitize` is separate because it must
run before any model sees the turn. `classify` is separate from `generate` so the scope
decision is made before resume text is in context. `interpret` is separate because it
is the only node that needs the conversation and it is skipped on most turns. There is
no second agent here -- one conversation, one decision per turn.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from contextvars import ContextVar
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import maintenance as mx
from app import prompts, scope
from app.config import Settings
from app.retriever import PolicyRetriever

log = logging.getLogger("agent.graph")

# Where a streaming caller parks its token sink for the duration of one turn.
#
# A ContextVar rather than a graph-config entry because the sink is a live coroutine:
# it cannot go in AgentState, which is serialised into the checkpoint, and threading it
# through the run config made LangGraph warn about the node signature. A ContextVar is
# set and reset around a single `ainvoke`, and asyncio copies the context into the
# tasks LangGraph spawns, so each concurrent turn sees only its own sink.
TOKEN_SINK: ContextVar[Any] = ContextVar("token_sink", default=None)

Route = Literal[
    "property_question", "maintenance", "emergency", "small_talk",
    "out_of_scope", "fabrication", "injection", "missing_info", "needs_context",
]

# A turn needs reference resolution only if it points at something unnamed.
#
# Deliberately narrow. An earlier, looser version matched bare "there", "that" and
# "the first", so "Hi there", "Thanks, that's helpful" and "introduce yourself in the
# first person" were all treated as ambiguous follow-ups. Over-matching here is
# expensive: it either costs a wasted model call or hijacks the turn entirely. A
# reference word now has to be doing referential work -- pointing at a noun, or
# standing alone as a pronoun subject.
_REFERENTIAL = re.compile(
    # A bare pronoun as the subject of a question: "is that refundable?", "does it
    # apply?". This is the common shape of a rental follow-up and the regex missed it
    # entirely at first, so "is that refundable?" after "what is the pet fee?" retrieved
    # the security-deposit section and answered about the wrong charge.
    r"\b(is|are|was|were|does|do|did|can|could|will|would|should|has|have)\s+"
    r"(that|this|those|these|it|they)\b"
    r"|\b(it|its)\b"                                            # "what does it cover"
    r"|\b(how much|what about|when is|where is)\s+(that|this|it|those)\b"
    # A demonstrative pointing at a noun this corpus contains. "the" is deliberately
    # NOT here: "what is the pet fee?" is a standalone question, and including it made
    # every such question look referential and pay for a pointless rewrite call.
    r"|\b(that|this|those|these)\s+"
    r"(fee|fees|deposit|deposits|charge|charges|cost|price|rate|rates|policy|rule|rules"
    r"|rent|lease|pet|pets|space|spaces|spot|unit|units|apartment|amenity|amenities"
    r"|one|ones|option|options|plan|plans|floor\s?plan)\b"
    r"|\b(of|about)\s+(those|them|that)\b"                      # "two of those"
    r"|\b(more about (it|that|this|them)|tell me more|what else|anything else)\b"
    r"|\bthe (first|second|third|last) (one|option|plan)\b",
    re.I,
)


class AgentState(TypedDict, total=False):
    """State for one call (one LangGraph thread).

    `messages` is the only reducer field: `add_messages` appends, so each turn adds the
    caller's turn and the agent's reply and the checkpoint grows by two. Everything else
    is per-turn scratch, overwritten on each pass -- the checkpoint keeps the last
    turn's values, which is what the debug panel reads.
    """

    # --- conversation (accumulates across turns) ---------------------------
    messages: Annotated[list[AnyMessage], add_messages]

    # --- session identity --------------------------------------------------
    call_id: str          # Vapi call id; also the LangGraph thread id
    turn_id: str          # unique per turn, for dedupe and log correlation

    # --- per-turn working values -------------------------------------------
    question: str         # the caller's turn, verbatim
    resolved_question: str  # after follow-up resolution; what the retriever sees
    route: Route
    route_reason: str
    flags: list[str]      # injection / fabrication labels from the sanitize node

    # --- retrieval evidence ------------------------------------------------
    evidence: list[dict]  # debug-shaped retrieved chunks, best first
    top_dense_score: float

    # --- maintenance request (persists across turns, like `messages`) --------
    # Deliberately NOT reset by `sanitize`: a request is collected over several turns
    # and must survive until it is submitted, cancelled, or the call ends.
    maintenance: dict

    # --- output ------------------------------------------------------------
    answer: str
    node_timings: dict[str, float]


# Capitalised words the rewriter may always use: they refer to the subject, not to a
# project or employer it would have to have found in the conversation.
_ALWAYS_ALLOWED = {"maple", "court", "apartments", "i", "my", "we", "our", "you",
                   "what", "which", "who", "where", "when", "how", "why", "did",
                   "does", "is", "are", "tell", "the", "in", "and", "of", "a", "an",
                   "do", "can", "was", "it", "that", "this"}


def _invented_entities(rewritten: str, history: list[AnyMessage], question: str) -> list[str]:
    """Capitalised tokens in the rewrite that appear nowhere in its inputs."""
    known = " ".join([question] + [str(m.content) for m in history]).lower()
    out = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9.+-]{2,}\b", rewritten):
        lowered = token.lower().strip(".")
        if lowered in _ALWAYS_ALLOWED or lowered in known:
            continue
        out.append(token)
    return sorted(set(out))


def _parse_extraction(raw: str) -> tuple[str | None, str | None]:
    """Read the two-line UNIT/ISSUE reply. NONE and empties become None."""
    unit = issue = None
    for line in (raw or "").splitlines():
        key, _, value = line.partition(":")
        value = value.strip().strip('"')
        if not value or value.upper() == "NONE":
            continue
        if key.strip().upper() == "UNIT":
            unit = value
        elif key.strip().upper() == "ISSUE":
            issue = value
    return unit, issue


def _history(messages: list[AnyMessage], limit: int) -> str:
    """Render the last `limit` messages as plain transcript for the router/rewriter."""
    recent = messages[-limit:] if limit else messages
    lines = []
    for message in recent:
        speaker = "Caller" if isinstance(message, HumanMessage) else "Assistant"
        lines.append(f"{speaker}: {message.content}")
    return "\n".join(lines) if lines else "(start of call)"


def _coverage_outline(retriever: PolicyRetriever) -> str:
    """Section -> entry titles, built from the indexed chunks.

    The router sees this so it can tell "what's the pet policy" (covered here) from
    "what are rents like in Chicago" (not). Titles only: the router must not be able to
    answer from it, only to recognise topics. Regenerates if the documents change.
    """
    grouped: dict[str, list[str]] = {}
    for chunk in retriever.chunks:
        titles = grouped.setdefault(chunk.section, [])
        if chunk.entry and chunk.entry not in titles:
            titles.append(chunk.entry)
    lines = []
    for section, titles in grouped.items():
        lines.append(f"- {section}" + (f": {'; '.join(titles)}" if titles else ""))
    return "\n".join(lines)


def build_graph(
    retriever: PolicyRetriever,
    llm: Any,
    settings: Settings,
    checkpointer: Any,
    http_client: Any = None,
):
    """Compile the workflow.

    `llm` is anything with `.invoke(messages) -> .content`. `http_client` is an optional
    shared httpx.AsyncClient for the n8n call; tests pass a transport-mocked one.
    """
    outline = _coverage_outline(retriever)

    # How many characters to hold back before releasing the first token to a streaming
    # client. The generator signals "I have no supporting evidence" by replying with the
    # bare sentinel, so streaming it would show the caller a control word. Holding back
    # exactly the sentinel's length is enough to tell them apart, and costs one chunk of
    # latency on real answers.
    _HOLDBACK = len(prompts.NO_EVIDENCE_SENTINEL)

    async def _stream_call(
        system: str, user: str, max_tokens: int, temperature: float, on_token: Any
    ) -> str:
        """Stream the model's reply to `on_token`, returning the complete text.

        Genuinely token by token: each chunk is forwarded as it arrives from the API,
        not reassembled and replayed afterwards. The only delay is the holdback above.
        """
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        full, buffer, released = "", "", False
        async for chunk in llm.astream(messages, max_tokens=max_tokens, temperature=temperature):
            piece = chunk.content or ""
            if not piece:
                continue
            full += piece
            if released:
                await on_token(piece)
                continue
            buffer += piece
            if len(buffer.strip()) < _HOLDBACK:
                continue
            if buffer.strip().upper().startswith(prompts.NO_EVIDENCE_SENTINEL):
                return full          # a refusal is coming; stream nothing
            released = True
            await on_token(buffer)
        if (
            not released
            and buffer.strip()
            and not buffer.strip().upper().startswith(prompts.NO_EVIDENCE_SENTINEL)
        ):
            await on_token(buffer)   # answer shorter than the holdback
        return full

    def _call(system: str, user: str, max_tokens: int, temperature: float) -> str:
        response = llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (response.content or "").strip()

    # --- nodes -------------------------------------------------------------

    def sanitize(state: AgentState) -> dict:
        """Deterministic checks on the raw turn, before any model call."""
        started = time.perf_counter()
        question = (state.get("question") or "").strip()
        injection = scope.detect_injection(question)
        fabrication = scope.detect_fabrication_request(question)
        uncovered = scope.sensitive_topics(question)
        flags = (
            [f"injection:{f}" for f in injection]
            + [f"fabrication:{f}" for f in fabrication]
            + [f"uncovered:{f}" for f in uncovered]
        )
        if flags:
            log.warning("turn=%s flags=%s", state.get("turn_id"), flags)

        update: dict = {
            "flags": flags,
            # Default; `interpret` overwrites it when the turn is referential.
            "resolved_question": question,
            "node_timings": {"sanitize": time.perf_counter() - started},
            # Reset the per-turn scratch fields. Only `messages` is meant to accumulate;
            # everything else is overwritten each pass. The checkpoint still holds LAST
            # turn's values when this node runs, so without an explicit reset here the
            # previous turn's `route` would make the downstream "has a route already
            # been decided?" guards fire on every turn after the first -- which silently
            # disabled follow-up resolution. Stale evidence would also surface in the
            # debug panel on a refusal turn that retrieved nothing.
            "route": None,
            "route_reason": "",
            "evidence": [],
            "top_dense_score": 0.0,
            "answer": "",
        }
        # Decide the route here, not in `classify`. These checks are deterministic and
        # must outrank both the reference resolver and the router: an injection attempt
        # that happens to contain a referential phrase ("tell me about the first
        # person") would otherwise be routed as an ambiguous follow-up and never
        # reach the injection check at all.
        pending = mx.MaintenanceRequest.from_state(state.get("maintenance"))

        if mx.is_emergency(question):
            # Read the documented emergency instruction instead of offering the routine
            # workflow. Never improvised, never routed through a model.
            update |= {"route": "emergency", "route_reason": "emergency keywords matched"}
        elif pending.status == "awaiting_confirmation" and mx.answers_pending_confirmation(question):
            # The caller is answering "should I submit this?". "Yes" would otherwise
            # classify as small talk and the confirmation would be lost. Only turns that
            # actually answer are captured: a pending request must not hijack an
            # unrelated question asked while it waits.
            update |= {"route": "maintenance", "route_reason": "answering a pending confirmation"}
        elif pending.status == "submitted" and mx.is_confirmation(question):
            # A repeated "yes" after submission, or Vapi re-delivering the turn. Routed
            # here so the node can say it is already done rather than starting over.
            update |= {"route": "maintenance", "route_reason": "confirmation after submit"}
        elif scope.is_pleasantry(question):
            # Decided here so a greeting cannot be read as an ambiguous follow-up.
            update |= {"route": "small_talk", "route_reason": "whole turn is a pleasantry"}
        elif injection:
            update |= {"route": "injection", "route_reason": f"pattern match: {flags}"}
        elif fabrication:
            update |= {"route": "fabrication", "route_reason": f"pattern match: {flags}"}
        elif uncovered:
            # About this property, but on a topic the documents provably do not cover
            # (an individual's balance, application status, lease dates, availability).
            # That is "relevant but missing", not "unrelated", and no retrieval or
            # generation can change it.
            update |= {"route": "missing_info",
                       "route_reason": f"topic not covered by a resume: {flags}"}
        return update

    def interpret(state: AgentState) -> dict:
        """Resolve follow-up references so the retriever gets a standalone query."""
        started = time.perf_counter()
        if state.get("route"):
            # sanitize already decided; resolving references would be wasted work and
            # could overwrite that decision.
            return {"node_timings": {"interpret_skipped": time.perf_counter() - started}}
        question = state.get("question", "")
        history = state.get("messages", [])
        referential = bool(_REFERENTIAL.search(question))
        if referential and not history:
            # "What did he use in that project?" as the first thing said on a call.
            # There is no antecedent, so any answer would be picking a project at
            # random. Ask instead. This is also what makes session isolation visible:
            # the same turn that resolves fine mid-call cannot resolve on a fresh one.
            return {"route": "needs_context", "route_reason": "referential turn with no prior turns",
                    "node_timings": {"interpret_no_antecedent": time.perf_counter() - started}}
        # Skip the call when there is nothing to resolve against, or nothing to resolve.
        if not history or not referential:
            return {"node_timings": {"interpret_skipped": time.perf_counter() - started}}
        try:
            rewritten = _call(
                prompts.REWRITE_SYSTEM,
                prompts.REWRITE_USER.format(
                    history=_history(history, settings.max_history_turns * 2),
                    question=question,
                ),
                max_tokens=80,
                temperature=0.0,
            )
        except Exception:
            log.exception("interpret failed; using the raw turn")
            return {"node_timings": {"interpret_error": time.perf_counter() - started}}
        if prompts.NO_ANTECEDENT_SENTINEL in rewritten.upper():
            return {"route": "needs_context", "route_reason": "rewriter found no antecedent",
                    "node_timings": {"interpret_no_antecedent": time.perf_counter() - started}}

        # A rewrite that balloons is a sign the model answered instead of rewriting.
        if not rewritten or len(rewritten) >= max(240, len(question) * 6):
            return {"node_timings": {"interpret_rejected": time.perf_counter() - started}}

        # The rewriter is an LLM, so it can invent the antecedent it was asked to find.
        # Reject any rewrite that introduces a capitalised entity which appears neither
        # in the conversation nor in the original turn -- that is a hallucinated subject,
        # and answering it would be confidently wrong about a different project.
        invented = _invented_entities(rewritten, history, question)
        if invented:
            log.warning("turn=%s rewrite invented %s; asking for clarification",
                        state.get("turn_id"), invented)
            return {"route": "needs_context",
                    "route_reason": f"rewrite introduced unsupported entity: {invented}",
                    "node_timings": {"interpret_invented": time.perf_counter() - started}}

        return {
            "resolved_question": rewritten,
            "node_timings": {"interpret": time.perf_counter() - started},
        }

    def classify(state: AgentState) -> dict:
        """Route the turn. Deterministic flags win over the model's opinion."""
        started = time.perf_counter()
        if state.get("route"):
            # Decided by sanitize (deterministic) or interpret (no antecedent).
            return {"node_timings": {"classify_skipped": time.perf_counter() - started}}

        try:
            label = _call(
                prompts.ROUTER_SYSTEM,
                prompts.ROUTER_USER.format(
                    outline=outline,
                    history=_history(state.get("messages", []), settings.max_history_turns * 2),
                    question=state.get("question", ""),
                ),
                max_tokens=8,
                temperature=0.0,
            ).upper()
        except Exception:
            # Fail toward attempting an answer: the generator is itself grounded, so a
            # router outage degrades to "retrieve and let grounding decide", not to
            # answering from world knowledge.
            log.exception("router failed; defaulting to resume_question")
            return {"route": "property_question", "route_reason": "router error, defaulted",
                    "node_timings": {"classify_error": time.perf_counter() - started}}

        mapping = {
            "PROPERTY_QUESTION": "property_question",
            "MAINTENANCE_REQUEST": "maintenance",
            "SMALL_TALK": "small_talk",
            "FABRICATION_REQUEST": "fabrication",
            "OUT_OF_SCOPE": "out_of_scope",
        }
        route = next((v for k, v in mapping.items() if k in label), "property_question")
        return {
            "route": route,
            "route_reason": f"router returned {label!r}",
            "node_timings": {"classify": time.perf_counter() - started},
        }

    def retrieve(state: AgentState) -> dict:
        """Hybrid search, then the low-similarity backstop."""
        started = time.perf_counter()
        query = state.get("resolved_question") or state.get("question", "")
        hits = retriever.search(query, settings.top_k)
        evidence = [h.as_debug() for h in hits]
        top = max((h.dense_score for h in hits), default=0.0)
        update: dict = {
            "evidence": evidence,
            "top_dense_score": top,
            "node_timings": {"retrieve": time.perf_counter() - started},
        }

        if state.get("route") == "out_of_scope":
            # The router said no. Overrule it only on strong evidence; otherwise leave
            # the refusal standing and drop the evidence so the debug panel doesn't
            # imply the agent nearly answered.
            if top >= settings.high_confidence_similarity:
                update["route"] = "property_question"
                update["route_reason"] = (
                    f"router said out_of_scope, overruled: top dense score {top:.3f} "
                    f">= {settings.high_confidence_similarity}"
                )
            else:
                update["evidence"] = []
            return update

        if top < settings.min_similarity:
            # Backstop only. Tuned low on purpose -- see Settings.min_similarity.
            update["route"] = "missing_info"
            update["route_reason"] = f"top dense score {top:.3f} < {settings.min_similarity}"
        return update

    async def generate(state: AgentState) -> dict:
        """Grounded answer, then the post-generation sensitive-topic check.

        When TOKEN_SINK holds a coroutine (the WebSocket path), the answer is streamed
        to it as the model produces it. The post-checks below still run on the finished
        text, so a streamed answer can still be retracted -- the socket protocol has a
        `reset` event for exactly that.
        """
        started = time.perf_counter()
        on_token = TOKEN_SINK.get()
        evidence = state.get("evidence", [])
        # Neutralise and fence each passage so document text reads as data, not direction.
        blocks = "\n\n".join(
            f"--- passage {i} ---\n{scope.neutralise(e['text'])}"
            for i, e in enumerate(evidence, start=1)
        )
        question = state.get("resolved_question") or state.get("question", "")
        user_prompt = prompts.ANSWER_USER.format(evidence=blocks, question=question)
        try:
            if on_token is not None:
                answer = await _stream_call(
                    prompts.ANSWER_SYSTEM, user_prompt,
                    settings.max_answer_tokens, settings.temperature, on_token,
                )
            else:
                answer = _call(
                    prompts.ANSWER_SYSTEM, user_prompt,
                    max_tokens=settings.max_answer_tokens,
                    temperature=settings.temperature,
                )
        except Exception:
            log.exception("generation failed")
            return {
                "answer": "Sorry, I had trouble with that. Could you ask again?",
                "route": "property_question",
                "route_reason": "generation error",
                "node_timings": {"generate_error": time.perf_counter() - started},
            }

        # The model's own no-evidence signal. This is the layer that catches a question
        # whose topic retrieved well but whose specific detail isn't in the resume.
        if prompts.NO_EVIDENCE_SENTINEL in answer.upper() or not answer:
            return {
                "answer": prompts.REFUSALS["missing_info"],
                "route": "missing_info",
                "route_reason": "generator reported no supporting evidence",
                "node_timings": {"generate": time.perf_counter() - started},
            }

        # Last layer: the question was about something the resume provably omits and
        # the model answered anyway. Discard whatever it said.
        breach = scope.check_sensitive_claim(state.get("question", ""), answer)
        if breach:
            log.warning("turn=%s discarded answer on uncovered topic %s", state.get("turn_id"), breach)
            return {
                "answer": prompts.REFUSALS["missing_info"],
                "route": "missing_info",
                "route_reason": f"post-check: answered on uncovered topic {breach}",
                "node_timings": {"generate": time.perf_counter() - started},
            }

        return {"answer": answer, "node_timings": {"generate": time.perf_counter() - started}}

    def small_talk(state: AgentState) -> dict:
        started = time.perf_counter()
        try:
            answer = _call(
                prompts.SMALL_TALK_SYSTEM,
                f"Caller said: {state.get('question','')}",
                max_tokens=60,
                temperature=0.4,
            )
        except Exception:
            log.exception("small talk generation failed")
            answer = ("Happy to help. Ask me about rent, the lease, pets, parking, "
                      "amenities or maintenance.")
        return {"answer": answer, "node_timings": {"small_talk": time.perf_counter() - started}}

    async def maintenance(state: AgentState) -> dict:
        """Collect, confirm, submit. The only node that leaves the process.

        Three rules make this safe to demo:
          * Submission happens only on an explicit confirmation matched by pattern, not
            by a model's judgement (`mx.is_confirmation`).
          * Success is announced only after n8n returns a validated 2xx with a ticket.
          * A `request_id` already in `submitted_ids` is never sent again, so a repeated
            "yes" or a duplicate delivery from Vapi cannot create a second ticket.
        """
        started = time.perf_counter()
        question = state.get("question", "")
        pending = mx.MaintenanceRequest.from_state(state.get("maintenance"))

        # --- answering "should I submit this?" ------------------------------
        if pending.status == "awaiting_confirmation":
            if mx.is_decline(question):
                pending.status = "cancelled"
                return {"maintenance": pending.to_state(), "answer": prompts.CANCELLED,
                        "route": "maintenance", "route_reason": "caller declined",
                        "node_timings": {"maintenance": time.perf_counter() - started}}

            if mx.is_confirmation(question):
                if pending.already_submitted:
                    # A repeated "yes", or Vapi delivering the same turn twice.
                    return {"maintenance": pending.to_state(),
                            "answer": prompts.ALREADY_SUBMITTED.format(ticket_id=pending.ticket_id),
                            "route": "maintenance", "route_reason": "duplicate confirmation ignored",
                            "node_timings": {"maintenance": time.perf_counter() - started}}

                result = await mx.submit_to_n8n(
                    pending, state.get("call_id", ""),
                    webhook_url=settings.n8n_webhook_url,
                    secret=settings.n8n_webhook_secret,
                    header_name=settings.n8n_header_name,
                    timeout=settings.n8n_timeout_seconds,
                    client=http_client,
                )
                if result.ok:
                    pending.status = "submitted"
                    pending.ticket_id = result.ticket_id
                    pending.submitted_ids = [*pending.submitted_ids, pending.request_id]
                    answer = prompts.submitted_reply(result.ticket_id or "")
                    reason = f"n8n accepted: {result.ticket_id}"
                else:
                    # Mark failed and do NOT retry: n8n may already have the request.
                    pending.status = "failed"
                    pending.error = result.error
                    answer = prompts.SUBMIT_FAILED
                    reason = f"n8n failed: {result.error}"
                    log.warning("maintenance submit failed id=%s: %s", pending.request_id, result.error)
                return {"maintenance": pending.to_state(), "answer": answer,
                        "route": "maintenance", "route_reason": reason,
                        "node_timings": {"maintenance": time.perf_counter() - started}}
            # Neither yes nor no: fall through to re-extract, in case they corrected a
            # detail ("actually it's 3C"), then re-confirm.

        # A bare "yes" AFTER a successful submit is the caller repeating themselves, or
        # Vapi re-delivering the turn. Without this it falls through to collection,
        # re-extracts the same details from history and asks for confirmation again --
        # which does not duplicate the ticket, but is confusing and invites a second
        # "yes" that would.
        if pending.status == "submitted" and mx.is_confirmation(question):
            return {"maintenance": pending.to_state(),
                    "answer": prompts.ALREADY_SUBMITTED.format(ticket_id=pending.ticket_id),
                    "route": "maintenance", "route_reason": "confirmation after submit ignored",
                    "node_timings": {"maintenance": time.perf_counter() - started}}

        # --- collecting details ---------------------------------------------
        if pending.status in ("idle", "submitted", "cancelled", "failed"):
            pending.start()

        try:
            extracted = _call(
                prompts.MAINTENANCE_EXTRACT_SYSTEM,
                prompts.MAINTENANCE_EXTRACT_USER.format(
                    history=_history(state.get("messages", []), settings.max_history_turns * 2),
                    unit=pending.unit or "NONE", issue=pending.issue or "NONE",
                    question=question,
                ),
                max_tokens=90, temperature=0.0,
            )
        except Exception:
            log.exception("maintenance extraction failed")
            extracted = ""

        unit, issue = _parse_extraction(extracted)
        if unit:
            pending.unit = unit
        if issue:
            pending.issue = issue

        if not pending.ready:
            pending.status = "collecting"
            answer = prompts.ASK_UNIT if pending.issue else prompts.ASK_ISSUE
            return {"maintenance": pending.to_state(), "answer": answer,
                    "route": "maintenance", "route_reason": "details incomplete",
                    "node_timings": {"maintenance": time.perf_counter() - started}}

        pending.status = "awaiting_confirmation"
        return {"maintenance": pending.to_state(), "answer": pending.summary(),
                "route": "maintenance", "route_reason": "read back, awaiting confirmation",
                "node_timings": {"maintenance": time.perf_counter() - started}}

    def emergency(state: AgentState) -> dict:
        """Documented emergency instructions, verbatim. No model call, no improvising."""
        return {"answer": mx.EMERGENCY_REPLY, "node_timings": {"emergency": 0.0}}

    def refuse(state: AgentState) -> dict:
        """Fixed strings. No model call, so a refusal is instant and cannot drift."""
        route = state.get("route", "out_of_scope")
        return {"answer": prompts.REFUSALS.get(route, prompts.REFUSALS["out_of_scope"]),
                "node_timings": {"refuse": 0.0}}

    def record(state: AgentState) -> dict:
        """Commit the turn to the conversation. The only node that grows `messages`."""
        return {
            "messages": [
                HumanMessage(content=state.get("question", "")),
                AIMessage(content=state.get("answer", "")),
            ]
        }

    # --- wiring ------------------------------------------------------------

    graph = StateGraph(AgentState)
    for name, fn in (
        ("sanitize", sanitize), ("interpret", interpret), ("classify", classify),
        ("retrieve", retrieve), ("generate", generate), ("small_talk", small_talk),
        ("maintenance", maintenance), ("emergency", emergency),
        ("refuse", refuse), ("record", record),
    ):
        graph.add_node(name, fn)

    graph.add_edge(START, "sanitize")
    graph.add_edge("sanitize", "interpret")
    graph.add_edge("interpret", "classify")

    def after_classify(state: AgentState) -> str:
        route = state.get("route", "out_of_scope")
        if route == "property_question":
            return "retrieve"
        if route == "small_talk":
            return "small_talk"
        if route == "maintenance":
            return "maintenance"
        if route == "emergency":
            return "emergency"
        if route == "out_of_scope":
            # Go through retrieval anyway so a high-confidence hit can overrule the
            # router. Costs one embedding call. Only the router's opinion is appealable
            # this way -- injection, fabrication and uncovered-topic routes come from
            # deterministic checks and go straight to a refusal.
            return "retrieve"
        return "refuse"   # fabrication, injection, missing_info, needs_context

    graph.add_conditional_edges(
        "classify", after_classify,
        {"retrieve": "retrieve", "small_talk": "small_talk", "refuse": "refuse",
         "maintenance": "maintenance", "emergency": "emergency"},
    )

    def after_retrieve(state: AgentState) -> str:
        # retrieve may have flipped the route either way: down to missing_info via the
        # backstop, or up from out_of_scope on a high-confidence hit.
        return "generate" if state.get("route") == "property_question" else "refuse"

    graph.add_conditional_edges("retrieve", after_retrieve, {"refuse": "refuse", "generate": "generate"})

    for node in ("generate", "small_talk", "maintenance", "emergency", "refuse"):
        graph.add_edge(node, "record")
    graph.add_edge("record", END)

    return graph.compile(checkpointer=checkpointer)


def new_turn_id() -> str:
    return uuid.uuid4().hex[:12]
