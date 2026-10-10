# Rental FAQ Voice Assistant

**Personal prototype · fictional sample property.** A voice assistant that answers
questions about one rental property using only that property's documents, and that can
file a demo maintenance request through an n8n workflow after the caller explicitly
confirms it.

Maple Court Apartments does not exist. Every price, policy, phone number and address in
`data/rental/` is invented. There are no real tenants, no accounts, no availability feed,
and nothing here is deployed for a real company.

## In 30 seconds

A caller asks "what is the pet fee?" and hears a grounded answer. They ask "is that
refundable?" and the agent resolves "that" against the previous turn rather than
guessing. They ask about their account balance and it says the documents do not cover
it, instead of inventing a number. They report a leaking faucet and it collects the
unit, reads the request back, waits for an explicit yes, and only then files a ticket.

Two ways in, one workflow behind both. Vapi handles real phone audio and wants a single
finished answer. A browser gets the same answer token by token over a WebSocket.

```mermaid
flowchart TB
    subgraph clients [Clients]
        V[Vapi voice call]
        B[Browser or TypeScript client]
    end

    V -->|POST /chat/completions<br/>OpenAI wire format| API
    B <-->|WS /ws/chat<br/>token by token| API

    subgraph server [FastAPI]
        API[Endpoints<br/>shared secret, rate limit, frame cap]
    end

    API --> G

    subgraph graph [LangGraph workflow]
        direction TB
        S[sanitize<br/>injection, emergencies, pleasantries] --> I[interpret<br/>resolve follow ups]
        I --> C[classify<br/>route the turn]
        C -->|property question| R[retrieve<br/>cosine + BM25 over 45 chunks]
        R --> GEN[generate<br/>grounded, or no evidence]
        C -->|maintenance| M[collect, read back, await yes]
        C -->|out of scope| X[fixed refusal, no model call]
    end

    G[ ]::hidden --> S
    R -.reads.-> DOCS[(data/rental<br/>10 policy documents)]
    M -->|only after an explicit yes| N8N[n8n webhook<br/>returns a demo ticket]
    GEN --> CP[(SQLite checkpoint<br/>one thread per call)]
    M --> CP

    classDef hidden fill:none,stroke:none
```

---

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
cp .env.example .env                 # add your OPENAI_API_KEY
python scripts/rebuild.py            # check the corpus, build the index
uvicorn app.server:app --port 8000
```

Open <http://127.0.0.1:8000>. Without Vapi credentials the page runs in **text mode**,
which drives the identical pipeline over the WebSocket and shows the answer appearing
token by token, every behaviour below is demonstrable with no spend and no public URL.

### Example questions

| | |
|---|---|
| Supported | "What are the office hours?" · "How much is a two bedroom?" · "What is the pet fee?" · "Can I have a dog?" · "Where can my guests park?" |
| Follow-up | "What is the pet fee?" → **"Is that refundable?"** |
| Missing | "What is my balance?" · "Was my application approved?" · "Are there any units available?" |
| Unrelated | "What is the capital of France?" |
| Maintenance | "I want to report a leaking kitchen faucet" → "it's unit 2B" → "yes" |
| Emergency | "My kitchen is flooding" |

### Checks

```bash
pytest -q                                                       # 265 offline tests
python scripts/check_corpus.py                                  # corpus contradictions
python eval/run_retrieval_eval.py --hybrid --dense-weight 0.9   # Hit@k, Recall@k, MRR
python eval/run_grounding_eval.py --base http://127.0.0.1:8000  # 33 behavioural cases
python eval/run_retrieval_eval.py --sweep                       # the threshold evidence
```

---

## The knowledge base

`data/rental/` holds ten markdown documents for one property: contact and hours,
application, rent and fees, lease terms, pets, parking, amenities, maintenance, and
move-in/move-out. One file per policy area, so a retrieved chunk names the document it
came from, an answer about the pet fee is traceable to `pets.md`, not to "the corpus".

Chunking follows each document's own `##` sections, and the document title is embedded
with each chunk so "what time does the gym open" can reach a section headed only
"Hours and access". **45 chunks.** `00-about-this-corpus.md` is excluded from the index:
it is a note to the reader, not an answer.

`scripts/check_corpus.py` guards consistency. Figures stated in more than one document
(pet fee, late fee, notice period, …) must agree, or the agent will give two different
answers depending on which chunk wins retrieval, which looks exactly like a
hallucination. It currently reports no contradictions, and it does catch injected ones.

**Embedding model: `text-embedding-3-small`, 1536d**: unchanged from the previous
version of this project. 45 chunks is one batched call at startup and one ~60ms query
embedding per turn; nothing in the measurements asks for a larger model.

Retrieval is brute-force cosine in numpy fused with a small BM25 signal by weighted RRF.
No vector database: 45 vectors do not need one. The fusion weight was re-swept on this
corpus (`--sweep-weight`) and 0.9 beat both dense-only and the 0.8 tuned on the previous
corpus, a reminder that the constant belongs to the documents, not to the code.

---

## Scope and grounding

Five layers, because a prompt is a request, not a guarantee.

| # | Layer | Model call? | Catches |
|---|---|---|---|
| 1 | Pattern checks on the raw turn | no | injection, requests to change policy, topics the documents cannot cover, emergencies, greetings |
| 2 | Router classification | yes | unrelated questions, maintenance intent |
| 3 | Similarity backstop | no | obvious garbage |
| 4 | Grounded prompt + no-evidence sentinel | yes | topic retrieved well, specific detail absent |
| 5 | Post-generation topic check | no | an answer that strayed anyway |

Three outcomes: **supported** → answered from retrieved documents; **relevant but
missing** → *"I don't have that in the property documents."*; **unrelated** → *"I can
help with questions about Maple Court's rent, lease, pets, parking, amenities or
maintenance."*

An individual's balance, application status, lease dates, or unit availability route to
**missing**: they are about this property, the documents simply have no such data.

**Why layered, with evidence.** The similarity gate cannot be the scope mechanism: on
the eval set the weakest answerable question scores **0.272** and the strongest off-topic
one **0.354**. They overlap. So `MIN_SIMILARITY` is 0.23, low enough never to block a
real question, and layer 4 does the work. The router is not trusted alone either, if it
says out-of-scope but the top cosine is ≥ 0.45, the refusal is reversed.

Retrieved documents are neutralised and fenced as quoted evidence, so an instruction
written into a policy file reads as data. Refusals are fixed strings with no model call.

**None of this eliminates hallucination.** It narrows the paths to it and makes the
common ones cheap to catch. Layers 1 and 5 are keyword-driven and defeatable by
paraphrase.

---

## Conversation state

`AgentState` is a TypedDict. Two fields persist across turns: `messages` (via the
`add_messages` reducer) and `maintenance` (a pending request). **Everything else is
per-turn scratch that `sanitize` explicitly resets** - a bug in the previous version left
`route` persisting and silently disabled follow-up resolution after turn one.

**Checkpointer: `AsyncSqliteSaver`**, opened once in the FastAPI lifespan. One Vapi
`call.id` → one thread (`vapi-call:<id>`). Same call reuses it; separate calls are
isolated. No call id means a fresh random one, never a shared conversation.

Four distinctions worth stating plainly:

- **Checkpoints preserve conversation state.** They store what was said, keyed by call.
- **A call ID identifies a session, not an authenticated tenant.** Anyone can present any
  id. There is no tenant authentication here, and the assistant has no per-tenant data to
  protect, it reads one public document set.
- **Vapi handles speech capture and turn detection.** Deciding the caller stopped talking
  is Vapi's voice-activity detection, upstream of this server, which only ever receives
  completed turns. LangGraph state does not identify speakers.
- **SQLite and in-process locks are single-process.** Two workers would each keep their
  own session registry, so the per-call serialisation guarantee breaks. Moving to
  multiple workers means `AsyncPostgresSaver`, a Redis lock keyed on the call id, and
  Redis-backed dedupe.

Follow-ups work by rewriting a referential turn into a standalone one before retrieval:
*"What is the pet fee?"* → *"Is that refundable?"* becomes *"Is the pet fee refundable?"*,
which retrieves the pet-charges chunk and answers **"no, it is non-refundable"** rather
than answering about the security deposit.

---

## The maintenance workflow

```
"my kitchen faucet is leaking"   -> collecting            asks which unit
"it's unit 2B"                   -> awaiting_confirmation reads back, asks to confirm
"actually 3C"                    -> awaiting_confirmation updates, asks again
"yes"                            -> POST to n8n           -> submitted or failed
```

Three rules make this safe to demo:

1. **Submission requires an explicit confirmation**, matched by pattern on the caller's
   own words, not by a model's judgement. "maybe", "I think so" and "hold on" do not
   submit.
2. **Success is announced only after validation**: 2xx, parseable JSON, `status:
   "accepted"`, a `ticket_id`, and a `request_id` matching the one sent. Anything else, timeout, non-2xx, malformed body, mismatched id, produces *"I couldn't confirm that
   the request went through"* and a pointer to the office number.
3. **A timeout is never retried.** n8n may already have created the ticket.

`request_id` is minted once per pending request and is stable across corrections, so a
repeated "yes" or a duplicate delivery from Vapi submits once. The id is recorded in
`submitted_ids` after success; a later confirmation replies "already submitted that one".

**Emergencies bypass the workflow entirely.** Flooding, gas, fire, no heat, sewage, a
burst pipe or a broken exterior lock produce the documented emergency instructions, call 911 first if anyone is in danger, then the 24-hour line, with no model call and no
request queued. Emergency keyword matching is crude and will miss paraphrases; it is a
floor, not a ceiling.

**"Accepted" means the demo workflow returned a ticket. No property staff were notified
and nobody is dispatched.**

### Setting up n8n

**One command, locally.** This is the fastest way to prove the whole path:

```bash
docker compose -f docker-compose.n8n.yml up -d     # imports and activates the workflow
# put these in .env, then start the API
#   N8N_WEBHOOK_URL=http://localhost:5678/webhook/maintenance
#   N8N_WEBHOOK_SECRET=local-demo-secret
uvicorn app.server:app --port 8000
python scripts/file_demo_ticket.py
```

`file_demo_ticket.py` drives the real graph over the text endpoint: it reports a
leaking faucet, gives a unit, confirms, and prints the ticket n8n returned. It exits
non zero if nothing was accepted, so it works as a smoke check.

The compose file imports the workflow and activates it before n8n starts. Activation
matters: an inactive workflow makes the production webhook URL return 404, and n8n
refuses to activate once it is already running.

**Against your own n8n instance:**

1. **Import** `n8n/maintenance-workflow.json` (Workflows → Import from File). It contains
   no credentials.
2. **Set the shared secret inside n8n** as an environment variable named
   `PROTOTYPE_SECRET`. The Code node reads `$env.PROTOTYPE_SECRET` and rejects requests
   whose `X-Prototype-Secret` header does not match. On n8n Cloud, use Variables; on
   self-hosted, set the env var on the n8n process and restart it.
3. **Activate the workflow.** This matters: the **production** URL
   (`/webhook/maintenance`) only works while the workflow is active. The **test** URL
   (`/webhook-test/maintenance`) accepts a single request and only while you are watching
   the editor, it is not usable for anything but one manual check.
4. **Copy the production webhook URL** from the Webhook node and put it in `.env`:

```bash
N8N_WEBHOOK_URL=https://your-n8n-host/webhook/maintenance
N8N_WEBHOOK_SECRET=<the same value as PROTOTYPE_SECRET>
```

5. Restart the server. `GET /health` reports `n8n_configured: true`.

The URL and the secret live only in `.env`. They are never sent to the browser, never
logged, and `/health` reports only whether a workflow is configured, not where it is.

### Testing without n8n

`scripts/mock_n8n.py` mimics the workflow, including the failure modes:

```bash
python scripts/mock_n8n.py --port 5678            # or --mode timeout|error|malformed
N8N_WEBHOOK_URL=http://127.0.0.1:5678/webhook/maintenance \
N8N_WEBHOOK_SECRET=test-secret uvicorn app.server:app --port 8000
```

---

## WebSocket chat

`/ws/chat` is a second way into the same graph. It exists because the Vapi endpoint
cannot stream: Vapi wants one finished answer, so `/chat/completions` returns a single
SSE chunk. A browser can do better, and this does.

Every frame is a JSON object with a `type`.

**Client → server**

| Frame | Meaning |
|---|---|
| `{"type":"auth","secret":"...","resume":"ws-..."}` | First frame when `SERVER_SECRET` is set. `resume` is optional and reattaches to an earlier thread. |
| `{"type":"resume","call_id":"ws-..."}` | Reattach when no secret is configured, so there is no auth frame to carry it. Must come before the first turn. |
| `{"type":"user","text":"..."}` | One caller turn. |
| `{"type":"ping"}` | Optional keepalive. |

**Server → client**

| Frame | Meaning |
|---|---|
| `{"type":"ready","call_id":"ws-...","resumed":false}` | Auth accepted. This id is the LangGraph thread. `resumed` says whether an earlier one was reattached. |
| `{"type":"delta","text":"..."}` | A piece of the answer, as it is generated. |
| `{"type":"reset"}` | Discard what was streamed; a correction follows. |
| `{"type":"ticket","status":"submitted"\|"failed","ticket_id":...,"error":...}` | Outcome of a maintenance submission. |
| `{"type":"done","route":"...","latency_seconds":n}` | Turn complete. |
| `{"type":"error","message":"...","retry_after_seconds":n}` | Recoverable, the socket stays open. `retry_after_seconds` is present only on a rate limit refusal. |
| `{"type":"pong"}` | |

**Close code 1008** for a missing or wrong auth frame, and for any frame over 8 KiB.
The secret travels in the first frame, never in the URL, a query string lands in
browser history, in `Referer` headers and in every proxy log on the way.

Each connection gets its own `call_id`, so two sockets are two LangGraph threads with
the same isolation and checkpointing as two Vapi calls. Closing the socket drops the
session from the in-process registry. Checkpoint rows are left alone, which is what
makes reattaching possible.

### Limits and backpressure

| Limit | Value | Why |
|---|---|---|
| Frame size | 8 KiB | A hostile client should not be able to make the server buffer megabytes before any check runs. Closes 1008. |
| Turns per connection | 12 per 60 seconds | Every turn costs two or three model calls, so this is a spend guard first. Refused with an `error` frame, not a disconnect: someone typing quickly should be told to wait. |
| Concurrent connections | 32 | Turns on one socket are already strictly sequential, because the loop awaits each turn before reading the next frame. What that does not bound is the number of sockets. Over the cap, the server sends an `error` and closes 1013. |

Both counters live in the process. A second worker would need them in Redis alongside
the session registry, same as the per-call lock.

### Reconnecting

A dropped socket does not lose the conversation. The checkpoint survives, so a client
can reattach by passing the old `call_id`:

```jsonc
// with a secret configured
{"type": "auth", "secret": "...", "resume": "ws-1a2b3c4d5e6f"}
// without one
{"type": "resume", "call_id": "ws-1a2b3c4d5e6f"}
```

The server replies `ready` with `resumed: true` and continues the thread, including the
turn counter that drives duplicate detection. Only ids matching the `ws-` plus twelve
hex characters that this server issues are accepted, so a client cannot probe for or
invent thread names, and a resume is refused once the connection has taken a turn
because switching threads mid conversation would silently discard what was said.

**A resume is accepted on the caller's word alone.** With a single shared secret there
is nothing else to check it against: anyone holding the secret can reattach to any
thread. That is acceptable for one operator's prototype and is not multi-tenant.

### Try it

```bash
python - <<'PY'
import asyncio, json, websockets
async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/chat") as ws:
        # send an auth frame first if SERVER_SECRET is set
        print(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "user", "text": "What is the pet fee?"}))
        while (m := json.loads(await ws.recv()))["type"] != "done":
            if m["type"] == "delta":
                print(m["text"], end="", flush=True)
asyncio.run(main())
PY
```

Or just open the page and type, text mode uses this socket.

### What streams, and what does not

**Only the grounded-answer path streams token by token.** Refusals, emergency
instructions and maintenance replies are fixed or short strings produced with no model
call, so they arrive as a single delta. The `done` frame names the route, so a client
can tell which it got.

The first token is held back by 11 characters, the length of the no-evidence sentinel
the generator uses to say it found nothing. Without that holdback a refusal would
briefly render a control word to the caller.

**A streamed answer can still be retracted.** The post-generation checks run on the
finished text, so an answer that has already reached the client can be rejected. The
server then sends `reset` followed by the replacement. A client that ignores `reset`
will show a retracted answer.

### Measured latency

20 sequential requests, **loopback, one machine**: client and server on the same host
over 127.0.0.1:

| | P50 | P95 | min | max |
|---|---|---|---|---|
| Time to first delta | **1.08s** | 2.25s | 0.94s | 2.25s |
| Full response | **1.44s** | 2.66s | 1.23s | 2.66s |

Median 36 deltas per answer. With n=20, P95 is the 19th of 20 samples, an order
statistic, not a smooth estimate.

This measures the graph: a router call, an optional rewrite call, one embedding and
generation. It measures nothing about a network. A real user adds internet round trips;
a voice caller adds speech-to-text and text-to-speech on top. **These numbers are a
floor.** Reproduce with `python scripts/measure_ws_latency.py --n 20`.

---

## Connecting Vapi

Expose the server over HTTPS, then create the assistant:

```bash
cloudflared tunnel --url http://localhost:8000
export VAPI_PRIVATE_KEY=...          # dashboard.vapi.ai/org/api-keys
python scripts/create_assistant.py --url https://YOUR-TUNNEL.trycloudflare.com
```

Two settings matter:

- **`metadataSendMode` stays at its default.** Setting it to `off` strips the `call`
  object, and `call.id` is what maps a call to its LangGraph thread.
- **Voice is `vapi`/`Elliot`.** ElevenLabs and most other providers need your own
  provider key added first; without it Vapi rejects the assistant at call start.

**Set `SERVER_SECRET` before exposing anything.** Only `/health` and `/greeting` stay
public.

### On streaming, honestly

The response uses SSE framing with OpenAI `chat.completion.chunk` objects terminated by
`data: [DONE]`, which is what Vapi consumes. It is **not token-by-token streaming**: the
graph produces one finished answer, emitted as a single content chunk plus a stop chunk.
Real streaming would not help here, the route (answer, refusal, or confirmation prompt)
is only known once the node finishes, so there is nothing to stream before then.

"OpenAI-compatible" here means the **Chat Completions wire format**. That is unrelated to
an OpenAPI specification.

---

## Results

### Retrieval: 57 labeled questions (51 answerable, 6 off-topic)

Labels name the chunk(s) containing the answer, written by reading `data/chunks.json`.

| k | Hit@k | Recall@k | |
|---|---|---|---|
| 1 | 0.902 | 0.863 | |
| **3** | **0.980** | **0.980** | **the generator only sees these** |
| 5 | 0.980 | 0.980 | |

MRR **0.935**. Dense-only baseline: Hit@1 0.882, MRR 0.925.

- **Hit@k**: did at least one relevant chunk reach the top k.
- **Recall@k**: the share of *all* relevant chunks retrieved. Differs on the 4 questions
  with more than one valid source.
- **Score separation**: answerable mean top-1 cosine **0.482** (min 0.272) vs off-topic
  **0.156** (max 0.354).

### Grounding: 33 behavioural cases, 33/33 across three runs

11 answered · 6 missing · 4 unrelated · 4 fabrication · 3 injection · 1 clarification ·
2 emergency · 2 maintenance (collecting, awaiting confirmation).

**This is pattern matching, not semantic judgement**: no model grades the output. A
refusal is recognised by a regex covering the phrasings this agent produces; grounding is
asserted with `must_not_contain` lists of facts that would only appear if invented. The
maintenance cases assert on the request **status**, because "did not submit" is the claim
that matters and only the state proves it.

Its limits are real and were hit during development: the agent correctly declined with
"I have no evidence regarding the price of a three bedroom" and the regex scored it as a
failure until the pattern was widened.

### Latency (localhost, excludes speech-to-text and text-to-speech)

Answer 2.2s · with follow-up resolution 2.5s · maintenance intake 1.5s · router-decided
refusal 0.7s · **deterministic refusal 0.01s** (no model call, most of why layer 1
exists, and why an emergency answer is instant).

### 265 offline tests, no network

Chunking · scope controls · retrieval · graph routing and state · the maintenance state
machine and n8n client · sessions and concurrency · the Vapi HTTP contract · endpoint
authorisation · **34 WebSocket tests** (auth close codes, delta reassembly, socket
isolation, disconnect cleanup, oversized frames, maintenance submit, n8n failure, rate
limiting, the connection cap, and reattaching to a thread).
Plus `ruff` clean.

Every test is offline: no API key, no network. GitHub Actions runs `ruff` and `pytest`
on each push, so the count above is verifiable rather than claimed.

### What was verified, and how

| | |
|---|---|
| **Verified locally, mocked** | 265 tests; retrieval and grounding evals; the n8n client against `httpx.MockTransport` for success, timeout, non-2xx, malformed body and mismatched request id |
| **Verified locally, real HTTP** | The full collect → confirm → submit flow against `scripts/mock_n8n.py` over real HTTP, including duplicate confirmations creating exactly one ticket, corrections, declines, and all three failure modes |
| **Verified on the running server** | Rental FAQs, the "is that refundable?" follow-up, all refusal classes, session isolation between two call ids, the Vapi SSE contract with a real `call` object |
| **Verified against a real n8n** | A local n8n 1.68.0 started from `docker-compose.n8n.yml` imported and activated the committed workflow, and `scripts/file_demo_ticket.py` filed a ticket end to end through the graph: the production webhook returned `{"status":"accepted","ticket_id":"DEMO-M15H8D"}`, and a wrong shared secret was rejected with 401. This was a local container, not a hosted n8n, and no person was notified by it |
| **NOT verified** | **No live Vapi voice call** has been made against the rental assistant, the voice path is unchanged from a version that was verified on live calls, but it has not been re-tested since the adaptation |

---

## Limitations

1. **n8n was verified against a local container, not a hosted instance.** A real n8n
   accepted a request and returned a ticket, but it was `docker-compose.n8n.yml` on the
   same machine. Nothing is deployed, and a demo ticket means the workflow replied, not
   that anyone was notified.
2. **No live voice call since the adaptation.** The Vapi integration is structurally
   unchanged and its HTTP contract is tested, but it has not been spoken to.
3. **Thresholds are tuned on the set they are measured on**: 51 answerable questions, no
   holdout.
4. **Only 6 off-topic examples** behind the score-separation claim.
5. **Emergency detection is keyword-based.** It will miss a paraphrase. A caller
   describing a dangerous situation in unusual words gets the routine workflow.
6. **Unit numbers are not validated.** The assistant accepts whatever the caller says as
   a unit identifier; there is no unit list to check against.
7. **Duplicate protection is per-session and in-process.** Two workers would each keep
   their own `submitted_ids`. If n8n times out after processing, the caller is told it
   could not be confirmed while a ticket may exist, deliberately, since the alternative
   is a retry that definitely duplicates.
8. **Pattern-based scope layers are defeatable by paraphrase.**
9. **No checkpoint retention.** The SQLite file grows forever and holds transcripts.
10. **The corpus checker verifies repeated figures, not every claim.** It cannot prove the
    documents are complete or sensible.
11. **Single shared access key, not per-user auth.** The WebSocket uses the same one.
12. **A streamed answer can be retracted.** Post-generation checks run on the finished
    text, so the client may see tokens that are then withdrawn with a `reset`. A client
    that ignores `reset` will display a retracted answer.
13. **Only the grounded-answer path streams.** Refusals and maintenance replies arrive
    as a single delta, because they are produced without a model call.
14. **Latency numbers are loopback on one machine**: no network, no speech-to-text, no
    text-to-speech. They are a floor, not what a user would experience.
15. **Rate limits and the connection cap are per process.** 12 turns a minute per
    socket and 32 sockets total, counted in memory. A second worker would double both
    without noticing, the same way it would double the session registry.
16. **Reattaching trusts the caller.** Any client holding the shared secret can resume
    any thread it names. There is no per-user identity to check it against.
17. **The browser client does not reconnect automatically.** The protocol supports it
    and the server honours it, but the page still opens a fresh thread on the next
    message. The TypeScript client in `client/` is where reconnect is exercised.

---

## Layout

```
app/        config · chunking · embeddings · retriever · scope · prompts · graph ·
            maintenance (n8n client + confirmation state machine) · sessions · server ·
            ws (the /ws/chat protocol)
data/rental/  the ten sample property documents
data/       chunks.json (the indexed corpus)
eval/       labeled sets and actual measured results
n8n/        importable workflow (no credentials)
scripts/    check_corpus · rebuild · create_assistant · mock_n8n · measure_latency ·
            measure_ws_latency
web/        the browser call page
tests/      265 offline tests
```

### Adapting to a different property

```bash
# edit or replace the documents in data/rental/
python scripts/check_corpus.py     # catch contradictions first
python scripts/rebuild.py          # re-chunk, flag stale eval labels
```

Eval labels name specific chunk ids, so different documents need them rewritten by hand.
That is deliberate, auto-generated labels would make the metrics meaningless.
