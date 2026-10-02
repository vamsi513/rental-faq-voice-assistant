"""Runtime configuration. Everything tunable lives here so the demo has one place to look."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# The knowledge base. Every document in here describes one fictional property; see
# data/rental/00-about-this-corpus.md.
RENTAL_CORPUS_DIR = DATA_DIR / "rental"


@dataclass(frozen=True)
class Settings:
    # --- Generation ---------------------------------------------------------
    # gpt-4o-mini: the credential available for this demo, and the lowest-latency
    # competent option on that account. Voice is latency-bound -- the caller hears
    # silence while we think -- so a small fast model beats a stronger slow one here.
    # repr=False: keeps the key out of tracebacks, logs and /health output.
    openai_api_key: str = field(default=os.getenv("OPENAI_API_KEY", ""), repr=False)
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    router_model: str = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_answer_tokens: int = int(os.getenv("MAX_ANSWER_TOKENS", "160"))

    # --- Retrieval ---------------------------------------------------------
    # text-embedding-3-small, 1536 dimensions. Chosen because the corpus is ~15
    # chunks: the whole index is one batched API call at startup and a single
    # ~60ms query embedding per turn, with no extra dependency (a local
    # sentence-transformers model pulls ~2GB of torch for no measurable gain at
    # this size). See README "Embedding model" for the measured comparison.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = 1536
    top_k: int = int(os.getenv("TOP_K", "3"))
    # Dense cosine fused with a small BM25 signal by weighted RRF. Re-swept on the
    # rental corpus (--sweep-weight): 0.9 beat dense-only on Hit@1 (.882 -> .902) and
    # MRR (.925 -> .938) with Hit@3 level. The previous value of 0.8 was tuned on a
    # different corpus and is slightly worse here -- a reminder that this constant
    # belongs to the documents, not to the code.
    hybrid_retrieval: bool = os.getenv("HYBRID_RETRIEVAL", "true").lower() == "true"
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "0.9"))
    # A deliberately LOW backstop, not the scope decision. On the rental eval set the
    # weakest answerable question scores 0.272 and the strongest off-topic one 0.354 --
    # they OVERLAP, so no threshold separates them. 0.23 sits under every answerable
    # question (it never blocks a real one) while still rejecting ~83% of off-topic ones
    # for free. The router and the generator's no-evidence sentinel do the real work.
    min_similarity: float = float(os.getenv("MIN_SIMILARITY", "0.23"))
    # The other end of the same measurement. Off-topic questions reached 0.354 on the
    # rental eval set, so this sits clear of that: a score this high is strong evidence
    # the documents really do cover the question -- enough to overrule the router when
    # it calls something out of scope. The two signals check each other rather than the
    # router deciding alone.
    high_confidence_similarity: float = float(os.getenv("HIGH_CONFIDENCE_SIMILARITY", "0.45"))

    # --- Conversation / session -------------------------------------------
    # Bound on history replayed into the model. Keeps prompt size and latency flat
    # over a long call; older turns stay in the checkpoint but are not re-sent.
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "8"))
    # Sessions idle longer than this are dropped from the in-process lock/dedupe
    # registry. Checkpoint rows survive -- see README "Retention".
    session_idle_seconds: int = int(os.getenv("SESSION_IDLE_SECONDS", "1800"))
    checkpoint_db: str = os.getenv("CHECKPOINT_DB", str(DATA_DIR / "checkpoints.sqlite"))

    # --- Server ------------------------------------------------------------
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    # Optional shared secret. If set, Vapi must send it as `X-Vapi-Secret`.
    server_secret: str = os.getenv("SERVER_SECRET", "")
    debug_panel: bool = os.getenv("DEBUG_PANEL", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Vapi (browser demo only) -----------------------------------------
    # Served to the page so a reviewer can open one link and press Start call without
    # pasting anything. The PUBLIC key is designed to be exposed in client code; the
    # private key is never read here and must never be set in this file.
    vapi_public_key: str = os.getenv("VAPI_PUBLIC_KEY", "")
    vapi_assistant_id: str = os.getenv("VAPI_ASSISTANT_ID", "")

    # --- n8n maintenance workflow -----------------------------------------
    # The PRODUCTION webhook URL from the n8n Webhook node. n8n's *test* URL only
    # accepts one request while the editor is listening, so it is useless here.
    n8n_webhook_url: str = os.getenv("N8N_WEBHOOK_URL", "")
    # Shared credential sent as a header. Not a user credential and not authentication
    # of any person -- it only proves the caller is this backend.
    n8n_webhook_secret: str = field(default=os.getenv("N8N_WEBHOOK_SECRET", ""), repr=False)
    n8n_header_name: str = os.getenv("N8N_HEADER_NAME", "X-Prototype-Secret")
    # A voice caller is waiting, so this is short. A timeout is NOT retried: n8n may
    # already have created the ticket, and a blind retry would duplicate it.
    n8n_timeout_seconds: float = float(os.getenv("N8N_TIMEOUT_SECONDS", "8"))

    @property
    def n8n_configured(self) -> bool:
        return bool(self.n8n_webhook_url)


settings = Settings()

GREETING = (
    "Hi, this is the Maple Court Apartments sample assistant. "
    "I can answer questions about rent, pets, parking, amenities, or maintenance. "
    "This is a demo property, so I can't look up anyone's account."
)

# Shown in the UI and stated by the assistant when asked what it is.
PROTOTYPE_NOTICE = "Personal prototype - fictional sample property"
PROPERTY_NAME = "Maple Court Apartments"

# Exact phrasing the agent uses for the two refusal classes. Tests assert on
# meaning rather than these strings, but keeping them here stops them drifting.
MISSING_INFO_REPLY = "I don't have that in the property documents."
OUT_OF_SCOPE_REPLY = (
    "I can help with questions about Maple Court's rent, lease, pets, parking, "
    "amenities or maintenance."
)
