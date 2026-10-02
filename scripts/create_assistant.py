"""Create (or update) the Vapi assistant that points at this server.

Doing it through the API rather than the dashboard means the whole configuration is in
version control and reviewable, instead of being a set of form fields someone clicked.

    export VAPI_PRIVATE_KEY=...                       # from dashboard.vapi.ai/org/api-keys
    python scripts/create_assistant.py --url https://your-tunnel.trycloudflare.com

Prints the assistant id. Pass --update <id> to change an existing one instead.
The private key is read from the environment only -- never pass it as an argument,
where it would land in your shell history.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import GREETING, settings  # noqa: E402

API = "https://api.vapi.ai/assistant"


def build(url: str, secret: str, voice_provider: str = "vapi", voice_id: str = "Elliot") -> dict:
    config: dict = {
        "name": "Maple Court sample FAQ assistant (prototype)",
        # Spoken before the caller says anything. Vapi says this itself, so it costs no
        # round trip to our server.
        "firstMessage": GREETING,
        "model": {
            # The whole integration: Vapi POSTs OpenAI-shaped chat completions to our
            # server and speaks whatever we stream back.
            "provider": "custom-llm",
            # Vapi appends /chat/completions itself. Our server also accepts
            # /v1/chat/completions, so either convention works.
            "url": url.rstrip("/"),
            "model": settings.chat_model,
            # metadataSendMode is deliberately NOT set to "off": leaving it at the
            # default is what makes Vapi include the `call` object, and `call.id` is
            # what maps this call to its LangGraph thread.
        },
        "transcriber": {"provider": "deepgram", "model": "nova-2", "language": "en"},
        # Vapi's own built-in voice. ElevenLabs and most other TTS providers need you to
        # add YOUR OWN provider key in the dashboard first; without it Vapi rejects the
        # assistant when the call starts, and the browser never even reaches the
        # microphone prompt. "vapi"/"Elliot" needs no third-party key, so it works on a
        # fresh account. Override with --voice once you've added a provider key.
        "voice": {"provider": voice_provider, "voiceId": voice_id},
        # Turn-taking and interruption handling are Vapi's, not ours.
        "startSpeakingPlan": {"waitSeconds": 0.4},
        "stopSpeakingPlan": {"numWords": 2},
        # Cost guards: the call cannot run away if someone walks off mid-demo.
        "silenceTimeoutSeconds": 20,
        "maxDurationSeconds": 600,
    }
    if secret:
        # Sent on every request so a stranger who finds the tunnel URL cannot run turns
        # on your OpenAI key.
        config["model"]["headers"] = {"X-Vapi-Secret": secret}
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="public HTTPS base URL of this server")
    parser.add_argument("--update", metavar="ASSISTANT_ID", help="update instead of create")
    parser.add_argument("--dry-run", action="store_true", help="print the config and stop")
    parser.add_argument("--voice-provider", default="vapi",
                        help="TTS provider (default: vapi, the only one needing no provider key)")
    parser.add_argument("--voice-id", default="Elliot", help="voice id (default: Elliot)")
    args = parser.parse_args()

    if not args.url.startswith("https://"):
        sys.exit("--url must be https:// -- Vapi will not call a plain http endpoint.")

    config = build(args.url, settings.server_secret, args.voice_provider, args.voice_id)

    if args.dry_run:
        # Redact the shared secret: a dry run is for pasting into a chat or a ticket.
        shown = json.loads(json.dumps(config))
        if shown["model"].get("headers", {}).get("X-Vapi-Secret"):
            shown["model"]["headers"]["X-Vapi-Secret"] = "<SERVER_SECRET from .env>"
        print(json.dumps(shown, indent=2))
        return 0

    key = os.getenv("VAPI_PRIVATE_KEY", "").strip()
    if not key:
        sys.exit("VAPI_PRIVATE_KEY is not set.\n"
                 "  export VAPI_PRIVATE_KEY=...   (dashboard.vapi.ai/org/api-keys, PRIVATE key)")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=30) as client:
        if args.update:
            response = client.patch(f"{API}/{args.update}", headers=headers, json=config)
        else:
            response = client.post(API, headers=headers, json=config)

    if response.status_code >= 300:
        print(f"Vapi returned {response.status_code}:\n{response.text}")
        return 1

    assistant = response.json()
    print(f"assistant id : {assistant['id']}")
    print(f"custom LLM   : {config['model']['url']}  (Vapi appends /chat/completions)")
    print(f"shared secret: {'sent as X-Vapi-Secret' if settings.server_secret else 'NONE — set SERVER_SECRET in .env'}")
    print("\nOpen the demo page with:")
    print(f"  http://127.0.0.1:{settings.port}/?key=<YOUR_PUBLIC_KEY>&assistant={assistant['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
