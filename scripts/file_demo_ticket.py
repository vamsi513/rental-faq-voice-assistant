"""File one maintenance ticket end to end and print what came back.

Proves the whole path without a voice call or a browser: it drives the real graph
over the HTTP text endpoint, collects the unit and issue, confirms, and shows the
ticket n8n returned.

    docker compose -f docker-compose.n8n.yml up -d      # or: python scripts/mock_n8n.py
    uvicorn app.server:app --port 8000
    python scripts/file_demo_ticket.py

Exits non zero if no ticket was accepted, so it is usable as a smoke check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

import httpx

TURNS = [
    "I want to report a leaking kitchen faucet",
    "it's unit 2B",
    "yes",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--secret", default=os.getenv("SERVER_SECRET", ""))
    args = parser.parse_args()

    headers = {"X-Vapi-Secret": args.secret} if args.secret else {}
    call_id = f"demo-{uuid.uuid4().hex[:8]}"

    with httpx.Client(base_url=args.base, timeout=60, headers=headers) as client:
        try:
            health = client.get("/health").json()
        except Exception as exc:
            print(f"cannot reach {args.base}: {exc}")
            print("start the API first:  uvicorn app.server:app --port 8000")
            return 1
        if not health.get("ready"):
            print("the API is up but not ready; is OPENAI_API_KEY set?")
            return 1
        if not health.get("n8n_configured"):
            print("N8N_WEBHOOK_URL is not set, so there is nothing to submit to.")
            print("Start n8n:  docker compose -f docker-compose.n8n.yml up -d")
            print("Then set N8N_WEBHOOK_URL and N8N_WEBHOOK_SECRET in .env and restart the API.")
            return 1

        print(f"call_id: {call_id}\n")
        payload: dict = {}
        for turn in TURNS:
            print(f"  caller    {turn}")
            payload = client.post(
                "/text", json={"call_id": call_id, "message": turn}
            ).json()
            print(f"  assistant {payload['answer']}\n")

    maintenance = payload.get("maintenance") or {}
    status = maintenance.get("status")
    print(json.dumps(maintenance, indent=2))

    if status == "submitted":
        print(f"\nAccepted. Demo ticket {maintenance.get('ticket_id')}.")
        print("That means the workflow returned a validated response. It does not mean")
        print("anyone was notified: there is no property staff and no dispatch.")
        return 0

    print(f"\nNot submitted. Status {status!r}, error {maintenance.get('error')!r}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
