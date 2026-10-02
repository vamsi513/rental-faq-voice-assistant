"""A stand-in for the n8n webhook, for testing without an n8n instance.

It mimics what the real workflow does: check the shared secret, validate the required
fields, mint a demo ticket, and echo the request_id back. Run it, point N8N_WEBHOOK_URL
at it, and the whole submit path is exercised for real over HTTP.

    python scripts/mock_n8n.py --port 5678
    N8N_WEBHOOK_URL=http://127.0.0.1:5678/webhook/maintenance \
    N8N_WEBHOOK_SECRET=test-secret uvicorn app.server:app --port 8000

Modes for testing failure handling: --mode timeout | error | malformed
"""
from __future__ import annotations

import argparse
import asyncio
import uuid

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI()
MODE = "ok"
SECRET = "test-secret"
SEEN: dict[str, str] = {}   # request_id -> ticket_id, to show idempotency


@app.post("/webhook/maintenance")
async def maintenance(request: Request, x_prototype_secret: str | None = Header(default=None)):
    if SECRET and x_prototype_secret != SECRET:
        raise HTTPException(401, "bad secret")

    body = await request.json()
    for field in ("request_id", "call_id", "unit", "issue", "source"):
        if not body.get(field):
            raise HTTPException(400, f"missing {field}")

    if MODE == "timeout":
        await asyncio.sleep(30)          # longer than the client timeout
    if MODE == "error":
        raise HTTPException(500, "simulated n8n failure")
    if MODE == "malformed":
        return PlainTextResponse("not json at all")

    request_id = body["request_id"]
    # Same request_id returns the same ticket, which is what the real workflow should do.
    ticket = SEEN.setdefault(request_id, f"DEMO-{uuid.uuid4().hex[:6].upper()}")
    return JSONResponse({"status": "accepted", "ticket_id": ticket, "request_id": request_id})


@app.get("/seen")
async def seen() -> dict:
    return {"count": len(SEEN), "tickets": SEEN}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--mode", default="ok", choices=["ok", "timeout", "error", "malformed"])
    parser.add_argument("--secret", default="test-secret")
    args = parser.parse_args()
    MODE, SECRET = args.mode, args.secret
    print(f"mock n8n on :{args.port}  mode={MODE}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
