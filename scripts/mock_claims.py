"""A stand-in for a claims intake endpoint, for testing without one.

PINEWRIGHT MUTUAL DOES NOT EXIST. This is a mock in a prototype: it checks a shared
secret, validates the required fields, mints a demo claim number and echoes the
request_id back. No insurer is contacted and no claim is opened anywhere.

    python scripts/mock_claims.py --port 5699
    CLAIMS_ENDPOINT_URL=http://127.0.0.1:5699/claims/intake \
    CLAIMS_ENDPOINT_SECRET=test-secret python eval/run_claims_eval.py --real

Modes for testing failure handling: --mode timeout | error | malformed | mismatch

Shaped deliberately like scripts/mock_n8n.py, including returning the same claim number
for a repeated request_id, which is what makes a double confirmation observably file once
over real HTTP rather than only in the state machine.
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
SEEN: dict[str, str] = {}   # request_id -> claim_id, to show idempotency

REQUIRED = ("request_id", "policy_number", "date_of_loss", "loss_type", "description", "source")


@app.post("/claims/intake")
async def intake(request: Request, x_prototype_secret: str | None = Header(default=None)):
    if SECRET and x_prototype_secret != SECRET:
        raise HTTPException(401, "bad secret")

    body = await request.json()
    for name in REQUIRED:
        if not body.get(name):
            raise HTTPException(400, f"missing {name}")
    if body.get("injured") is None:
        raise HTTPException(400, "missing injured")
    # A real intake would refuse an injury claim through this path; the prototype's
    # flow hands off before it gets here, and this mirrors that.
    if body.get("injured") is True:
        raise HTTPException(409, "injury claims are not accepted through automated intake")

    if MODE == "timeout":
        await asyncio.sleep(30)          # longer than the client timeout
    if MODE == "error":
        raise HTTPException(500, "simulated endpoint failure")
    if MODE == "malformed":
        return PlainTextResponse("not json at all")

    request_id = body["request_id"]
    claim = SEEN.setdefault(request_id, f"DEMO-C{uuid.uuid4().hex[:6].upper()}")
    echoed = "clm-does-not-match" if MODE == "mismatch" else request_id
    return JSONResponse({"status": "accepted", "claim_id": claim, "request_id": echoed})


@app.get("/seen")
async def seen() -> dict:
    return {"count": len(SEEN), "claims": SEEN}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5699)
    parser.add_argument(
        "--mode", default="ok", choices=["ok", "timeout", "error", "malformed", "mismatch"]
    )
    parser.add_argument("--secret", default="test-secret")
    args = parser.parse_args()
    MODE, SECRET = args.mode, args.secret
    print(f"mock claims endpoint on :{args.port}  mode={MODE}  (Pinewright Mutual is fictional)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
