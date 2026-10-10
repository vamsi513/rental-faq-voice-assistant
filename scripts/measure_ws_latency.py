"""Measure WebSocket latency: time to first delta, and full response time.

LOOPBACK, ONE MACHINE. Client and server are the same host over 127.0.0.1, so this
measures the graph (router call, optional rewrite, embedding, generation) and nothing
about a network. A real user adds internet round trips; a voice caller adds
speech-to-text and text-to-speech on top of that. These numbers are a floor.

Run: python scripts/measure_ws_latency.py --n 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time

import websockets

QUESTIONS = [
    "What are the office hours?",
    "How much is a two bedroom?",
    "What is the pet fee?",
    "Can I have a dog?",
    "Where can my guests park?",
    "When is rent due?",
    "What documents do I need to apply?",
    "How long is the lease?",
    "What time does the gym open?",
    "How much notice before moving out?",
]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. With n=20 this is an order statistic, not a smooth
    estimate -- P95 is simply the 19th of 20 samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


async def one(url: str, secret: str, question: str) -> tuple[float, float, int]:
    async with websockets.connect(url, max_size=2**20) as ws:
        if secret:
            await ws.send(json.dumps({"type": "auth", "secret": secret}))
        await ws.recv()                                  # ready
        start = time.perf_counter()
        await ws.send(json.dumps({"type": "user", "text": question}))
        first, deltas = None, 0
        while True:
            message = json.loads(await ws.recv())
            if message["type"] == "delta":
                if first is None:
                    first = time.perf_counter() - start
                deltas += 1
            elif message["type"] == "done":
                return (first or 0.0), time.perf_counter() - start, deltas


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/chat")
    parser.add_argument("--secret", default=os.getenv("SERVER_SECRET", ""))
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    firsts: list[float] = []
    totals: list[float] = []
    deltas: list[int] = []

    print(f"{args.n} requests, sequential, loopback\n")
    for i in range(args.n):
        question = QUESTIONS[i % len(QUESTIONS)]
        try:
            first, total, count = await one(args.url, args.secret, question)
        except Exception as exc:
            print(f"  request {i + 1} failed: {exc}")
            return 1
        firsts.append(first)
        totals.append(total)
        deltas.append(count)
        print(f"  {i + 1:>2}. first {first:5.2f}s  total {total:5.2f}s  "
              f"{count:>3} deltas  {question[:38]}")

    print(f"\n{'':22}{'P50':>8}{'P95':>8}{'min':>8}{'max':>8}")
    for label, series in (("time to first delta", firsts), ("full response", totals)):
        print(f"  {label:<20}{percentile(series, 50):>7.2f}s{percentile(series, 95):>7.2f}s"
              f"{min(series):>7.2f}s{max(series):>7.2f}s")
    print(f"\n  median deltas per answer: {statistics.median(deltas):.0f}")
    print("\n  LOOPBACK, ONE MACHINE. No network, no speech-to-text, no text-to-speech.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
