/**
 * The client is driven against a real in-process WebSocket server, not a stubbed
 * socket object. That way the framing, the JSON and the close codes are all
 * genuinely exercised; only the model behind the server is scripted.
 */
import type { AddressInfo } from "node:net";

import { WebSocketServer, type WebSocket as WS } from "ws";
import { afterEach, describe, expect, it } from "vitest";

import {
  ConnectionClosedError,
  RateLimitedError,
  RentalChatClient,
} from "../src/client.js";
import { parseServerEvent } from "../src/protocol.js";

type Handler = (socket: WS, frame: Record<string, unknown>) => void;

interface Harness {
  url: string;
  close: () => Promise<void>;
  received: Record<string, unknown>[];
}

const harnesses: Harness[] = [];

/** A mock server that mirrors app/ws.py's protocol closely enough to test against. */
async function serve(handler: Handler, onConnect?: (socket: WS) => void): Promise<Harness> {
  const wss = new WebSocketServer({ port: 0 });
  const received: Record<string, unknown>[] = [];
  wss.on("connection", (socket) => {
    onConnect?.(socket);
    socket.on("message", (data: Buffer) => {
      const frame = JSON.parse(data.toString()) as Record<string, unknown>;
      received.push(frame);
      handler(socket, frame);
    });
  });
  await new Promise<void>((resolve) => wss.once("listening", resolve));
  const { port } = wss.address() as AddressInfo;
  const harness: Harness = {
    url: `ws://127.0.0.1:${port}`,
    received,
    close: () =>
      new Promise<void>((resolve) => {
        wss.clients.forEach((c) => c.terminate());
        wss.close(() => resolve());
      }),
  };
  harnesses.push(harness);
  return harness;
}

function send(socket: WS, payload: unknown): void {
  socket.send(JSON.stringify(payload));
}

/** The common case: auth, then stream an answer word by word. */
function streamingServer(answer: string, route = "property_question"): Handler {
  return (socket, frame) => {
    if (frame["type"] === "auth") {
      send(socket, { type: "ready", call_id: "ws-aaaabbbbcccc", resumed: Boolean(frame["resume"]) });
      return;
    }
    if (frame["type"] === "user") {
      for (const word of answer.split(" ")) send(socket, { type: "delta", text: `${word} ` });
      send(socket, { type: "done", route, latency_seconds: 1.23 });
    }
  };
}

afterEach(async () => {
  while (harnesses.length) await harnesses.pop()?.close();
});

describe("protocol parsing", () => {
  it("rejects anything that is not a known event", () => {
    expect(parseServerEvent("not json")).toBeNull();
    expect(parseServerEvent("[]")).toBeNull();
    expect(parseServerEvent('"a string"')).toBeNull();
    expect(parseServerEvent('{"type":"unknown_future_frame"}')).toBeNull();
  });

  it("accepts every documented event", () => {
    expect(parseServerEvent('{"type":"delta","text":"hi"}')?.type).toBe("delta");
    expect(parseServerEvent('{"type":"reset"}')?.type).toBe("reset");
    expect(parseServerEvent('{"type":"done","route":"x","latency_seconds":1}')?.type).toBe("done");
  });
});

describe("connecting", () => {
  it("authenticates and reports the thread id", async () => {
    const h = await serve(streamingServer("the pet fee is two fifty"));
    const client = new RentalChatClient({ url: h.url, secret: "s3cret" });
    await client.connect();
    expect(client.callId).toBe("ws-aaaabbbbcccc");
    expect(h.received[0]).toEqual({ type: "auth", secret: "s3cret" });
    client.close();
  });

  it("sends no auth frame when no secret is configured", async () => {
    // app/ws.py greets immediately when SERVER_SECRET is unset and never waits for
    // an auth frame. A client that sent one anyway would be sending an unknown
    // frame; a client that waited for one would hang.
    const h = await serve(
      (socket, frame) => {
        if (frame["type"] === "user") {
          send(socket, { type: "delta", text: "hi there" });
          send(socket, { type: "done", route: "small_talk", latency_seconds: 0.1 });
        }
      },
      (socket) => send(socket, { type: "ready", call_id: "ws-ddddeeeeffff" }),
    );
    const client = new RentalChatClient({ url: h.url });
    await client.connect();
    expect(client.callId).toBe("ws-ddddeeeeffff");
    const result = await client.ask("hello");
    expect(result.text).toBe("hi there");
    // Only the user turn was sent; no auth frame.
    expect(h.received.map((f) => f["type"])).toEqual(["user"]);
    client.close();
  });

  it("reattaches with a resume frame when there is no secret", async () => {
    const h = await serve(
      (socket, frame) => {
        if (frame["type"] === "resume") {
          send(socket, {
            type: "ready",
            call_id: frame["call_id"],
            resumed: true,
            turns: 3,
          });
        }
      },
      (socket) => send(socket, { type: "ready", call_id: "ws-freshfresh00" }),
    );
    const client = new RentalChatClient({ url: h.url, resume: "ws-0123456789ab" });
    await client.connect();
    expect(h.received[0]).toEqual({ type: "resume", call_id: "ws-0123456789ab" });
    expect(client.callId).toBe("ws-0123456789ab");
    expect(client.resumed).toBe(true);
    client.close();
  });

  it("surfaces a 1008 rejection as a closed connection", async () => {
    const h = await serve((socket) => socket.close(1008, "bad secret"));
    const client = new RentalChatClient({ url: h.url, secret: "wrong" });
    await expect(client.connect()).rejects.toBeInstanceOf(ConnectionClosedError);
  });

  it("passes resume on the auth frame", async () => {
    const h = await serve(streamingServer("resumed answer"));
    const client = new RentalChatClient({
      url: h.url,
      secret: "s3cret",
      resume: "ws-0123456789ab",
    });
    await client.connect();
    expect(h.received[0]).toEqual({
      type: "auth",
      secret: "s3cret",
      resume: "ws-0123456789ab",
    });
    expect(client.resumed).toBe(true);
    client.close();
  });
});

describe("streaming a turn", () => {
  it("reassembles deltas and reports the route", async () => {
    const h = await serve(streamingServer("the pet fee is two fifty"));
    const client = new RentalChatClient({ url: h.url, secret: "s" });
    await client.connect();
    const pieces: string[] = [];
    const result = await client.ask("what is the pet fee?", (p) => pieces.push(p));
    expect(result.text.trim()).toBe("the pet fee is two fifty");
    expect(pieces.length).toBeGreaterThan(1);
    expect(result.route).toBe("property_question");
    expect(result.retracted).toBe(false);
    client.close();
  });

  it("discards streamed text on reset and keeps only the correction", async () => {
    // The case a naive client gets wrong: tokens already shown are withdrawn.
    const h = await serve((socket, frame) => {
      if (frame["type"] === "auth") {
        send(socket, { type: "ready", call_id: "ws-aaaabbbbcccc" });
        return;
      }
      if (frame["type"] === "user") {
        send(socket, { type: "delta", text: "yes we have units " });
        send(socket, { type: "delta", text: "available now" });
        send(socket, { type: "reset" });
        send(socket, { type: "delta", text: "I do not have that in the documents." });
        send(socket, { type: "done", route: "missing_info", latency_seconds: 2 });
      }
    });
    const client = new RentalChatClient({ url: h.url, secret: "s" });
    await client.connect();
    let sawRetraction = false;
    const result = await client.ask("any units available?", (_p, retracted) => {
      if (retracted) sawRetraction = true;
    });
    expect(sawRetraction).toBe(true);
    expect(result.retracted).toBe(true);
    expect(result.text).toBe("I do not have that in the documents.");
    expect(result.text).not.toContain("available now");
    client.close();
  });

  it("surfaces a maintenance ticket", async () => {
    const h = await serve((socket, frame) => {
      if (frame["type"] === "auth") {
        send(socket, { type: "ready", call_id: "ws-aaaabbbbcccc" });
        return;
      }
      if (frame["type"] === "user") {
        send(socket, { type: "delta", text: "Submitted." });
        send(socket, {
          type: "ticket",
          status: "submitted",
          ticket_id: "DEMO-123",
          error: null,
        });
        send(socket, { type: "done", route: "maintenance", latency_seconds: 1 });
      }
    });
    const client = new RentalChatClient({ url: h.url, secret: "s" });
    await client.connect();
    const result = await client.ask("yes");
    expect(result.ticket?.status).toBe("submitted");
    expect(result.ticket?.ticket_id).toBe("DEMO-123");
    client.close();
  });

  it("reports a failed submission without inventing success", async () => {
    const h = await serve((socket, frame) => {
      if (frame["type"] === "auth") {
        send(socket, { type: "ready", call_id: "ws-aaaabbbbcccc" });
        return;
      }
      if (frame["type"] === "user") {
        send(socket, { type: "delta", text: "I could not confirm that." });
        send(socket, { type: "ticket", status: "failed", ticket_id: null, error: "timeout" });
        send(socket, { type: "done", route: "maintenance", latency_seconds: 8 });
      }
    });
    const client = new RentalChatClient({ url: h.url, secret: "s" });
    await client.connect();
    const result = await client.ask("yes");
    expect(result.ticket?.status).toBe("failed");
    expect(result.ticket?.ticket_id).toBeNull();
    expect(result.text).not.toContain("DEMO");
    client.close();
  });
});

describe("failure handling", () => {
  it("raises RateLimitedError with the wait, leaving the socket usable", async () => {
    const h = await serve((socket, frame) => {
      if (frame["type"] === "auth") {
        send(socket, { type: "ready", call_id: "ws-aaaabbbbcccc" });
        return;
      }
      if (frame["type"] === "user") {
        send(socket, {
          type: "error",
          message: "rate limit reached, try again in 12.5 seconds",
          retry_after_seconds: 12.5,
        });
        return;
      }
      if (frame["type"] === "ping") send(socket, { type: "pong" });
    });
    const client = new RentalChatClient({ url: h.url, secret: "s" });
    await client.connect();
    await expect(client.ask("hello")).rejects.toBeInstanceOf(RateLimitedError);
    // Not a disconnect: the socket still answers.
    await expect(client.ping()).resolves.toBeUndefined();
    client.close();
  });

  it("raises a plain error for a non rate limit error frame", async () => {
    const h = await serve((socket, frame) => {
      if (frame["type"] === "auth") {
        send(socket, { type: "ready", call_id: "ws-aaaabbbbcccc" });
        return;
      }
      if (frame["type"] === "user") send(socket, { type: "error", message: "empty text" });
    });
    const client = new RentalChatClient({ url: h.url, secret: "s" });
    await client.connect();
    await expect(client.ask("x")).rejects.toThrow("empty text");
    client.close();
  });

  it("rejects an in-flight turn when the server disappears mid stream", async () => {
    const h = await serve((socket, frame) => {
      if (frame["type"] === "auth") {
        send(socket, { type: "ready", call_id: "ws-aaaabbbbcccc" });
        return;
      }
      if (frame["type"] === "user") {
        send(socket, { type: "delta", text: "half an ans" });
        socket.close(1011, "boom");
      }
    });
    const client = new RentalChatClient({ url: h.url, secret: "s" });
    await client.connect();
    await expect(client.ask("hello")).rejects.toBeInstanceOf(ConnectionClosedError);
  });

  it("ignores frame types it does not know", async () => {
    const h = await serve((socket, frame) => {
      if (frame["type"] === "auth") {
        send(socket, { type: "ready", call_id: "ws-aaaabbbbcccc" });
        return;
      }
      if (frame["type"] === "user") {
        send(socket, { type: "some_future_frame", payload: 1 });
        send(socket, { type: "delta", text: "still fine" });
        send(socket, { type: "done", route: "property_question", latency_seconds: 1 });
      }
    });
    const client = new RentalChatClient({ url: h.url, secret: "s" });
    await client.connect();
    const result = await client.ask("hello");
    expect(result.text).toBe("still fine");
    client.close();
  });
});
