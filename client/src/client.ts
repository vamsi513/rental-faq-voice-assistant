/**
 * A client for the rental assistant's /ws/chat endpoint.
 *
 * Transport agnostic on purpose: it takes a factory that returns something shaped
 * like a WebSocket, so the tests drive it against a real in-process server without
 * the client knowing the difference.
 *
 * What it handles that a naive client would not:
 *   - `reset`, which retracts an answer that was already streamed. A client that
 *     ignores it shows the caller a withdrawn answer.
 *   - rate limit refusals, which arrive as an `error` frame and leave the socket
 *     open, so the right response is to wait rather than reconnect.
 *   - reattaching to a thread after a drop, so a reconnect does not lose context.
 */
import WebSocket from "ws";

import {
  type ClientFrame,
  type ServerEvent,
  type TicketEvent,
  parseServerEvent,
} from "./protocol.js";

export interface ClientOptions {
  url: string;
  /** Required only when the server has SERVER_SECRET set. */
  secret?: string;
  /** Reattach to this thread instead of starting a new one. */
  resume?: string;
  /** Injected by tests. Defaults to a real ws connection. */
  createSocket?: (url: string) => WebSocketLike;
  /** How long to wait for `ready` before giving up. */
  readyTimeoutMs?: number;
}

/** The slice of the WebSocket API this client uses. */
export interface WebSocketLike {
  send(data: string): void;
  close(code?: number, reason?: string): void;
  on(event: "open", listener: () => void): void;
  on(event: "message", listener: (data: unknown) => void): void;
  on(event: "close", listener: (code: number, reason: unknown) => void): void;
  on(event: "error", listener: (err: Error) => void): void;
}

export interface TurnResult {
  /** The answer as the caller should finally see it, after any retraction. */
  text: string;
  route: string;
  latencySeconds: number;
  /** True when a `reset` arrived, meaning earlier tokens were withdrawn. */
  retracted: boolean;
  ticket: TicketEvent | null;
}

export class RateLimitedError extends Error {
  constructor(
    message: string,
    readonly retryAfterSeconds: number,
  ) {
    super(message);
    this.name = "RateLimitedError";
  }
}

export class ConnectionClosedError extends Error {
  constructor(
    message: string,
    readonly code: number,
  ) {
    super(message);
    this.name = "ConnectionClosedError";
  }
}

type Waiter = {
  resolve: (event: ServerEvent) => void;
  reject: (err: Error) => void;
};

export class RentalChatClient {
  private socket: WebSocketLike | null = null;
  private readonly queue: ServerEvent[] = [];
  private readonly waiters: Waiter[] = [];
  private closed: { code: number } | null = null;
  private readyState: { callId: string; resumed: boolean } | null = null;

  constructor(private readonly options: ClientOptions) {}

  /** The thread id this connection is attached to. Pass it to `resume` later. */
  get callId(): string | null {
    return this.readyState?.callId ?? null;
  }

  get resumed(): boolean {
    return this.readyState?.resumed ?? false;
  }

  async connect(): Promise<void> {
    const create = this.options.createSocket ?? defaultSocket;
    const socket = create(this.options.url);
    this.socket = socket;

    // Attach the frame handlers BEFORE awaiting open. The server greets immediately
    // when it has no secret configured, and a listener attached after the open
    // callback can miss that first frame entirely, which hangs the connect.
    socket.on("message", (data: unknown) => {
      const event = parseServerEvent(String(data));
      // Unrecognised frames are dropped, not thrown: a server that gains a frame
      // type should not break an older client mid conversation.
      if (event) this.deliver(event);
    });
    socket.on("close", (code: number) => {
      this.closed = { code };
      const err = new ConnectionClosedError(`socket closed (${code})`, code);
      while (this.waiters.length) this.waiters.shift()?.reject(err);
    });

    await new Promise<void>((resolve, reject) => {
      socket.on("open", resolve);
      socket.on("error", reject);
    });

    // With no secret the server sends `ready` immediately and expects no auth
    // frame, so sending one would be an unknown frame. Resume gets its own frame.
    if (this.options.secret) {
      const auth: ClientFrame = this.options.resume
        ? { type: "auth", secret: this.options.secret, resume: this.options.resume }
        : { type: "auth", secret: this.options.secret };
      this.send(auth);
    }

    const ready = await this.next(this.options.readyTimeoutMs ?? 10_000);
    if (ready.type !== "ready") {
      throw new Error(`expected ready, got ${ready.type}`);
    }
    this.readyState = { callId: ready.call_id, resumed: ready.resumed ?? false };

    if (!this.options.secret && this.options.resume) {
      this.send({ type: "resume", call_id: this.options.resume });
      const resumed = await this.next();
      if (resumed.type === "ready") {
        this.readyState = { callId: resumed.call_id, resumed: resumed.resumed ?? false };
      }
    }
  }

  /**
   * Send one turn and collect the reply.
   *
   * `onDelta` fires per token so a caller can render progressively. It is also
   * called with the empty-string signal on a reset, so a terminal UI knows to clear.
   */
  async ask(text: string, onDelta?: (piece: string, retracted: boolean) => void): Promise<TurnResult> {
    this.send({ type: "user", text });

    let buffer = "";
    let retracted = false;
    let ticket: TicketEvent | null = null;

    for (;;) {
      const event = await this.next();
      switch (event.type) {
        case "delta":
          buffer += event.text;
          onDelta?.(event.text, false);
          break;
        case "reset":
          // The server withdrew what it streamed. Everything before this is void.
          buffer = "";
          retracted = true;
          onDelta?.("", true);
          break;
        case "ticket":
          ticket = event;
          break;
        case "done":
          return {
            text: buffer,
            route: event.route,
            latencySeconds: event.latency_seconds,
            retracted,
            ticket,
          };
        case "error":
          if (typeof event.retry_after_seconds === "number") {
            throw new RateLimitedError(event.message, event.retry_after_seconds);
          }
          throw new Error(event.message);
        case "ready":
        case "pong":
          break;
      }
    }
  }

  async ping(): Promise<void> {
    this.send({ type: "ping" });
    const event = await this.next();
    if (event.type !== "pong") throw new Error(`expected pong, got ${event.type}`);
  }

  close(): void {
    this.socket?.close();
  }

  private send(frame: ClientFrame): void {
    if (!this.socket) throw new Error("not connected");
    if (this.closed) {
      throw new ConnectionClosedError("socket is closed", this.closed.code);
    }
    this.socket.send(JSON.stringify(frame));
  }

  private deliver(event: ServerEvent): void {
    const waiter = this.waiters.shift();
    if (waiter) waiter.resolve(event);
    else this.queue.push(event);
  }

  private next(timeoutMs = 30_000): Promise<ServerEvent> {
    const buffered = this.queue.shift();
    if (buffered) return Promise.resolve(buffered);
    if (this.closed) {
      return Promise.reject(
        new ConnectionClosedError(`socket closed (${this.closed.code})`, this.closed.code),
      );
    }
    return new Promise<ServerEvent>((resolve, reject) => {
      const timer = setTimeout(() => {
        const index = this.waiters.findIndex((w) => w.resolve === wrapped);
        if (index >= 0) this.waiters.splice(index, 1);
        reject(new Error(`timed out after ${timeoutMs}ms waiting for a frame`));
      }, timeoutMs);
      const wrapped = (event: ServerEvent): void => {
        clearTimeout(timer);
        resolve(event);
      };
      this.waiters.push({
        resolve: wrapped,
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      });
    });
  }
}

function defaultSocket(url: string): WebSocketLike {
  return new WebSocket(url);
}
