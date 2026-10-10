/**
 * The /ws/chat wire protocol.
 *
 * Mirrors app/ws.py. Kept as a discriminated union so a missing case in a switch is
 * a compile error rather than a frame silently ignored at runtime.
 */

export interface AuthFrame {
  type: "auth";
  secret: string;
  /** Reattach to an earlier thread instead of starting a new one. */
  resume?: string;
}

export interface ResumeFrame {
  type: "resume";
  call_id: string;
}

export interface UserFrame {
  type: "user";
  text: string;
}

export interface PingFrame {
  type: "ping";
}

export type ClientFrame = AuthFrame | ResumeFrame | UserFrame | PingFrame;

export interface ReadyEvent {
  type: "ready";
  call_id: string;
  resumed?: boolean;
  turns?: number;
}

export interface DeltaEvent {
  type: "delta";
  text: string;
}

/** Discard everything streamed so far. A correction follows as fresh deltas. */
export interface ResetEvent {
  type: "reset";
}

export interface TicketEvent {
  type: "ticket";
  status: "submitted" | "failed";
  ticket_id: string | null;
  error: string | null;
}

export interface DoneEvent {
  type: "done";
  route: string;
  latency_seconds: number;
}

export interface ErrorEvent {
  type: "error";
  message: string;
  /** Present only on a rate limit refusal. */
  retry_after_seconds?: number;
}

export interface PongEvent {
  type: "pong";
}

export type ServerEvent =
  | ReadyEvent
  | DeltaEvent
  | ResetEvent
  | TicketEvent
  | DoneEvent
  | ErrorEvent
  | PongEvent;

/** Close code the server uses for a bad auth frame or an oversized frame. */
export const CLOSE_POLICY_VIOLATION = 1008;
/** Close code the server uses when it is at its connection cap. */
export const CLOSE_TRY_AGAIN_LATER = 1013;

const SERVER_EVENT_TYPES = new Set([
  "ready",
  "delta",
  "reset",
  "ticket",
  "done",
  "error",
  "pong",
]);

/**
 * Narrow an untrusted payload to a ServerEvent.
 *
 * The server is not assumed to be well behaved. Anything unrecognised returns null
 * and is dropped rather than crashing the client mid stream.
 */
export function parseServerEvent(raw: string): ServerEvent | null {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof value !== "object" || value === null) return null;
  const candidate = value as { type?: unknown };
  if (typeof candidate.type !== "string" || !SERVER_EVENT_TYPES.has(candidate.type)) {
    return null;
  }
  return value as ServerEvent;
}
