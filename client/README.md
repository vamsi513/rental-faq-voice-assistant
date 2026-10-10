# TypeScript client for /ws/chat

A terminal client for the rental assistant's WebSocket endpoint. It authenticates,
streams the answer token by token, and handles the two things a naive client gets
wrong: a retracted answer, and a rate limit refusal that is not a disconnect.

```bash
npm install
npm run build
RENTAL_WS_SECRET=your-key node dist/cli.js --url ws://127.0.0.1:8000/ws/chat
```

The secret is read from the environment rather than an argument, so it does not land
in shell history or in the process list. Omit it if the server has no
`SERVER_SECRET` set.

```
you  what is the pet fee?
     The pet fee is two hundred and fifty dollars per pet, and it is a one time,
     non refundable fee.
     [property_question  1.56s]
```

## What it handles

| Case | Behaviour |
|---|---|
| `delta` | Written to the terminal as it arrives. 35 frames for a typical answer. |
| `reset` | The server retracted what it already streamed. The line is cleared and the correction is shown instead. A client that ignores this leaves a withdrawn answer on screen. |
| `ticket` | Prints the demo ticket id, or the reason a submission could not be confirmed. Never prints a ticket for a failed submission. |
| `error` with `retry_after_seconds` | Raised as `RateLimitedError`. The socket stays open, so the right move is to wait, not reconnect. |
| Close mid stream | Raised as `ConnectionClosedError` with the code, and the CLI prints the `--resume` command to reattach. |
| Unknown frame type | Dropped. A server that gains a frame type should not break an older client. |

## Reconnecting

The server keeps the conversation in its checkpoint, so a dropped connection does not
lose context:

```bash
node dist/cli.js --resume ws-1a2b3c4d5e6f
```

`RentalChatClient.callId` exposes the id to save, and `resumed` says whether the
server honoured it.

## Using it as a library

```ts
import { RentalChatClient } from "./dist/client.js";

const client = new RentalChatClient({ url, secret });
await client.connect();

const result = await client.ask("what is the pet fee?", (piece, retracted) => {
  if (retracted) clearTheLine();
  else process.stdout.write(piece);
});

console.log(result.text, result.route, result.retracted, result.ticket);
```

`createSocket` can be injected to supply a different transport, which is how the
tests drive the client against an in-process server.

## Development

```bash
npm run typecheck    # tsc --noEmit, strict, covers tests too
npm run lint         # eslint with type-aware rules, zero warnings allowed
npm test             # vitest, 15 tests
npm run build        # emits dist/ from src/ only
```

The tests run against a real `ws` server on an ephemeral port rather than a stubbed
socket object, so the framing, the JSON and the close codes are genuinely exercised.
Only the model behind the server is scripted.

`tsconfig.json` is strict and adds `noUncheckedIndexedAccess`,
`exactOptionalPropertyTypes` and `verbatimModuleSyntax`.

## Limits

- One request in flight at a time per client. The server serialises turns per
  connection anyway, so pipelining would only queue.
- No automatic reconnect loop. `--resume` is manual, deliberately: an automatic
  retry against a server that is rate limiting would make things worse.
- Node 20 or newer, for the native `node:readline/promises` API.
