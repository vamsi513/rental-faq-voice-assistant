/**
 * Terminal chat against a running rental assistant.
 *
 *   npm run build && node dist/cli.js --url ws://127.0.0.1:8000/ws/chat
 *
 * Reads the secret from RENTAL_WS_SECRET rather than an argument, so it does not
 * land in shell history or in the process list.
 */
import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";

import { RateLimitedError, RentalChatClient, ConnectionClosedError } from "./client.js";

interface Args {
  url: string;
  resume: string | undefined;
}

function parseArgs(argv: readonly string[]): Args {
  let url = "ws://127.0.0.1:8000/ws/chat";
  let resume: string | undefined;
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (flag === "--url" && value) {
      url = value;
      i += 1;
    } else if (flag === "--resume" && value) {
      resume = value;
      i += 1;
    }
  }
  return { url, resume };
}

async function main(): Promise<number> {
  const { url, resume } = parseArgs(process.argv.slice(2));
  const secret = process.env["RENTAL_WS_SECRET"];

  const client = new RentalChatClient({
    url,
    ...(secret ? { secret } : {}),
    ...(resume ? { resume } : {}),
  });

  try {
    await client.connect();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`could not connect: ${message}`);
    if (err instanceof ConnectionClosedError && err.code === 1008) {
      console.error("the server rejected the access key; set RENTAL_WS_SECRET");
    }
    return 1;
  }

  console.log(`connected  call ${client.callId}${client.resumed ? "  (resumed)" : ""}`);
  console.log("type a question, or /quit to leave\n");

  const rl = createInterface({ input: stdin, output: stdout });
  try {
    for (;;) {
      const line = (await rl.question("you  ")).trim();
      if (!line) continue;
      if (line === "/quit" || line === "/exit") break;

      stdout.write("     ");
      let wrote = false;
      try {
        const result = await client.ask(line, (piece, retracted) => {
          if (retracted) {
            // The answer was withdrawn. Clear the line and start again rather than
            // leaving a retracted sentence on screen.
            stdout.write("\r\u001b[2K     [retracted] ");
            wrote = false;
            return;
          }
          stdout.write(piece);
          wrote = true;
        });
        if (!wrote) stdout.write(result.text);
        stdout.write(`\n     [${result.route}  ${result.latencySeconds}s]\n\n`);
        if (result.ticket) {
          const t = result.ticket;
          console.log(
            t.status === "submitted"
              ? `     ticket ${t.ticket_id ?? "?"} accepted\n`
              : `     not submitted: ${t.error ?? "unknown"}\n`,
          );
        }
      } catch (err) {
        if (err instanceof RateLimitedError) {
          console.log(`\n     rate limited, wait ${err.retryAfterSeconds}s\n`);
        } else if (err instanceof ConnectionClosedError) {
          console.error(`\n     connection closed (${err.code})`);
          if (client.callId) {
            console.error(`     reconnect with: --resume ${client.callId}\n`);
          }
          return 1;
        } else {
          console.error(`\n     ${err instanceof Error ? err.message : String(err)}\n`);
        }
      }
    }
  } finally {
    rl.close();
    client.close();
  }
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch((err: unknown) => {
    console.error(err);
    process.exit(1);
  });
