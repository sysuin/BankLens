# MCP server runbook

**Tracked, ships.** How to expose BankLens to an MCP client — Claude Desktop,
an IDE, another agent — and how to use it once connected. Written for someone
who has never touched the pipeline.

## What the MCP server is

`mcp_server.py` is the third front door onto the same engine, beside the
Streamlit console and the REST/SSE API. It speaks the Model Context Protocol
over stdio, so an MCP host can call BankLens as a tool instead of a person
driving the UI.

The wrapper is deliberately thin: it re-exports the functions the pipeline
already provides, in the order the app calls them. No analysis logic lives in
it, so the MCP surface cannot drift from what the product does.

## The three tools

| Tool | Arguments | Cost | Returns |
|---|---|---|---|
| `compute_statement_metrics` | `csv_path` | **Free** — no model call | Income, expenses, savings rate, expense ratio, risk band, health score, essential/discretionary split, top categories |
| `analyze_statement` | `csv_path`, `tenant` (optional) | One profiling call | The metrics above **plus** the generated profile: persona, two product recommendations, RM pitch hooks, retrieved sources |
| `search_products` | `query`, `tenant` (optional) | Embedding call | Ranked passages from that bank's catalogue, each with its source filename |

`compute_statement_metrics` takes no `tenant` on purpose: the numbers are
arithmetic over the statement and do not depend on which bank's products are
being matched. Only the two tools that touch a catalogue are tenant-scoped.

### Tenants

`tenant` is a bank slug — a directory under `knowledge_base/`. Today:

| Slug | Catalogue |
|---|---|
| `meridian` | The default when `tenant` is omitted |
| `harbor` | The second synthetic bank |

The slug is validated by `app.pipeline.rag.validate_tenant_slug` and checked
against the catalogues on disk. An unknown slug fails with an error that
**names the ones that exist**, so a calling agent can correct itself rather
than guess.

## Registering it in Claude Desktop

### The one rule that matters

**Quit Claude Desktop completely (⌘Q) before editing its config.** The app
rewrites `claude_desktop_config.json` when it exits and silently drops
entries added while it was running. Closing the window is not enough — check
there is no dot under the Dock icon. Two separate attempts were lost to this
before it was understood.

### Steps

1. Quit Claude Desktop (⌘Q).
2. Open `~/Library/Application Support/Claude/claude_desktop_config.json` and
   add `banklens` inside `mcpServers`, beside anything already there:

```json
{
  "mcpServers": {
    "banklens": {
      "command": "/absolute/path/to/BankLens/.venv/bin/python",
      "args": ["/absolute/path/to/BankLens/mcp_server.py"]
    }
  }
}
```

   Both paths must be absolute, and the interpreter must be the project's
   virtualenv — the one with the dependencies installed, not the system
   Python. Mind the commas: invalid JSON makes Claude Desktop ignore **every**
   server, not just this one.
3. Start Claude Desktop. Open the tools/connector control in the chat input;
   `banklens` should be listed with three tools.

No API key goes in the config. The server changes its working directory to
the project root before importing anything, so `.env` — and the key in it —
resolves however the host launches the process.

## Using it

Tools are chosen by the model from their descriptions; you ask in plain
language. Statement paths must be **absolute**, because the host's working
directory is not the project.

```
Use the banklens tools to analyze
/abs/path/to/BankLens/data/sample_3_cashflow_stressed.csv
and tell me what you would pitch this customer.

Compute the metrics for /abs/path/to/BankLens/data/sample_1_high_saver.csv —
no profile, just the numbers.

Search the harbor catalogue for something suitable for a customer with high
idle cash who still wants liquidity.

Analyze /abs/path/to/BankLens/data/sample_2_active_spender.csv against the
meridian catalogue, then against harbor, and explain why the recommendations
differ.
```

That last one is the demonstration worth doing: the same statement, the same
engine, two banks' catalogues, and the recommendations change because the
catalogues do — tenancy visible from the outside.

`sample_3_cashflow_stressed.csv` is the other one to try. It comes back
`High` risk, `is_cashflow_negative: true`, with Debt Payments as the largest
outflow and a debt-consolidation recommendation rather than a credit card —
the guardrail is a computed flag, not a model judgement.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `banklens` missing from the tools list | Config edited while the app was running, or invalid JSON | Quit, re-add, check the JSON parses, start |
| Server listed but every call errors | Wrong interpreter — a Python without the dependencies | Point `command` at the project virtualenv |
| "No statement file at: …" | Relative path | Use an absolute path |
| `unknown tenant 'x'; available: [...]` | Working as intended | Use a slug the error names |
| Calls fail with an auth error | `.env` missing or without a key | The key lives in `.env` at the project root, never in the Desktop config |
| First `search_products` for a tenant is slow | That tenant's vector index is being built | Expected once per tenant and provider; it persists under `chroma_db/<tenant>/<provider>/` |

The server's own log is `~/Library/Logs/Claude/mcp-server-banklens.log`. The
line `Server started and connected successfully` means the host launched it;
its absence means the entry never loaded.

### Why logs go to stderr

Under stdio transport **stdout is the JSON-RPC wire**. A log line printed
there is protocol corruption, not a log — a strict client drops the session.
`mcp_server.py` calls `route_logs_to_stderr()` before serving, which is where
MCP hosts collect server logs. Keep it that way when adding logging.

## Checking it without Claude Desktop

Any MCP client works. A direct handshake, useful when something is broken and
you want to know whether the fault is the server or the host:

```bash
.venv/bin/python mcp_server.py   # then speak JSON-RPC on stdin
```

The repository's own check is `tests/test_mcp_server.py`, which asserts the
three tools are registered with descriptions and that a missing file raises.
