# cline-to-claude

A drop-in proxy that lets **Cline** talk to **Claude** over the Anthropic
Messages API.

Cline speaks the OpenAI Chat Completions wire format (`/v1/chat/completions`,
`tools`, streaming SSE). Anthropic speaks the Messages API
(`/v1/messages`, `tool_use` / `tool_result`, `content_block_delta` events).
This proxy sits in the middle and translates both directions — including
streaming, tool calls, images, usage and errors.

It is the mirror image of its sibling project
[`claude-code-proxy`](../tor-proxy-toolkit/claude-code-proxy), which does
*Claude-wire client → OpenAI-wire backend*. This one does
*OpenAI-wire client (Cline) → Claude-wire backend*.

```
┌──────────┐   OpenAI chat/completions   ┌──────────────────┐   Anthropic messages   ┌────────────────────┐
│  Cline   │ ──────────────────────────► │  cline-to-claude │ ─────────────────────► │ api.anthropic.com  │
│ (VS Code │ ◄────────────────────────── │  (this proxy)    │ ◄───────────────────── │ …or any Anthropic- │
│  or CLI) │      streaming chunks       │                  │      SSE events        │ compatible gateway │
└──────────┘                             └──────────────────┘                        └────────────────────┘
```

## Features

| Area | What is translated |
| --- | --- |
| Endpoints | `POST /v1/chat/completions` (also `/chat/completions`), `GET /v1/models`, `/health`, `/test-connection`, `/api/status`, dashboard at `/` |
| Passthrough | `POST /v1/messages` and `/v1/messages/count_tokens` for Anthropic-wire clients |
| Messages | `system`/`developer` roles lifted into Anthropic `system`; consecutive user/tool turns merged; `tool` messages become `tool_result` blocks (always ordered first in their turn) |
| Tools | OpenAI `tools` → Anthropic `tools` with normalised object schemas; `tool_choice` `auto`/`required`/`none`/named-function mapping; `parallel_tool_calls: false` → `disable_parallel_tool_use` |
| Tool calls | Anthropic `tool_use` → OpenAI `tool_calls`, including streamed `input_json_delta` fragments re-assembled per tool index |
| Images | OpenAI `image_url` data: URLs → base64 sources, http(s) URLs → url sources |
| Reasoning | Anthropic `thinking` → `reasoning_content` (streamed too); optional extended thinking driven by Cline's reasoning effort |
| Usage | `input_tokens`/`output_tokens` + cache reads → `prompt_tokens`/`completion_tokens`/`prompt_tokens_details.cached_tokens` |
| Streaming | SSE comment keepalives during silent thinking phases, final usage chunk, `[DONE]` terminator, upstream errors surfaced as real HTTP status codes |
| Resilience | Retries with backoff on 429/5xx/529 (stream-safe: only before the first byte), and the last upstream failure is persisted for the dashboard |
| Structured outputs | OpenAI `response_format` (json_schema) → Anthropic `output_config` |
| Prompt caching | `cache_control: {type: "ephemeral"}` on text/image blocks passed through to Anthropic |
| Tool error flag | OpenAI `is_error` on tool messages → Anthropic `tool_result.is_error` |
| Vision detail | OpenAI `detail` (low/high/auto) on images preserved in logs/metadata |

## Quick start

```bash
cd ~/cline-to-claude
cp .env.example .env                         # then edit it
python3 -m pip install -r requirements.txt   # or: uv sync
python3 start_proxy.py
```

`start_proxy.py --help` prints every supported environment variable.

Minimal `.env` for talking to Anthropic directly:

```ini
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=https://api.anthropic.com
```

Pointing at an Anthropic-compatible gateway (OAuth-style tokens often want a
bearer header instead of `x-api-key`):

```ini
ANTHROPIC_API_KEY=your-token
ANTHROPIC_BASE_URL=https://your-gateway.example.com
ANTHROPIC_AUTH_MODE=bearer
```

Chaining into another local Anthropic-wire proxy (this is what the checked-in
`.env` does, using the `claude-code-proxy` on port 4013):

```ini
ANTHROPIC_API_KEY=proxy
ANTHROPIC_BASE_URL=http://127.0.0.1:4013
```

Verify the upstream before involving Cline:

```bash
curl -s http://127.0.0.1:4014/health
curl -s http://127.0.0.1:4014/test-connection   # one real upstream round trip
```

## Wiring Cline to it

The proxy listens on `http://127.0.0.1:4014` by default and exposes the OpenAI
wire, so use Cline's **OpenAI Compatible** provider.

**VS Code / desktop extension**

| Field | Value |
| --- | --- |
| Provider | OpenAI Compatible |
| Base URL | `http://127.0.0.1:4014/v1` |
| API Key | anything (e.g. `proxy`) unless you set `CLIENT_API_KEY` |
| Model ID | any advertised model, e.g. `claude-sonnet-4-5` (`/v1/models` lists them) |

**Cline CLI** (verified with `cline --version` 3.0.62)

```bash
cline auth openai-compatible --apikey proxy --baseurl http://127.0.0.1:4014/v1 --modelid claude-sonnet-4-5
cline -P openai-compatible -m claude-sonnet-4-5 "explain this repo"
```

> **CLI 3.0.62 notes** (verified against the shipped binary):
> - `cline auth` writes to the **default config dir**
>   (`~/.cline/data/settings/providers.json`) and ignores `--data-dir`; use
>   `--config <dir>` if you want an isolated sandbox.
> - Use the long flags. The persisted key is `baseUrl`; a short-flag `-b` run
>   kept `apiKey`/`model` but dropped the URL, which silently sends traffic to
>   `api.openai.com`. If that happens, add it by hand:
>
> ```json
> {
>   "version": 1,
>   "lastUsedProvider": "openai-compatible",
>   "modes": {},
>   "providers": {
>     "openai-compatible": {
>       "settings": {
>         "provider": "openai-compatible",
>         "apiKey": "proxy",
>         "baseUrl": "http://127.0.0.1:4014/v1",
>         "model": "claude-sonnet-4-5"
>       },
>       "tokenSource": "manual"
>     }
>   }
> }
> ```
>
> Sanity check with `grep -o 'http://127.0.0.1:4014[^"]*' ~/.cline/logs/cline.log`;
> if requests show `api.openai.com` instead, the base URL was not persisted.

## Model mapping

Requests are routed by name class, so `-m claude-sonnet-4-5` works out of the
box and vendor-specific ids still land somewhere sensible:

| Cline asks for | Upstream model |
| --- | --- |
| anything containing `opus` | `BIG_MODEL` |
| anything containing `sonnet` | `MIDDLE_MODEL` |
| anything containing `haiku` | `SMALL_MODEL` |
| an id already starting with `claude-` | passed through unchanged |
| anything else | `DEFAULT_MODEL` (or unchanged with `PASSTHROUGH_UNKNOWN_MODELS=true`) |

`MODEL_MAP` adds exact aliases on top:

```ini
MODEL_MAP=fast=claude-haiku-4-5,smart=claude-opus-4-1
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/chat/completions` | OpenAI wire in, translated to Anthropic (stream + non-stream) |
| POST | `/chat/completions` | Same, for base URLs without the `/v1` suffix |
| GET | `/v1/models`, `/models` | Advertised model ids |
| POST | `/v1/messages` | Anthropic passthrough (streaming included) |
| POST | `/v1/messages/count_tokens` | Token counting passthrough |
| GET | `/health` | Liveness + upstream config summary |
| GET | `/test-connection` | One real upstream round trip, `{"ok": true|false}` |
| GET | `/api/status` | Machine-readable config, stats, failure history |
| GET | `/` | HTML dashboard |

## Configuration reference

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | – | Upstream credential |
| `ANTHROPIC_AUTH_TOKEN` | – | Bearer-style alternative to `ANTHROPIC_API_KEY` |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Upstream base URL |
| `ANTHROPIC_AUTH_MODE` | `x-api-key` | `x-api-key` or `bearer` |
| `ANTHROPIC_VERSION` | `2023-06-01` | `anthropic-version` header |
| `ANTHROPIC_BETA` | – | Optional `anthropic-beta` header |
| `BIG_MODEL` / `MIDDLE_MODEL` / `SMALL_MODEL` | `claude-opus-4-1` / `claude-sonnet-4-5` / `claude-haiku-4-5` | Upstream ids per name class |
| `DEFAULT_MODEL` | `MIDDLE_MODEL` | Fallback upstream id |
| `MODEL_MAP` | – | `alias=model,alias2=model2` |
| `PASSTHROUGH_UNKNOWN_MODELS` | `false` | Forward unknown ids unchanged |
| `CLIENT_API_KEY` | – | Shared secret Cline must send (unset = open) |
| `THINKING_BUDGET_TOKENS` | `0` | Base extended-thinking budget, scaled by reasoning effort (`low` 0.25× … `xhigh` 2×); `0` disables thinking translation |
| `THINKING_MIN_BUDGET_TOKENS` | `1024` | Lower bound for the computed thinking budget |
| `DEFAULT_MAX_TOKENS` | `8192` | `max_tokens` used when Cline omits it |
| `MAX_TOKENS_LIMIT` / `MIN_TOKENS_LIMIT` | `128000` / `100` | Clamp for `max_tokens` |
| `HOST` / `PORT` | `127.0.0.1` / `4014` | Listen address |
| `LOG_LEVEL` | `INFO` | `DEBUG` logs per-request translation details |
| `REQUEST_TIMEOUT` | `180` | Upstream read timeout (seconds) |
| `MAX_RETRIES` | `2` | Retries for 429/5xx/529 before the first streamed byte |
| `STREAM_KEEPALIVE_SECS` | `15` | SSE comment interval during silent phases; `0` disables |
| `STREAM_INCLUDE_USAGE` | `true` | Emit a usage chunk when Cline does not specify `stream_options` |
| `PROXY_LOG_DIR` | `./logs` | Where failure dumps are written (`last_upstream_failure.json`) |
| `CUSTOM_HEADER_*` | – | Extra upstream headers (`CUSTOM_HEADER_X_FOO=bar` → `X-Foo: bar`) |

## Tests

```bash
python3 -m pytest -q     # 63 tests: converters, streaming, HTTP end-to-end
```

The suite runs the FastAPI app against a real (fake) Anthropic server on a
loopback port, so request translation, SSE re-emission, the full tool loop,
error propagation and the status endpoints are all exercised over HTTP.

`scripts/run_fake_upstream.py 4020 tool_then_text` starts that fake upstream
standalone — handy for driving the real Cline CLI without burning tokens:

```bash
python3 scripts/run_fake_upstream.py 4020 tool_then_text &
ANTHROPIC_BASE_URL=http://127.0.0.1:4020 python3 start_proxy.py
```

### Verified against

- `cline` CLI **3.0.62** with the `openai-compatible` provider: streaming with
  25 tools, plus a complete agent loop (tool call out, `tool_result` back in,
  final answer) — the proxy log shows messages growing `1 → 3` across the two
  upstream turns.
- Upstream failures: a real `500` from a misbehaving local gateway surfaced as
  `{"detail": "Upstream error (500). …"}` with the upstream body preserved.

## Behaviour notes and limitations

- **Reasoning history is dropped.** Anthropic rejects replayed thinking blocks
  that were not signed by the same model, so `reasoning_content` from earlier
  turns is stripped from outgoing requests. It is still *returned* to Cline.
- **`n > 1` is ignored** — Anthropic returns one candidate per request and the
  proxy always reports `index: 0`.
- **`tool_choice: "none"`** drops the tool list entirely, because Anthropic has
  no "none" mode.
- **`temperature`/`top_p` are clamped to 0–1**, and extended thinking forces
  `temperature = 1`, matching Anthropic's constraints.
- **Usage chunk timing.** Streaming usage arrives in a final chunk with an empty
  `choices` array. Cline reads it correctly; strict OpenAI clients that expect
  usage on every chunk will only see it at the end.
- **Empty tool output** becomes `"(no output)"`, so the upstream never receives
  an empty `tool_result`.
- **Streaming errors after the first byte** cannot change the HTTP status; they
  are emitted as an SSE `error` object followed by `[DONE]`. Failures while
  *opening* the stream are returned as proper JSON errors with the upstream
  status code (401/404/429/5xx) because the proxy primes the upstream stream
  before sending response headers.
- **Extended thinking is opt-in** and only switches on when the client asks for
  reasoning (`reasoning_effort`, `thinking`, or a `reasoning` block) *and*
  `THINKING_BUDGET_TOKENS > 0`.