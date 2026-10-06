# Providers, retry and context management

Only `src/vbt/providers/` knows about a specific LLM API. Agents, tools, MCP
servers, orchestration, bulk runs and case studies use the neutral types in
`providers/base.py`, plus two provider-neutral services: `providers/retry.py`
(transient-error retry) and `context.py` (context-window management). Neither
service, nor `base.py` or `mock.py`, imports a vendor SDK (a test enforces this).

## Neutral types

- `Message(role, content=[TextBlock | ThinkingBlock | ToolCall | ToolResult | OpaqueBlock])`
- `ToolResult(tool_call_id, content, is_error)`, where `content` is a string or a
  list of parts: `TextBlock`, `ImagePart(media_type, data_b64, source)` and
  `DocumentPart(data_b64=..., media_type="application/pdf", title=...)`.
  `content_text(content)` flattens parts for logs and for providers without
  vision (images render as `[image: <source>]`).
- `ToolSpec(name, description, input_schema, strict=False)` — JSON Schema, passed
  through from MCP. `strict=True` asks providers that support it to constrain the
  arguments to the schema (OpenAI-compatible servers; the Claude adapter ignores it).
- `ModelSettings(provider, model, max_tokens, effort, thinking, temperature, extra)`.
- System prompt: a string or a list of `SystemSegment(text, cache=False)`.
  `cache=True` marks the end of a stable prefix (the static upstream prompt);
  later segments are volatile (date, run paths). `system_text()` joins them.
- `ModelResponse(message, stop_reason, usage, model, cost_usd, stop_detail,
  request_id, served_model, fallback_used, retries, iterations)`.
- `Usage(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
  cache_write_1h_tokens, server_tool_requests)`. `cache_write_tokens` counts all
  cache writes; `cache_write_1h_tokens` is the 1-hour-TTL subset;
  `server_tool_requests` counts provider-side tool calls, e.g. `{"web_search": 2}`.
- `StopReason`: `end_turn`, `tool_use`, `max_tokens`, `refusal`, `pause_turn`,
  `context_exceeded` (the model ran out of window mid-generation), `other`.
- Errors: `ProviderError` (non-retryable: bad request, auth, not found),
  `RetryableProviderError(retry_after, status)` (overload, rate limit, 5xx,
  dropped connection, error events after streaming started) and
  `ContextOverflowError` (the request does not fit the window).
- `ProviderCapabilities(images, documents, web_search, server_context_management,
  history_bound_thinking, replays_reasoning, tool_choice)` from
  `provider.capabilities(model=None)`. `replays_reasoning`: earlier reasoning text
  is re-sent on every request (it counts against the window, so compaction should
  clear old reasoning with old tool results); `tool_choice`: the provider honours
  `extra["tool_choice"]`.
- History helpers: `validate_tool_pairing(messages, allow_pending=False)` lists
  violations of the tool_use/tool_result rules vendor APIs enforce (every call
  answered by a result at the start of the next user message, no orphan or
  duplicate results); `unanswered_tool_calls(messages)`; lossless
  `message_to_dict` / `message_from_dict` (including native payloads such as
  thinking signatures) for saving and resuming sessions.

## The provider interface

```python
class LLMProvider:
    async def complete(self, *, settings, system, messages, tools,
                       on_text=None, on_thinking=None) -> ModelResponse
    def capabilities(self, model=None) -> ProviderCapabilities
    def context_window(self, model) -> int | None
    def check_credentials(self) -> str | None      # problem text, no network
    async def prepare(self) -> None                # optional readiness check (default no-op)
    async def server_info(self) -> dict            # optional backend facts (default {})
    async def web_search(self, query, *, max_results=8,
                         allowed_domains=None, blocked_domains=None) -> dict
    def supports_web_search(self) -> bool
```

`prepare()` runs before the first call where a provider needs it (a local
server's health check and model discovery) and raises `ProviderError` with a
fix hint; `server_info()` returns facts such as the engine version and served
models for run records.

`on_text` / `on_thinking` receive streamed text and reasoning deltas; providers
without reasoning never call `on_thinking`. `web_search` returns
`{"results": [...], "summary": str, "cost_usd": float}`.

`ModelSettings.extra` carries harness hints; adapters consume the keys they know
and never forward any `extra` key to a vendor API. Keys in use:
`agent_name` (set by the roster; the mock uses it to pick a script queue),
`thinking_budget` (budget-thinking models), `context_window_tokens` (overrides
the provider's window table for compaction), `context_management` (request
server-side context editing) and `prompt_cache` (False disables caching for a
one-off request such as a compaction summary). The OpenAI-compatible adapter
also reads `session_key` (sticky replica routing), `tool_choice` (`'auto'`,
`'required'`, `'none'` or `{'name': <tool>}`), `reasoning_effort` (explicit
wire value, still mapped to what the model accepts) and `sampling` (a dict
overriding the model family's sampling; a `None` value removes a key).

## Add a provider

1. Subclass `LLMProvider` and implement `complete(...)`:
   - encode neutral messages to the vendor format (tool calls ↔ function calls,
     tool results ↔ tool/function messages; image/document parts as vision
     inputs, or `content_text()` if the model has no vision);
   - map the vendor's stop reason to `StopReason` (`tool_use` whenever the reply
     contains tool calls);
   - raise `RetryableProviderError` for transient failures (with `retry_after`
     from the response headers when present) and `ContextOverflowError` for
     "prompt too long"; everything else the API rejects is `ProviderError`;
   - return `Usage` and a USD cost from your price table.
   - Keep vendor-only items (reasoning signatures, server-tool blocks) in
     `native` / `OpaqueBlock` so they replay unchanged to the same provider.
2. Report `capabilities()`, `context_window()` and `check_credentials()`.
3. Optionally implement `web_search(...)` (used by the `WebSearch` tool) and
   return `supports_web_search() -> True`.
4. Register it in `providers/__init__.py`: `register_provider("myvendor", factory)`.
5. Create a profile, e.g. `configs/profiles/myvendor.yaml`:

```yaml
provider:
  name: myvendor
  options: {}
models:
  orchestrator: {model: <id>, effort: null, max_tokens: 32000, thinking: true, context_window_tokens: 128000}
  scientist:    {model: <id>, effort: null, max_tokens: 32000, thinking: true}
  support:      {model: <cheaper id>, effort: null, max_tokens: 16000, thinking: false}
  bulk:         {model: <id>, effort: null, max_tokens: 16000, thinking: false}
```

6. Run `vbt --profile myvendor doctor --smoke` and the test suite; the
   scripted `mock` provider in `providers/mock.py` shows the minimal contract.

Requirements on the model: reliable parallel tool calling and long contexts
(the CSO accumulates a multi-turn conversation; scientists read large tool
outputs, which the harness truncates to `limits.tool_output_max_chars` and the
context manager clears once they age).

## Transient-error retry (`providers/retry.py`)

Vendor SDKs retry only the initial HTTP response, briefly (`provider.options.max_retries`).
Every agent-loop call goes through

```python
resp = await complete_with_retry(provider, policy=RetryPolicy.from_config(config.get("retry")),
                                 on_retry=callback, settings=..., system=..., messages=..., tools=...)
```

which retries `RetryableProviderError` with jittered exponential backoff
(`base_delay_s * 2**n`, ± `jitter`, capped at `max_delay_s`), or exactly the
server's `retry-after` when it sent one (capped at `max_retry_after_s`). That
includes overload/`api_error` events that arrive after streaming has started and
connections dropped mid-stream. Re-sending is safe: histories are append-only
and a failed response is never appended. Non-retryable errors propagate
immediately; after the last attempt the final error is re-raised with
`.attempts`. `resp.retries` reports how many attempts failed first.

Config (in-code defaults, also listed in `configs/default.yaml`): `retry.attempts: 8`,
`retry.base_delay_s: 2`, `retry.max_delay_s: 60`, `retry.jitter: 0.25`,
`retry.max_retry_after_s: 600`. The vendor SDK's own retries
(`provider.options.max_retries`) apply to each attempt, before the harness retry.

## Context-window management (`context.py`)

`ContextManager(provider, ContextPolicy.from_config(config.get("context")), spill_dir)`
keeps an agent's history inside its model's window. The window is
`models.<tier>.context_window_tokens` if set, else `provider.context_window(model)`,
else `context.default_window_tokens`; either way it is capped at the limit the serving
engine enforces when the provider knows it (`provider.server_context_window(model)`: a
local server's `max_model_len` from `/v1/models`), so a 262K tier against a server started
with `--max-model-len 65536` compacts against 65,536. The prompt size is read from the last response
(`input + cache_read + cache_write`, plus its output and any tool results
appended since).

- `maybe_compact(messages, last_usage=, settings=, system=, agent=)` — call after
  each response. Above `soft_ratio` (0.70) it clears old tool results: content of
  results older than the last `keep_recent_calls` (6) model calls and at least
  `min_clear_chars` (2000) long is saved to `spill_dir` (or the file the harness
  already saved it to is reused) and replaced by
  `[cleared by harness: N chars; full output at <path>; use QueryToolOutput or Read]`.
  Image/document parts are saved as files too. A clearing pass is skipped when
  it would free less than `min_clear_fraction` (5%) of the window, because every
  edit restarts the prompt cache from the edited message.
- Above `hard_ratio` (0.85), or still above the soft ratio after clearing, it
  summarises the oldest span with one provider call (same model, thinking off,
  no tools, `agent_name` `context-compactor`). The span never splits a
  tool_use/tool_result pair: the head user message (the task) is kept verbatim
  and the summary is appended to it as a `[Conversation summary — harness
  compaction]` block; the kept tail starts with an assistant message. The
  summary prompt asks for the plan, clarification answers, delegations with
  their key numbers and artifact paths, reviewer verdicts, filed claim IDs and
  open issues; the user's own messages from the span are appended verbatim by
  the harness, and carried forward through later compactions. Thinking blocks
  inside the span are dropped. If the summariser fails, a mechanical digest is
  used instead, so compaction always succeeds.
- `recover_overflow(messages, settings=, system=, agent=)` — call after a
  `ContextOverflowError` or a `context_exceeded` stop, then retry the request
  once: it clears everything but the last round, summarises with only that
  round kept, and as a last resort clears the last round's large results.
  Raises `ContextOverflowError` when nothing is left to compact.
- Both mutate `messages` in place, restore it on any failure, and guarantee
  `validate_tool_pairing(messages, allow_pending=True) == []` afterwards. They
  return `CompactionResult(strategy, tokens_before, tokens_after_estimate, usage,
  cost_usd, ...)` so the caller can charge and trace the summariser call.
- Thinking blocks bound to the conversation (`history_bound_thinking`: Claude
  Fable 5.1, Opus 5.5, Sonnet 5.5 "preserved thinking") become invalid after any
  client-side edit of earlier turns; the manager strips every thinking block
  after the first edited message for those models. Other models keep them
  (budget-thinking models need the last assistant turn's thinking during a tool
  loop).
- Server-side alternative: with `context.server_side: true` and a provider that
  reports `server_context_management`, `ContextManager.settings_for(settings)`
  adds `extra["context_management"]`, and the Claude adapter sends the
  `clear_tool_uses_20250919` edit with the `context-management-2025-06-27` beta.
  Server-side edits do not count as history edits, so caching and preserved
  thinking stay valid.

Config. In-code defaults: `context.enabled: true`, `soft_ratio: 0.70`, `hard_ratio: 0.85`,
`keep_recent_calls: 6`, `min_clear_chars: 2000`, `min_clear_fraction: 0.05`,
`summary_max_tokens: 4000`, `server_side: false`, `default_window_tokens: 200000`; optional
`models.<tier>.context_window_tokens`. The shipped `configs/default.yaml` (the local Qwen3.8
default) compacts earlier: `soft_ratio: 0.55`, `hard_ratio: 0.7`, `default_window_tokens:
262144` (and 262,144 per tier); `--profile claude` / `--profile paper` restore 0.70 / 0.85 /
200,000. When the provider reports `capabilities().replays_reasoning` (the local adapter: the
chat template replays every past reasoning block), the soft-clear pass also empties the
reasoning of turns older than `keep_recent_calls`, in the same pass as the tool results (one
prefix-cache miss, not one per turn), and summaries drop thinking.
Compactions and retries are traced (`compaction`, `provider_retry`), counted in each turn
record, and shown as notices by `vbt chat` and `vbt web`.

## Choosing a model at the command line

`--model` (chat, run, replay, scenario) sets the orchestrator, scientist and bulk tiers. It
accepts a `model_aliases` label, a model id already configured in a tier, or an id matching
`provider.model_pattern`. The aliases depend on the profile: the default local config has only
`qwen` (`qwen3.8-27b`); `--profile claude` and `--profile paper` add `opus`, `sonnet`, `haiku`
and `paper`. The default pattern is `^claude-[a-z0-9.-]+$` for the Anthropic provider, any
served-model-like name (`qwen3.8-27b`, `Qwen/Qwen3.8-27B-FP8`, `model.gguf`) for the local
providers (vllm, sglang, llamacpp, openai_compat), and no check for others. Anything else exits
with code 2 and lists the aliases.

For a local provider the tier's model is the name requested from the server, so `--model` must
be a name the server serves (`--served-model-name`); `prepare()` checks every configured tier
model before the first turn. `provider.options.served_model_name`, when set, overrides every
tier on the wire; the shipped configs leave it unset, and `--model` (or the web model picker)
with a different name than a configured `served_model_name` exits with code 2 rather than
recording a model that never ran. The chosen models, with the effective thinking/effort per
tier, are pinned in the run's `inputs/config.json`.

## Local models (OpenAI-compatible servers)

`providers/openai_compat.py` (`OpenAICompatProvider`) runs the harness on a
self-hosted open-weight model behind an OpenAI Chat Completions API. It is
registered as `vllm` (primary), `sglang`, `llamacpp` and `openai_compat`; the
registered name is the provider's name, and it selects small dialect
differences (SGLang and llama.cpp get prior reasoning back as
`reasoning_content`, vLLM as `reasoning`). It uses only `httpx`; no SDK.

The reference deployment is **Qwen3.8-27B** (Apache-2.0) on **vLLM 0.31.0** on
one 80-96 GB GPU (`Qwen/Qwen3.8-27B-FP8` on H100/H200,
`RedHatAI/Qwen3.8-27B-NVFP4` on RTX PRO 6000 / B200), served as
`qwen3.8-27b` with `--reasoning-parser qwen3 --enable-auto-tool-choice
--tool-call-parser qwen3_coder --tool-strict-level function
--enable-prompt-tokens-details --enable-prefix-caching --language-model-only`.
Serving profiles and commands live in `configs/local_models.yaml` and
`vbt local serve`.

```yaml
provider:
  name: vllm
  options:
    base_url: ${VBT_LLM_BASE_URL:-http://localhost:8000/v1}
    # served_model_name: qwen3.8-27b # optional: overrides models.<tier>.model (and --model) on the wire
    family: auto                     # or qwen3_8 | qwen3_6 | qwen3 | deepseek_v4 | generic
    read_timeout_s: 900              # max silence between streamed chunks (no total cap)
    max_concurrency: 48              # ~ the server's --max-num-seqs
models:
  orchestrator: {model: qwen3.8-27b, effort: high,   thinking: true,  max_tokens: 40960, thinking_budget: 24576}
  scientist:    {model: qwen3.8-27b, effort: medium, thinking: true,  max_tokens: 32768, thinking_budget: 8192}
  support:      {model: qwen3.8-27b, effort: null,   thinking: false, max_tokens: 16000}
  bulk:         {model: qwen3.8-27b, effort: medium, thinking: true,  max_tokens: 8192,  thinking_budget: 3072}
```

**Options.** `base_url` (env `VBT_LLM_BASE_URL` when unset; a bare
`http://host:port` gets `/v1`), `base_urls` (several replicas), `model`,
`served_model_name` (unset in the shipped configs; when set it overrides every tier's model
on the wire), `api_key` (optional; else env `VBT_LLM_API_KEY`, else the variable named by
`api_key_env`; never required, and `OPENAI_API_KEY` is never read implicitly, so the user's
OpenAI credential is not sent to a self-hosted server: opt in with `api_key_env:
OPENAI_API_KEY` for a hosted OpenAI-compatible API), `family`, `timeout_s` (connect/write, 30),
`read_timeout_s` (600), `max_concurrency`, `data_parallel_size` + `routing`
(`header` | `urls`), `pricing` (`{input_per_mtok, cached_per_mtok,
output_per_mtok}`, default 0), `extra_body` (merged into every request last,
dicts one level deep), `context_window`, `vision` (default false),
`reasoning_effort_supported` (generic family), `replay_reasoning_field`,
`parallel_tool_calls` (true), `auto_discover` (true), `wait_ready_s` (0),
`trust_env` (default: honour proxy variables unless every base URL is a
loopback address), `headers`.

**Model families** (`providers/families.py`, `resolve_family(model_id,
explicit=None)`): matched from the served name, the tier model or the served
model's `root` path (from `/v1/models`).

| family | models | reasoning control | thinking budget | sampling (thinking / plain) |
|---|---|---|---|---|
| `qwen3_8` | Qwen3.8-* | top-level `reasoning_effort`: low→`low`, medium→`medium`, high/xhigh/max→`xhigh`; thinking off or no effort → `none` (+ `chat_template_kwargs.enable_thinking=false`) | `thinking_token_budget` | T 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, rep 1.0 / T 0.7, top_p 0.8, top_k 20, min_p 0, presence 1.5 |
| `qwen3_6` | Qwen3.5-*, Qwen3.6-* | `chat_template_kwargs {enable_thinking, preserve_thinking: true}` (no effort levels) | yes | T 1.0, top_p 0.95, top_k 20, presence 1.5 / T 0.7, top_p 0.8, presence 1.5 |
| `qwen3` | Qwen3-* (2025, e.g. Qwen3-0.6B for CPU smoke tests) | `chat_template_kwargs.enable_thinking` | yes | T 0.6, top_p 0.95, top_k 20 / T 0.7, top_p 0.8, top_k 20 |
| `deepseek_v4` | DeepSeek-V4-Flash | `reasoning_effort` none/low/high/max (medium→low, xhigh→high, and the harness's max→high: Think-Max needs `max_tokens` >= 128K, so `max` is sent only as an explicit `extra["reasoning_effort"]`) + `chat_template_kwargs.thinking` | no | T 1.0, top_p 0.95 |
| `generic` | anything else | nothing (or standard low/medium/high with `reasoning_effort_supported`) | no | server defaults |

`high`, `max` and `minimal` are never sent to Qwen3.8: its chat template raises
("Unexpected reasoning effort", HTTP 400). Keep an agent's effort fixed: `xhigh`
and `low` add a line at the top of the system prompt, so switching invalidates
the prefix cache (`medium`↔`none` does not). `thinking_token_budget` is
`models.<tier>.thinking_budget`, else `min(max_tokens // 2, 16384)`, always below
`max_tokens` (which includes reasoning tokens). Sampling is sent explicitly on
every request (vLLM otherwise applies `generation_config.json` defaults);
`extra["sampling"]` overrides it, `ModelSettings.temperature` overrides the
temperature, and `top_k <= 0` is never sent.

**Request encoding.**
- One `system` message at index 0 (all `SystemSegment`s joined; put volatile
  segments last). Harness reminders are user text, never mid-conversation
  system messages.
- Assistant turns: `content` (or `null` with only tool calls), `tool_calls`
  with JSON-object `arguments` strings, and the turn's reasoning in `reasoning`
  — only reasoning this provider (or the same family through another
  OpenAI-compatible name) produced; Anthropic thinking and `OpaqueBlock`s are
  dropped. The Qwen3.8 template keeps `preserve_thinking` at its default (true),
  so the prompt stays append-only and prefix-cache friendly.
- A harness user message becomes `role: tool` messages in the order of the
  preceding assistant's tool calls (the template pairs them by position, not by
  id), then a separate `user` message with any remaining text. Tool content is a
  string (`content_text`); error results are prefixed `Error:` when they are not
  already. Images become `[image: <source>]` text, or, with `vision: true`,
  `image_url` data URIs in that user message (never inside tool messages); PDFs
  become a `[document: <title>]` placeholder.
- A conversation without any user message raises `ProviderError` (the template
  would raise "No user query found").
- Tools: `{type: function, function: {name, description, parameters, strict?}}`.
  Names outside `^[A-Za-z0-9_-]{1,64}$` get a stable alias (sanitised prefix +
  `_` + 8 hex of SHA-1) that is mapped back on decode. `tool_choice` comes from
  `extra["tool_choice"]` (`{'name': 'submit_result'}` forces one tool), else
  `auto`; `parallel_tool_calls: true`.
- `max_tokens` is clamped to the known window minus a rough prompt estimate
  (chars / 4) and never below 1024 tokens (the server then reports a real
  overflow). `stream: true` with `stream_options.include_usage`.
- Session affinity: `extra["session_key"]` (else `agent_name`) is hashed
  (SHA-1, stable across processes) to pick one of `base_urls`, and with
  `data_parallel_size > 1` and `routing: header` to send
  `X-data-parallel-rank: <hash % N>`, so each agent keeps hitting the same
  replica's prefix cache. N must equal the server's `--data-parallel-size` (vLLM rejects a
  higher rank with HTTP 400): `prepare()` counts the server's data-parallel engines from
  `/metrics` (`engine="k"` labels) and refuses to start when N is larger (a smaller N only
  warns); the local profiles read N from `$VBT_LLM_DP_SIZE`.

**Response decoding.** SSE chunks (keep-alive comments ignored, `data: [DONE]`
ends the stream; servers that ignore `stream` and answer JSON are handled too):
`delta.reasoning` (vLLM) or `delta.reasoning_content` (SGLang, llama.cpp) →
`on_thinking` and a `ThinkingBlock(provider=<name>, native={"field", "family"})`;
`delta.content` → `on_text`; `delta.tool_calls` accumulated by `index` (id and
name on the first chunk, argument fragments appended). Arguments that are not a
JSON object become `ToolCall(input={}, native={"invalid_arguments": raw,
"error": ...})`, so the runtime can ask the model to re-issue the call. If the
server returned no `tool_calls` but the text holds `<tool_call>{json}</tool_call>`
or Qwen XML `<tool_call><function=name><parameter=k>v</parameter>...` blocks
(server started without a tool-call parser) they are extracted (parameter values
typed by the tool schema) and a warning is logged once; `<think>...</think>` in
the content is split off the same way. Stop reasons: `abort` / `error` →
`RetryableProviderError` (checked first: an aborted stream may hold half-streamed calls with
truncated arguments, which must never become a `tool_use` turn); any tool calls → `tool_use`;
`stop` → `end_turn`; `length` → `max_tokens`, or `context_exceeded` when
prompt + completion reached the window or `max_tokens` had been cut to fit it;
`content_filter` → `refusal`. Usage: `cache_read = prompt_tokens_details.
cached_tokens` (needs `--enable-prompt-tokens-details`), `input = prompt_tokens
- cached`, `cache_write = 0`; cost is 0 unless `pricing` is set, so budget local
runs in tokens.

**Context window.** `context_window(model)` is the `context_window` option, else
`max_model_len` from `GET /v1/models` (read once before the first call, or by
`prepare()`), else None (the context manager then uses its default). An
overflow (vLLM HTTP 400 "This model's maximum context length is L tokens ... your
prompt contains N input tokens", SGLang "is longer than the model's context
length" / "exceeds the model's maximum context length", llama.cpp
`exceed_context_size_error`) is retried once inside `complete()` with
`max_tokens = L - N - 256` when `L - N >= 2048`; otherwise it raises
`ContextOverflowError` and the context manager compacts. A learned `L` is
remembered. Recent vLLM (observed on 0.30) stops tokenizing at `L - max_tokens + 1` tokens, so
its message says only "your prompt contains **at least** N input tokens" (or
"contains C characters (more than X characters, which is the upper bound for Y
input tokens)"), which is not the prompt size: the adapter then counts the
prompt with vLLM's `POST /tokenize` (same messages, tools and template kwargs)
and falls back to a chars/4 estimate when the server has no such endpoint
(verified against vLLM 0.30, docs/LOCAL_LLM_VERIFICATION.md).

**Errors.** Connection errors (with a "is the inference server running? `vbt
local serve ...`" hint), timeouts, dropped streams, error events mid-stream,
`finish_reason: abort`, HTTP 408/409/425/429/500/502/503/504/529 →
`RetryableProviderError` (honouring `Retry-After`). 404 → `ProviderError` listing
the served models; 401/403 → `ProviderError` naming `VBT_LLM_API_KEY`; template
exceptions ("Unexpected reasoning effort", "System message must be at the
beginning", "No user query found") are reported as adapter bugs; other 4xx →
`ProviderError`. If a server's request validation rejects the top-level
`reasoning_effort` field itself (a stricter enum than vLLM's), the request is
re-sent once with the equivalent `chat_template_kwargs.reasoning_effort` form,
and that form is kept for the provider's lifetime (`none` is never put in the
kwargs; thinking off is `enable_thinking: false`).

**Readiness and introspection.** `check_credentials()` only checks that a
base URL is configured (no network). `prepare(models=None, wait_s=None)` polls
`GET /health` (up to `wait_ready_s`), reads `/v1/models` and fails with the list
of served names if a requested model is missing. `server_info()` returns
`{provider, base_urls, family, served_model_name, models, max_model_len,
version}` and never raises; `list_models()` and `health()` are also available.
`max_concurrency` caps in-flight requests client-side (vLLM queues excess
requests rather than returning 429). Capabilities: `images` = `vision`,
`documents`, `web_search` and `server_context_management` false,
`replays_reasoning` and `tool_choice` true. There is no provider-native web
search; use a search backend (`vbt.tools.search_backends`).

## The Claude adapter (`anthropic_provider.py`)

Facts below come from the `claude-api` skill's model, pricing, migration,
prompt-caching and error-code references (cached 2026-09-25).

**Pricing.** `PRICES` holds per-model `{input, output, cache_read,
cache_write_5m, cache_write_1h}` in USD per million tokens. Cache writes cost
1.25× (5-minute TTL) and 2× (1-hour TTL) input; cache reads cost 0.1× input
except Claude Opus 5.5 (0.05×, $0.20) and Claude Fable 5.1 / Mythos 5.1
(0.025×, $0.25); Claude Mythos 5.1 is $10/$50. Server-side web search adds $10
per 1,000 searches. `usage.cache_creation` is split into 5-minute and 1-hour
writes and `usage.server_tool_use.web_search_requests` is priced. When
`usage.iterations` is present (server-side fallback, compaction), each
iteration is priced at its own model's rates and listed in
`ModelResponse.iterations`. Dated and platform IDs (`claude-x-20250929`,
`anthropic.claude-x-v1:0`, `claude-x@date`) resolve to the base model; an
unlisted model warns once and is priced as the closest family or, failing that,
the most expensive listed model so budgets stay conservative. Override or add
models with `provider.options.prices: {<model>: {input: ..., output: ...}}`
(missing keys default to the standard multipliers).

**Thinking.** Adaptive models (Fable 5/5.1, Mythos 5/5.1, Opus 5.5/5/4.8/4.7/4.6,
Sonnet 5.5/5/4.6) get `thinking: {type: adaptive, display: summarized}` and
`output_config.effort`. Fable, Mythos and Opus 5.5 cannot disable thinking, so
it is always sent. Older models (Sonnet 4.5, Haiku 4.5, Opus 4.5 and earlier
Claude 4.x) use a fixed `budget_tokens` from `models.<tier>.thinking_budget`
(default 8000, clamped below `max_tokens`); when tools are present the
`interleaved-thinking-2025-05-14` beta is sent so they can think between tool
calls. Effort is never sent to models that reject it. The paper profile pins
budgets explicitly (orchestrator 32000, scientist 16000, bulk 8000; Haiku
support agents without thinking — see the comments in
`configs/profiles/paper.yaml`). `configs/profiles/upstream-web.yaml` reproduces
the upstream web app instead: CSO effort `xhigh`, specialists `high`,
single-cell analyst `max`.

**Refusal fallback.** For the families `claude-fable-5`, `claude-opus-5` and
`claude-sonnet-5-5` (any member, e.g. Opus 5.5) the adapter sends
`fallbacks: "default"` with the `server-side-fallback-2026-07-01` beta
(`provider.options.refusal_fallback`, `fallback_families`). `served_model` and
`fallback_used` report what happened; blocks of a declined partial before the
final `fallback` marker are dropped except their text, so a declined tool call
is never executed or replayed.

**Prompt caching.** Top-level automatic caching (`cache_control` on the request)
caches the conversation tail. A `SystemSegment` list is sent as system text
blocks with `cache_control` on the last segment marked `cache=True`, so the
tools + stable system prefix is shared across delegations and bulk items
(`provider.options.system_cache_ttl: 1h` for long human pauses).

**Images and documents.** Tool-result parts are sent as `image` (base64 source)
and `document` (base64 PDF) blocks inside `tool_result`.

**Errors.** `RateLimitError`, `InternalServerError`, overloaded (529),
connection/timeout errors, transport errors and `event: error` frames after
streaming started (`overloaded_error`, `api_error`) become
`RetryableProviderError` with `retry_after` from `retry-after(-ms)`. A 400
"prompt is too long" (or another context-window message) and 413 become
`ContextOverflowError`; stop reason `model_context_window_exceeded` maps to
`StopReason.CONTEXT_EXCEEDED`. Authentication, permission and not-found stay
`ProviderError`.

**Streaming.** With `on_text`/`on_thinking` the adapter iterates raw stream
events and dispatches `text_delta` and `thinking_delta`.

**Endpoint.** `provider.options` is one namespace that profiles deep-merge onto the local
default, so the Anthropic factory ignores the OpenAI-compatible adapter's options (`base_url`,
`base_urls`, `served_model_name`, `family`, ...): an inherited `base_url` would otherwise send
every request, with `ANTHROPIC_API_KEY` in `x-api-key`, to the local server's address. Another
Anthropic endpoint (a gateway) is `provider.options.anthropic_base_url` (or the SDK's
`ANTHROPIC_BASE_URL`).

**Credentials.** `check_credentials()` treats a blank or whitespace
`ANTHROPIC_API_KEY` (e.g. an empty `.env` entry) as missing unless
`ANTHROPIC_AUTH_TOKEN` is set, and otherwise accepts an API key, auth token,
workload-identity variables or an `ant auth login` profile.

**Context windows.** `context_window(model)` returns 1,000,000 for the current
Fable/Mythos/Opus/Sonnet models and 200,000 for Haiku 4.5; Sonnet 4.5 is not in
the table, so `configs/profiles/paper.yaml` sets `context_window_tokens: 200000`.

**Server-side context editing** (optional; see above): `context_management:
{edits: [{type: clear_tool_uses_20250919}]}` with the
`context-management-2025-06-27` beta, enabled per request through
`extra["context_management"]` or for all requests with
`provider.options.context_editing: true` (or a dict of edit parameters).
