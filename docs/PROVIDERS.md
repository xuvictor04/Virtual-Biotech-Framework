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
- `ToolSpec(name, description, input_schema)` — JSON Schema, passed through from MCP.
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
  history_bound_thinking)` from `provider.capabilities(model=None)`.
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
    async def web_search(self, query, *, max_results=8,
                         allowed_domains=None, blocked_domains=None) -> dict
    def supports_web_search(self) -> bool
```

`on_text` / `on_thinking` receive streamed text and reasoning deltas; providers
without reasoning never call `on_thinking`. `web_search` returns
`{"results": [...], "summary": str, "cost_usd": float}`.

`ModelSettings.extra` carries harness hints; adapters consume the keys they know
and never forward any `extra` key to a vendor API. Keys in use:
`agent_name` (set by the roster; the mock uses it to pick a script queue),
`thinking_budget` (budget-thinking models), `context_window_tokens` (overrides
the provider's window table for compaction), `context_management` (request
server-side context editing) and `prompt_cache` (False disables caching for a
one-off request such as a compaction summary).

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
else 200,000. The prompt size is read from the last response
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

Config (in-code defaults, also listed in `configs/default.yaml`): `context.enabled: true`,
`soft_ratio: 0.70`, `hard_ratio: 0.85`, `keep_recent_calls: 6`, `min_clear_chars: 2000`,
`min_clear_fraction: 0.05`, `summary_max_tokens: 4000`, `server_side: false`,
`default_window_tokens: 200000`; optional `models.<tier>.context_window_tokens`.
Compactions and retries are traced (`compaction`, `provider_retry`), counted in each turn
record, and shown as notices by `vbt chat` and `vbt web`.

## Choosing a model at the command line

`--model` (chat, run, replay, scenario) sets the orchestrator, scientist and bulk tiers. It
accepts a `model_aliases` label (`configs/default.yaml`: `opus`, `sonnet`, `haiku`, `paper`),
a model id already configured in a tier, or an id matching `provider.model_pattern`
(default `^claude-[a-z0-9.-]+$` for the Anthropic provider; no check for others). Anything
else exits with code 2 and lists the aliases. The chosen models, with the effective
thinking/effort per tier, are pinned in the run's `inputs/config.json`.

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
