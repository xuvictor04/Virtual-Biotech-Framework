# Local LLM verification against a real inference engine (CPU)

The OpenAI-compatible adapter (`src/vbt/providers/openai_compat.py`, provider
names `vllm` / `sglang` / `openai_compat` / `llamacpp`) was written against the
vLLM documentation and tested offline against fake servers. This report records
what happened when it talked to a **real** inference server. The sandbox has no
GPU, so the engine ran on CPU with a tiny model from the same family as the
production model. Tiny models answer poorly; the probes test the plumbing (what
the server accepts, what it returns, how the adapter maps it), not quality.

Date: 2026-10-06. Live probes: `tests/test_live_local.py` (skipped unless
`VBT_LIVE_LOCAL_URL` is set). Server launcher: `scripts/dev/cpu_server.sh`.

## Summary

- **Engine**: vLLM **0.30.0** (the community `vllm-cpu` wheel from PyPI, a CPU
  build of vLLM; `torch 2.13.0+cpu`, `transformers 5.18.0`, Python 3.12).
- **Model**: `Qwen/Qwen3.5-0.8B` (0.87B, BF16), served as `qwen3.5-0.8b` with
  `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser
  qwen3_coder --language-model-only`. Qwen3.5 has the same hybrid architecture
  as Qwen3.8 (Gated DeltaNet linear attention + gated full attention at 3:1,
  MTP head, vision tower dropped by `--language-model-only`), the same
  `qwen3_coder` XML tool-call format and `qwen3` reasoning parser, and a chat
  template with the same structure and error strings. The adapter resolved it to
  the `qwen3_6` family (Qwen3.5/3.6 dialect: `enable_thinking` toggle,
  `preserve_thinking: true`) from the served name, with no configuration.
- **Result**: all 18 live probes pass against vLLM 0.30, and a full CSO turn
  with one delegation ran through `vbt run` (see "Harness turn"). Every request
  field the production Qwen3.8 dialect sends (`reasoning_effort`
  `xhigh`/`medium`/`none`, `thinking_token_budget`, `top_k`, `min_p`,
  `repetition_penalty`, `presence_penalty`, `chat_template_kwargs`,
  `parallel_tool_calls`, `strict` tools, named `tool_choice`, `stream_options`)
  passed vLLM's request validation.
- **Second engine**: llama.cpp `llama-server` (built from source) with a Q8_0
  GGUF of the same model; see "llama.cpp".
- **Harness version**: the live runs used the harness from before the L2
  integration. L2 made the local provider the default and added several runtime
  behaviours: forced final `submit_result`, `tool_choice: "none"` on no-tool
  final calls, argument checks before tools run, the empty-reply nudge, a
  `session_key` per invocation and token budgets. These have offline tests
  (`tests/test_local_integration.py`) but have not been run against a live
  server. The adapter-level probes above (named `tool_choice`, `strict`, `none`,
  `X-data-parallel-rank`) cover the requests those behaviours send.
- **Bugs found and fixed** (adapter, with offline regression tests in
  `tests/test_openai_compat.py`):
  1. Context-overflow recovery read vLLM's prompt-size *lower bound* as the
     prompt size, so "retry with a smaller `max_tokens`" could never succeed
     against vLLM 0.30 (observed; recent vLLM bounds prompt tokenization). The prompt is now counted with `POST /tokenize`.
  2. llama.cpp's window was not discovered: its `/v1/models` has no
     `max_model_len`; the per-slot window is `meta.n_ctx`.
  3. With `tool_choice: "none"`, llama.cpp returns the model's attempted call as
     text and the adapter's text fallback turned it into a tool call anyway.
  4. With thinking off, a stray `</think>` in the answer moved the answer's
     first sentence into a hidden reasoning block.
  5. An out-of-range `X-data-parallel-rank` (provider `data_parallel_size`
     larger than the server's) now names the option instead of a bare HTTP 400.
- **Finding for the GPU deployment**: prefix-cache reuse in a growing agent
  loop was partial and irregular on vLLM 0.30's hybrid-model ("align") cache,
  although the harness's requests are append-only; measure it on the GPU
  (see "Prefix caching in agent loops").

## Environment

| | |
|---|---|
| Machine | 4 vCPU x86_64 (AVX-512, VNNI; no AMX), 15 GB RAM, no GPU, Ubuntu 24.04 |
| Engine | `vllm-cpu==0.30.0` in an isolated venv under the session scratchpad (not in the repo) |
| Model | `Qwen/Qwen3.5-0.8B` (HF, `Qwen3_5ForConditionalGeneration`, BF16 safetensors 1.75 GB) |
| Window | `--max-model-len 32768`; CPU KV cache 4 GiB = 308,471 tokens ("Maximum concurrency for 32,768 tokens per request: 9.41x") |
| Harness | this repository, system Python 3.11, `httpx 0.28.1` |

## Commands

```bash
# 1. engine in its own venv (uv; plain venv + pip also works). Pulls torch 2.13.0+cpu from the PyTorch CPU index.
scripts/dev/cpu_server.sh --install --venv "$SCRATCH/venv-vllm"

# 2. serve (foreground; add & or a second terminal)
HF_HOME="$SCRATCH/hf" scripts/dev/cpu_server.sh --venv "$SCRATCH/venv-vllm" --model Qwen/Qwen3.5-0.8B
#   == VLLM_CPU_KVCACHE_SPACE=4 VLLM_CPU_OMP_THREADS_BIND=auto [LD_LIBRARY_PATH=<venv>/libnuma/...] \
#      <venv>/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-0.8B \
#      --served-model-name qwen3.5-0.8b --host 127.0.0.1 --port 8011 --max-model-len 32768 \
#      --dtype bfloat16 --max-num-batched-tokens 2048 --enable-prefix-caching \
#      --enable-prompt-tokens-details --reasoning-parser qwen3 --enable-auto-tool-choice \
#      --tool-call-parser qwen3_coder --language-model-only

# 3. probes (from the harness environment)
VBT_LIVE_LOCAL_URL=http://127.0.0.1:8011/v1 VBT_LIVE_LOCAL_MODEL=qwen3.5-0.8b \
  VBT_LIVE_LOCAL_LOG=/tmp/live.jsonl python3 -m pytest -v tests/test_live_local.py
#   VBT_LIVE_LOCAL_HARNESS=1 adds one CSO turn through the runtime;
#   VBT_LIVE_LOCAL_PROVIDER=llamacpp|sglang selects the registered provider name.

# 4. llama.cpp instead (llama-server binary, GGUF): built from the llama.cpp sources vendored in the
#    llama-cpp-python 0.3.36 sdist (cmake -G Ninja -DGGML_NATIVE=ON -DLLAMA_CURL=OFF; target llama-server)
LLAMA_SERVER=/path/to/llama-server scripts/dev/cpu_server.sh --engine llamacpp \
  --model Qwen3.5-0.8B-Q8_0.gguf --served-name qwen3.5-0.8b --port 8012
#   == llama-server -m Qwen3.5-0.8B-Q8_0.gguf --alias qwen3.5-0.8b --host 127.0.0.1 --port 8012 -c 32768 \
#      --jinja --reasoning-format deepseek -np 1 --metrics
VBT_LIVE_LOCAL_URL=http://127.0.0.1:8012/v1 VBT_LIVE_LOCAL_MODEL=qwen3.5-0.8b VBT_LIVE_LOCAL_PROVIDER=llamacpp \
  VBT_LIVE_LOCAL_HARNESS=1 python3 -m pytest -v tests/test_live_local.py
```

The same probes run unchanged against a GPU server: point `VBT_LIVE_LOCAL_URL`
at it and set `VBT_LIVE_LOCAL_MODEL=qwen3.8-27b` (the family then resolves to
`qwen3_8`).

### Engine setup problems hit (and how the script handles them)

1. **`vllm` console script is broken in the `vllm-cpu` wheel**: it calls
   `importlib.metadata.version("vllm")`, but the distribution is named
   `vllm-cpu`, so `vllm serve` dies with `PackageNotFoundError: No package
   metadata was found for vllm`. `python -m vllm.entrypoints.openai.api_server`
   takes the same arguments (it prints a deprecation warning) and is what the
   script runs.
2. **`libnuma.so.1` missing**: vLLM's CPU kernels (`_C_AVX512`, `_C_AVX2`) link
   it. Without it the server logs `Failed to import from vllm._C_AVX512:
   ImportError('libnuma.so.1: ...')` and the engine then dies with
   `AttributeError: '_OpNamespace' '_C' object has no attribute
   'init_cpu_memory_env'`. Fix: `apt-get install libnuma1`; without root,
   `--install` unpacks the `libnuma1` .deb into `<venv>/libnuma` and the script
   adds it to `LD_LIBRARY_PATH`.
3. **Startup time**: about 3.5 minutes the first time (weights, CPU KV
   allocation, `torch.compile` warm-up: "init engine (profile, create kv cache,
   warmup model) took 119.71 s" on the second start), about 3 minutes after.
   `prepare(wait_s=...)` / `wait_ready_s` cover this.
4. **One long prefill blocks the CPU engine**: each prompt was prefilled in a
   single engine step, also with `--max-num-batched-tokens 2048` (the engine
   statistics report a whole 12.5K-token prompt in one interval). A 26K-token
   prompt ran for more than 10 minutes: the adapter's 600 s inter-chunk read
   timeout fired (correctly, as a `RetryableProviderError`), but the server
   kept computing the abandoned request and answered nothing else meanwhile, so
   it was restarted. On CPU, keep probe prompts small and `read_timeout_s` above
   the longest prefill (`live-cpu` profile below: 1800 s). GPU servers prefill
   orders of magnitude faster; the production `read_timeout_s: 900` stands.

Throughput on this machine (for orientation only; other jobs shared the four
cores): prefill of long prompts ~35-40 tokens/s (12.5K tokens in about 6
minutes), decode 4-7 tokens/s with vLLM; llama.cpp (Q8_0 GGUF) prefilled ~90
tokens/s and decoded ~11 tokens/s.

## Probe results

Excerpts are verbatim from the probe log (`VBT_LIVE_LOCAL_LOG`) or from raw
HTTP requests made without the adapter.

| # | Probe | Result |
|---|---|---|
| 1 | `prepare()`, `health()`, `server_info()`, `context_window()` | PASS. `/health` 200; `/v1/models` lists `qwen3.5-0.8b` with `max_model_len: 32768` and `root` = the checkpoint path; `/version` = `0.30.0`; family `qwen3_6`; `prepare(models=["no-such-model-xyz"])` raises "not served ... served: ['qwen3.5-0.8b']". |
| 2 | Plain completion, thinking off | PASS. Sent `chat_template_kwargs {"preserve_thinking": true, "enable_thinking": false}`, card sampling with the `extra['sampling']` override (`temperature 0, top_p 1`, `top_k 20`, `min_p 0`, `presence_penalty 1.5`, `repetition_penalty 1`). Reply `pong`, `end_turn`, usage `input 29 / output 2`, no reasoning block, cost 0. |
| 3 | Streaming with reasoning | PASS. 383 `delta.reasoning` chunks then 3 content chunks; `ThinkingBlock(provider="vllm", native={"field": "reasoning", "family": "qwen3_6"})`; the joined `on_thinking` deltas equal the block text; answer `391`. `thinking_token_budget: 384` was enforced: 389 output tokens in total (reasoning cut mid-sentence, then the answer). |
| 4 | Single tool call (`tool_choice: auto`, qwen3_coder parser) | PASS on the first greedy attempt: `get_weather({"city": "Paris"})`, `tool_use`, id `chatcmpl-tool-...`. |
| 5 | Parallel tool calls | PASS: three calls (Paris, Tokyo, Lima) in one turn, distinct ids, streamed as `index` 0..2 deltas. |
| 6 | Tool-result round trip | PASS. Results were given in reverse order; the adapter re-ordered them to the call order: roles `system, user, assistant, tool, tool, user`, `tool_call_id`s in call order, assistant `content: null` with JSON-object argument strings. The template accepted it; answer "Paris is currently 17 C with light rain, while Tokyo is 24 C and sunny." |
| 7 | Strict schema + forced named `tool_choice` | PASS. `{"type": "function", "function": {"name": "submit_result"}}` with `strict: true` on `submit_result` only (enum, regex `^NCT[0-9]{8}$`, bounded number, `minItems`/`maxItems` array, `anyOf` with null, `additionalProperties: false`): `{"nct_id": "NCT01234567", "outcome": "success", "confidence": 0.95, "evidence": [...], "notes": "..."}`, schema-valid. |
| 8 | `tool_choice: required` / `none` | PASS. `required` forced a `get_weather` call for "Hello! How are you?". `none`: no tool call, but see observation 2. |
| 9 | Prefix cache / usage | PASS. Three requests sharing a 7,059-token system prompt: `cache_read_tokens` 0, 7040, 7040 (`input_tokens` 7059, 19, 20). Prefix caching works on the hybrid Gated-DeltaNet model (vLLM sets "Mamba cache mode ... 'align'"; hits land on block boundaries, hence 7040 of 7059), with and without `--max-num-batched-tokens 2048`. `cache_write_tokens` is always 0. |
| 9b | Prefix cache in a growing agent loop (3.4K-token system, call -> results -> call) | PASS (at least one follow-up hit): cached 0 of 3416, then **0 of 3510**, then 3200 of 3538. See "Prefix caching in agent loops". |
| 10 | Overflow without room | PASS: `ContextOverflowError: vllm: request does not fit the model's context window (limit 32768, prompt 142364 counted via /tokenize): HTTP 400: This model's maximum context length is 32768 tokens. However, you requested 64 output tokens and your prompt contains at least 32705 input tokens, ...`; the window 32768 was learned. |
| 11 | Overflow with room, retried | PASS after the fix (failed before it, see below). `max_tokens` 32768 (and 32512) -> one retry with 30193 = 32768 - 2319 - 256; the retried request's `prompt_tokens` was 2319, exactly the `/tokenize` count. |
| 12 | Unknown model | PASS: "model 'no-such-model-xyz' not found (HTTP 404: The model `no-such-model-xyz` does not exist.); served models: ['qwen3.5-0.8b']. ..." |
| 13 | Production Qwen3.8 request shape (`family: qwen3_8` forced) | PASS for `xhigh`, `medium` and thinking off: vLLM 0.30 accepted top-level `reasoning_effort` `xhigh`/`medium`/`none` (its enum is `none, minimal, low, medium, high, xhigh, max`), `thinking_token_budget`, `chat_template_kwargs {"enable_thinking": false}` and the card sampling; no fallback to the `chat_template_kwargs` effort form was needed. vLLM passes `reasoning_effort` to the template as a kwarg and sets `enable_thinking = effort != "none"` unless the request sets it. |
| 14 | Concurrency with `max_concurrency: 2` | PASS: four concurrent requests with distinct `session_key`s all completed. |

Raw excerpts (no adapter):

```text
# streamed tool calls (qwen3_coder parser): id/type/name first, then argument fragments, by index
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"chatcmpl-tool-82672f40890f8f33","type":"function","index":0,"function":{"name":"get_weather"}}]},"finish_reason":null}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\": \""}}]},"finish_reason":null}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Paris"}}]},"finish_reason":null}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]},"finish_reason":null}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"chatcmpl-tool-8e3b23c3ab2694f1","type":"function","index":1,"function":{"name":"get_weather"}}]},"finish_reason":null}]}
...
data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}
data: {"choices":[],"usage":{"prompt_tokens":287,"total_tokens":340,"completion_tokens":53,"prompt_tokens_details":{"cached_tokens":0,"created_cache_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}}}
data: [DONE]

# streamed reasoning (thinking_token_budget 64 -> reasoning_tokens 63)
data: {"choices":[{"index":0,"delta":{"reasoning":"Thinking"},"finish_reason":null}]}
data: {"choices":[{"index":0,"delta":{"reasoning":" Process"},"finish_reason":null}]}
...
data: {"choices":[],"usage":{"prompt_tokens":25,"total_tokens":325,"completion_tokens":300,"prompt_tokens_details":{"cached_tokens":0,"created_cache_tokens":0},"completion_tokens_details":{"reasoning_tokens":63}}}

# template errors the adapter must never trigger (it raises ProviderError before sending instead)
HTTP 400 {"error": {"message": "No user query found in messages.", "type": "BadRequestError", "code": 400}}
HTTP 400 {"error": {"message": "System message must be at the beginning.", "type": "BadRequestError", "code": 400}}

# DP rank header on a server without data parallelism
X-data-parallel-rank: 0 -> HTTP 200
X-data-parallel-rank: 1 -> HTTP 400 {"error":{"message":"data_parallel_rank 1 is out of range [0, 1).","type":"BadRequestError","code":400}}

# POST /tokenize (chat form; exact and unbounded)
{"model": "qwen3.5-0.8b", "messages": [...], "add_generation_prompt": true, "tools": [...],
 "chat_template_kwargs": {"enable_thinking": false}} -> {"count": 276, "max_model_len": 32768}
```

## Bug found and fixed: overflow recovery used a lower bound as the prompt size

The digest and the adapter assumed vLLM's overflow message reports the prompt
size ("your prompt contains N input tokens"), and retried once with
`max_tokens = L - N - 256`. vLLM 0.30 instead tokenizes at most
`L - max_tokens + 1` prompt tokens before rejecting (to bound the work) and
reports that bound, or rejects on a character pre-check before tokenizing at all:

```text
max_tokens=32768: "... you requested 32768 output tokens and your prompt contains 31800 characters (more than 0
                   characters, which is the upper bound for 0 input tokens). ..." (parameter=input_text)
max_tokens=32512: "... you requested 32512 output tokens and your prompt contains at least 257 input tokens,
                   for a total of at least 32769 tokens. ..." (parameter=input_tokens, value=257)
max_tokens=64:    "... your prompt contains at least 32705 input tokens ..." (the prompt had 142,364)
```

The "at least" number is always `L - max_tokens + 1`, so the computed room was
`max_tokens - 1` and the retry (`max_tokens - 257`) overflowed again; for the
character form the regex picked "0 input tokens" as the prompt size and retried
with `max_tokens = L - 256`. Either way an overflow with room never recovered,
and the `ContextOverflowError` message showed a wrong prompt size ("prompt
257" for a 26K-token prompt).

Fix (`openai_compat.py`): `overflow_info()` separates an exact prompt size from
a lower bound (`parse_overflow()` now returns `prompt=None` for the bound and
character forms); when only a bound is known, `complete()` counts the prompt
exactly with vLLM's `POST /tokenize` (same messages, tools and
`chat_template_kwargs`, plus the `reasoning_effort`/`enable_thinking` vLLM would
pass to the template), falls back to `max(bound, chars/4 estimate)` when the
server has no such endpoint, and retries once. Offline regression tests in
`tests/test_openai_compat.py` use the verbatim vLLM 0.30 messages
(`test_vllm_lower_bound_overflow_messages_are_not_prompt_sizes`,
`test_lower_bound_overflow_counts_the_prompt_via_tokenize_then_retries`, ...).

Related: the client-side `max_tokens` clamp estimates the prompt as chars/4,
which is far off for ID- and number-heavy text: the probe text `w0 w1 w2 ...`
was 31,655 characters but 26,207 tokens (1.2 characters per token). Gene, trial
and variant identifiers behave similarly, so the server's overflow error (now
handled correctly) remains the real safety net; the clamp only avoids obviously
oversized requests.

## Other observations

1. **Thinking must be switched off explicitly.** Without
   `enable_thinking: false`, vLLM's `qwen3` reasoning parser treats the whole
   answer as reasoning when the template's default is non-thinking (Qwen3.5
   small models): `"content": null, "reasoning": "Hello! How can"`. The adapter
   always sends the toggle for the `qwen3_6`/`qwen3` families and
   `reasoning_effort: "none"` + `enable_thinking: false` for `qwen3_8`, so this
   did not affect it; a `generic` family on such a model would.
2. **`tool_choice: "none"` can yield an empty turn.** With tools present and
   `tool_choice: "none"`, the model still wrote a tool call (26 tokens) and vLLM
   0.30 returned `content: null` with `finish_reason: stop`. Since L2 the
   runtime sends `tool_choice: "none"` on calls where no tool may run (turn
   limit, budget grace, tools disabled for a clarification call). An empty
   reply there ends the agent with the harness's placeholder text ("Turn limit
   reached before a final report was written."), the same outcome as an
   unexecuted call before; on llama.cpp the attempted call stays in the report
   as text (row "tool_choice none" below). Ordinary calls use `auto`, and the
   bulk final call uses a forced named choice. This was not re-probed live
   after L2.
3. **Unknown Anthropic options**: running `--profile local-h100` on the then
   (still Anthropic) `configs/default.yaml` logged `OpenAICompatProvider: ignoring
   unknown provider options ['max_retries', 'prompt_caching',
   'refusal_fallback', 'web_search_model']` (profile deep-merge; harmless). L2
   made the local provider the default, and the warning no longer appears.
4. vLLM 0.30 also reports `prompt_tokens_details.created_cache_tokens` and
   `completion_tokens_details.reasoning_tokens`; the adapter does not need them
   (cache writes are free locally; reasoning is counted in output tokens).

## Harness turn (vLLM, through `vbt run`)

The full harness (upstream CSO prompt, real tool registry, Task delegation,
runtime loop, run directory) ran one turn against the CPU server, layered on
the shipped `local-h100` profile with a small override profile:

```yaml
# live-cpu.yaml (passed as a second --profile)
provider:
  name: vllm
  options: {base_url: http://127.0.0.1:8011/v1, served_model_name: qwen3.5-0.8b, family: auto,
            read_timeout_s: 1800, max_concurrency: 2}
models:   # every tier: {model: qwen3.5-0.8b, thinking: false, max_tokens: 1024, context_window_tokens: 32768}
  orchestrator: {model: qwen3.5-0.8b, effort: medium, thinking: false, max_tokens: 1024, thinking_budget: null,
                 context_window_tokens: 32768}
  # scientist / support / bulk likewise
limits: {max_cso_turns: 6, max_specialist_turns: 4, max_parallel_agents: 1}
orchestration: {strategic_orientation: false, enforce_review: false}
preflight: {skip: true}
web: {enabled: false}
context: {default_window_tokens: 32768}
```

```bash
vbt --profile local-h100 --profile live-cpu.yaml --no-mcp --runs-dir "$SCRATCH/runs" -v \
  run --events ndjson "Delegate to exactly one specialist: ask the genomics-analyst for one sentence on whether \
PCSK9 is a genetically supported target for LDL cholesterol. Then summarise its answer in one sentence."
```

What happened (from `logs/trace.jsonl`; 24 minutes of wall time on CPU):

| Call | Agent | Prompt tokens (cached) | Output | Result |
|---|---|---|---|---|
| 1 | cso | 12,568 (0) | 143 | text + `Task(subagent_type="genomics-analyst", description="Evaluate PCSK9 genetic evidence for LDL cholesterol", prompt=...)`, parsed by `qwen3_coder` from the stream |
| 2-6 | genomics-analyst | 6,365 (0), 6,573 (0), 6,750 (6,400), 6,932 (6,400), 7,127 (0) | 157-188 each | one `Grep` call per turn ("(no matches)": no MCP servers or data); stopped by `max_specialist_turns: 4` (`status: turn_limit`) |
| 7 | cso | 12,895 (0) | 112 | `end_turn`, final reply |

The turn completed (`turn_end status: completed`, cost 0.0 USD); `vbt run`
exited 1 because the run's verification was not COMPLETE (no reviewer, no
claims), which is the strict default of `run`, not a provider failure. The
0.8B model's conclusion ("PCSK9 is not a genetically supported target") is
wrong, as expected from a tiny model without data tools; the plumbing (system
prompt + 10 CSO tools + 13 specialist tools, streamed text and tool calls,
delegation, tool results, turn limit, final synthesis) worked end to end.

## Prefix caching in agent loops

The cached-token column above shows that most calls re-prefilled their whole
prompt although each agent's requests were append-only. That was checked by
replaying the same turn with canned model answers and diffing consecutive
request bodies per agent: every request's `messages` extend the previous one
and the `tools` are identical. Controlled raw-HTTP loops on the same server:

```text
3.2K-token prompt:  A 0 cached -> B (A + assistant + tool) 3200 -> A again 3200 -> C (B + assistant + user) 3200
6.5K-token prompt:  A 2560     -> B 2560 (not 6400)       -> A again 6400 -> C 6400 -> new user question 6400
live probe 9b:      0 of 3416  -> 0 of 3510               -> 3200 of 3538
```

vLLM caches the hybrid model's recurrent (Gated DeltaNet) state only at
block-aligned positions ("Mamba cache mode is set to 'align'"; 640-token blocks
here), so reuse is partial and depends on which aligned states earlier requests
left behind; on this build a follow-up often re-used nothing. llama.cpp, which
keeps the slot's full KV cache, re-used the whole previous prompt on every
follow-up (3412 of 3496). Two further effects for the GPU deployment:

- Qwen3.5's template renders reasoning only for assistant turns after the last
  real user message, so a harness reminder sent as user text re-renders the
  earlier assistant turns and invalidates the prefix from the first of them
  (call 6 above follows such a message). The Qwen3.6/3.8 templates keep
  reasoning with `preserve_thinking: true` (the template default for Qwen3.8),
  which the digest says keeps prompts append-only; not verified here.
- The digest's capacity plan assumes high prefix reuse. `vbt local check`
  currently probes an identical repeated prefix only; on the GPU server also
  run probe 9b (or a longer loop) and read `cached_tokens`. vLLM 0.31 lists
  several hybrid prefix-caching fixes (#59175, #59146, #58368).

## llama.cpp (`llama-server`, provider name `llamacpp`)

Second engine, same model: `unsloth/Qwen3.5-0.8B-GGUF` `Qwen3.5-0.8B-Q8_0.gguf`
served by `llama-server` 0.5.0-dev (commit `0c1e570`, the llama.cpp vendored in
the `llama-cpp-python` 0.3.36 sdist, built with CMake/GCC 13; GitHub release
downloads were not reachable from the sandbox), `--jinja --reasoning-format
deepseek`. Starts in seconds; prefill ~90 tokens/s (12,412 tokens in 141 s),
decode ~11 tokens/s.

`VBT_LIVE_LOCAL_PROVIDER=llamacpp`: **15 passed, 4 skipped** (unknown-model
404 and the Qwen3.8 request-shape probes are vLLM/SGLang-specific), including
the harness CSO turn (`VBT_LIVE_LOCAL_HARNESS=1`, 4 minutes: CSO -> `Task` ->
genomics-analyst with `TodoWrite`/`Skill`/`Glob`/`Bash`/
`mcp__provenance__list_artifacts` calls -> turn limit -> CSO `completed`).
Differences from vLLM, all handled or documented:

| | llama.cpp behaviour | Adapter |
|---|---|---|
| Window | `/v1/models` has no `max_model_len`; `meta.n_ctx` is the per-slot window (`-c 32768 -np 2` gives 16384), `meta.n_ctx_train` the model's | **fixed**: reads `meta.n_ctx` (was: window unknown until an overflow) |
| Reasoning | streamed as `delta.reasoning_content` | read; `ThinkingBlock.native.field = "reasoning_content"`, replayed as `reasoning_content` |
| `thinking_token_budget` | ignored (only the server flag `--reasoning-budget`): 768 of 768 tokens were reasoning, no answer | none needed; set `--reasoning-budget` on the server |
| `tool_choice: "required"` | not enforced for this template: plain text answer | the live probe records it instead of asserting |
| `tool_choice: "none"` | the model's attempted call comes back as content `<tool_call>\n<function=get_weather>...` | **fixed**: no text-call extraction when `tool_choice` is `none` |
| Stray `</think>` with thinking off | content `"I'm doing well ... \n</think>\n\nI'm doing well ..."` | **fixed**: kept as answer text when thinking is off |
| Forced named `tool_choice` + `strict` | honoured (schema-valid `submit_result`) | as vLLM |
| Overflow | exact: `request (142364 tokens) exceeds the available context size (32768 tokens), try increasing it` (`exceed_context_size_error`, `n_prompt_tokens`, `n_ctx`) | `ContextOverflowError (limit 32768, prompt 142364)` |
| `prompt + max_tokens > n_ctx` | not rejected (generation stops when the window is full), so there is nothing to retry | the probe asserts a single successful request |
| Prefix cache | `prompt_tokens_details.cached_tokens`; whole previous prompt re-used in agent loops (3412 of 3496) | `cache_read_tokens` |
| `/version`, unknown model | no `/version`; any model name is accepted | `server_info().version` is None; 404 probe skipped |

## What this proves, and what it does not, about the GPU deployment

Proven against a real vLLM OpenAI server and the real Qwen3.5 chat template:

- The adapter's request body passes vLLM's validation, including every field
  of the Qwen3.8 dialect (`reasoning_effort` `xhigh`/`medium`/`none`,
  `thinking_token_budget`, card sampling extras, `strict` tools, named
  `tool_choice`, `parallel_tool_calls`, `stream_options`).
- Message encoding satisfies the Qwen-family template: one system message at
  index 0, assistant `content: null` + `tool_calls` with JSON-object argument
  strings, `role: tool` messages in call order followed by a user message,
  reasoning replay field accepted.
- Streaming decode of `delta.reasoning`, content and index-keyed tool-call
  deltas from the `qwen3_coder` parser; parallel calls; usage with
  `cached_tokens` (prefix caching on a hybrid Gated-DeltaNet model);
  `thinking_token_budget` is enforced server-side.
- Forced tool calls with a strict schema (enum, regex pattern, numeric bounds,
  array bounds, `anyOf` null) come back schema-valid through xgrammar.
- Error mapping for overflow (after the fix), unknown model (404), template
  errors and DP rank; model discovery (`/v1/models` `max_model_len`, `/version`).
- The whole harness loop (CSO prompt and tools, Task delegation, specialist tool
  loop, turn limits, final synthesis) runs on the adapter against both engines.
- The same adapter works with llama.cpp's `llama-server` (`reasoning_content`,
  exact overflow errors, `meta.n_ctx` window) after the fixes above.

Not proven here:

- **vLLM 0.31.0** (`--tool-strict-level function`) and the GPU code paths:
  CUDA graphs, FP8 / NVFP4 / INT4 kernels, FP8 KV cache, MTP speculative
  decoding, chunked prefill at 262K, DP > 1 routing. The CPU build is 0.30.0.
- The **Qwen3.8 template itself**: its effort instruction lines for
  `xhigh`/`low`, the "Unexpected reasoning effort" error for `high`/`max`, and
  its `preserve_thinking` default. Qwen3.5's template ignores
  `reasoning_effort`; only vLLM's validation of the field was exercised.
- Model quality: tool-call reliability, malformed-call rate, reasoning loops,
  long-context recall of Qwen3.8-27B. A 0.8B model happened to call tools
  correctly with greedy sampling here; that says nothing about the 27B model.
- The full `TrialAnnotation` schema in xgrammar, capacity, throughput and
  latency. `vbt local check` (configs/local_models.yaml profiles) measures
  those on the GPU server; run it, then these probes with
  `VBT_LIVE_LOCAL_URL` pointing at it.
- SGLang: no CPU run was attempted (vLLM worked; llama.cpp was the second engine). Its
  `reasoning_content` field and overflow wording are covered by offline tests only.
