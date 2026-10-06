# Running the harness on a local model

The harness talks to a local **vLLM 0.31** server through its OpenAI-compatible API (provider
`vllm`). The default model is **Qwen3.8-27B** (Apache-2.0) on **one 80-96 GB NVIDIA GPU**. Other
hardware has its own serving profile. Every profile is defined in
[`configs/local_models.yaml`](../../configs/local_models.yaml), and the harness settings that go
with it are in `configs/profiles/local-*.yaml`.

| File | Purpose |
|---|---|
| `docker-compose.yml` | Two kinds of service: `vllm-<profile>` (one per GPU class, chosen with a compose profile) and `searxng` (web search) |
| `serve_vllm.sh` | Renders or starts the bare-metal `vllm serve` command for a profile. Needs only Python and PyYAML |
| `searxng/settings.yml` | SearxNG configuration for the `WebSearch` tool (see [docs/WEB_SEARCH.md](../../docs/WEB_SEARCH.md)) |

## Quick start

```bash
vbt local profiles --detect                 # read nvidia-smi and recommend a serving profile + docker tag
docker compose -f deploy/local/docker-compose.yml --profile h100 up -d    # vLLM + SearxNG
#   or bare metal:  vbt local serve --profile h100      (deploy/local/serve_vllm.sh --profile h100)
export VBT_LLM_BASE_URL=http://localhost:8000/v1 SEARXNG_URL=http://localhost:8888
vbt --profile local-h100 local check        # wait until the server is up, then probe it
vbt --profile local-h100 chat
```

The first start downloads 20-50 GB of weights into the Hugging Face cache. It also compiles CUDA
graphs, so allow 10-30 minutes. `/health` returns 200 once the server is ready. Before you rely on
the capacity estimates below, read the `GPU KV cache size: N tokens` line in the startup log.

## Serving profiles

| Serving profile | Hardware | Checkpoint | Context | KV / seqs / MTP | Harness profile | Status |
|---|---|---|---|---|---|---|
| `h100` | 1x H100 80GB | `Qwen/Qwen3.8-27B-FP8` | 262,144 | fp8 / 48 / k=3 (~1.2M tokens, ~8.5 x 128K) | `local-h100` | supported, smoke-test |
| `h200` | 1x H200 141GB | `Qwen/Qwen3.8-27B-FP8` | 262,144 | BF16 / 96 / k=3 (~1.5M tokens, ~11 x 128K) | `local-h200` | supported, smoke-test |
| `rtxpro6000` | 1x RTX PRO 6000 Blackwell 96GB | `RedHatAI/Qwen3.8-27B-NVFP4` | 262,144 | fp8 / 48 / k=3 (~1.85M tokens, ~13 x 128K) | `local-rtxpro6000` | supported, smoke-test |
| `b200` | 1x B200 / B300 | `RedHatAI/Qwen3.8-27B-NVFP4` | 262,144 | fp8 / 128 / k=3 | `local-h200` | supported, smoke-test |
| `5090` | 1x RTX 5090 32GB (any card with 32 GB or more) | `RedHatAI/Qwen3.8-27B-INT4` | 131,072 | fp8 / 4 / off (~360K tokens) | `local-5090` | supported, smoke-test |
| `dp` | 2-8x H200 / B200 (variants: h100, b200, rtxpro6000) | `Qwen/Qwen3.8-27B` (BF16) | 262,144 | BF16 / 64 per rank / k=3 | `local-dp` | supported, smoke-test |
| `deepseek-v4` | 4x H200 / 4x B200 | `deepseek-ai/DeepSeek-V4-Flash-0731` | 393,216 | fp8 / default / off | `local-deepseek-v4` | supported, smoke-test (recipe-verified only on the v0.28.0 image: `--engine-version 0.28.0`) |

For hardware, capacity, alternative checkpoints, variants, caveats and the full command of one
profile, run `vbt local profiles --show NAME`.

- **Variants** change a profile. Examples: `--bulk` (no MTP, more sequences, 64K context on
  Hopper), `--variant mtp2`, `--variant fp8kv` (H200), and `--variant eager` or
  `--variant ampere` (5090). `--variant bulk-moe` (5090) serves Qwen3.6-35B-A3B NVFP4 for bulk
  throughput.
- **Alternative checkpoints**: `--hf-id nvidia/Qwen3.8-27B-NVFP4` is the recipe-verified fallback
  on the RTX PRO 6000.
- **vLLM 0.30.0 fallback**: `--engine-version 0.30.0` drops `--tool-strict-level`. Per-tool
  `strict: true` and a forced `tool_choice` are still grammar-enforced on 0.30, but some
  hybrid-model prefix-cache fixes are missing.
- **Docker image tag**: `vllm/vllm-openai:v0.31.0` needs NVIDIA driver 580 or newer (CUDA 13.0).
  For drivers 575-579, use `v0.31.0-cu129` (`VLLM_TAG=v0.31.0-cu129` for compose). With
  `--docker`, `vbt local serve` picks the tag from `nvidia-smi` or from `--driver X.Y`.
- **Bare metal**: install vLLM in its own environment with `pip install 'vllm==0.31.0'`. Let vLLM
  pin transformers (>=5.10.4,<5.18.0) and never upgrade transformers separately.

## The harness side

`configs/profiles/local-*.yaml` set the following:

- **Provider.** `vllm`, with `base_url` taken from `$VBT_LLM_BASE_URL` (default
  `http://localhost:8000/v1`) and `data_parallel_size` from `$VBT_LLM_DP_SIZE` (it must equal
  the server's `--data-parallel-size`; `vbt local serve` prints the export when they differ).
  The model requested on the wire is the tier's model (`qwen3.8-27b`, the server's
  `--served-model-name`), so `--model` works; the model family is matched from that name.
  `max_concurrency` is sized to `--max-num-seqs`.
- **Tiers.** All tiers use one model and differ only in per-request fields:

  | Tier | Effort | `thinking_token_budget` | `max_tokens` |
  |---|---|---|---|
  | CSO | `xhigh` | 24576 | 40960 |
  | Scientists | `medium` (single-cell analyst: `xhigh`, see below) | 8192 | 32768 |
  | Support | thinking off | — | 16000 |
  | Bulk | `medium` | 3072 | 8192 |

  Effort stays fixed per agent, so prefix caches stay valid. One exception to the tier effort:
  `configs/agents.yaml` runs the single-cell analyst at `effort: max` (as upstream does), which
  overrides the scientist tier. Qwen3.8 maps it to `xhigh` (still capped by the tier's 8192-token
  reasoning budget); DeepSeek-V4 maps it to `high` (its Think-Max needs `max_tokens` >= 128K, so
  `max` is sent only as an explicit `reasoning_effort`). To keep it at the tier's effort, set
  `agent_overrides: {single-cell-analyst: {effort: medium}}`.
- **Token budgets.** `limits.max_turn_tokens` and `limits.max_item_tokens` apply because local
  runs cost 0 USD. The USD caps remain, but they only trigger when `provider.options.pricing` is
  set.
- **Bulk concurrency.** `bulk.default_concurrency` is sized to the server. BulkDispatch jobs need
  `budget_tokens` (a USD-only budget never stops a 0-USD model, so it is refused), capped by
  `bulk.dispatch_max_budget_tokens` (20M); `vbt bulk` / `vbt case1 annotate` need `--budget-tokens`
  rather than `--budget` alone.
- **Web search.** `web.search` uses SearxNG (`$SEARXNG_URL`, default `http://localhost:8888`).
- **Context compaction.** It starts earlier than with Claude, because the chat template replays
  every past reasoning block.

To use Claude again, run `vbt --profile claude ...`. To reproduce the paper's Claude
Sonnet/Haiku 4.5 setup, run `vbt --profile paper ...`.

## Checking a server: `vbt local check`

`vbt local check` probes the server through the harness's own adapter, built from the same
`provider.options` the runtime uses (`base_urls`, `routing`, `data_parallel_size`, `headers`,
`extra_body`, ...). Each check passes, warns, fails or is skipped against a threshold.

| Check | Passes when |
|---|---|
| `health`, `models`, `version`, `prepare` | The server is up. The served name is listed. `max_model_len` is at least the configured window. vLLM is 0.31 or newer (older versions only warn). `prepare` also compares `data_parallel_size` with the server's DP engines (`/metrics`) |
| `routing` | Every configured `base_urls` replica serves the model, and with `data_parallel_size` > 1 every `X-data-parallel-rank` 0..N-1 is accepted (skipped for one server without DP) |
| `tool_call`, `parallel_tool_calls` | One correct call is made. Three independent lookups produce at least 2 calls in one turn |
| `strict_schema` | `submit_result` with the Case 1 `TrialAnnotation` schema (13 anyOf, 7 $ref, 2 regex patterns, `strict: true`) validates with pydantic |
| `forced_tool_choice` | A `tool_choice` naming a tool is obeyed |
| `reasoning_effort` | `xhigh` and `medium` return reasoning, and thinking off returns none |
| `thinking_budget` | `thinking_token_budget` caps reasoning: the returned reasoning, counted with the server's `/tokenize`, stays within the budget (an answer that runs on to `max_tokens` after a capped reasoning only warns) |
| `needle_32k` | A code hidden mid-way in a 32K-token document is retrieved. Add `--long-context` for 200K |
| `prefix_cache` | A repeated 12K-token prefix reports `cached_tokens > 0` |
| `throughput` | Output tokens/s at concurrency 8 reach at least `--min-tps` (default 20) |
| `malformed_calls` | At most 10% of 10 nested-argument tool calls are invalid JSON or fail the schema |

You can run a subset with `--only tool_call,strict_schema` or leave checks out with `--skip`. The
command writes `check.json` and `check.md` to `runs/local/check-<UTC time>/`, or to the directory
given by `--out`. It exits 1 if any check fails.

`vbt local bench` sweeps throughput at concurrency 1, 8 and 32 (`--levels`). It reports aggregate
and per-request tokens/s, latency percentiles and time to first token. Each request generates
exactly `--output-tokens` tokens (`ignore_eos`).

## Multi-GPU

- **`dp` (one server with `--data-parallel-size N`).** Use `vbt local serve --profile dp
  --data-parallel 4`. Prefix caches are per replica, so the adapter pins each agent session to
  one rank with the `X-data-parallel-rank` header. The harness's `data_parallel_size` must equal
  N: `local-dp.yaml` reads it from `$VBT_LLM_DP_SIZE` (default 4), and `vbt local serve` /
  `vbt local profiles --detect` print `export VBT_LLM_DP_SIZE=N` when N differs. A mismatch is
  caught before the first turn: `prepare()` compares it with the engines in `/metrics`, and
  `vbt local check` probes every rank.
- **One server per GPU.** Start each one with `CUDA_VISIBLE_DEVICES=i vbt local serve --profile
  h100 --port 800i`. Then set `provider.options.base_urls` to the server URLs and
  `provider.options.routing: urls`.
- **`deepseek-v4`.** DP4 plus expert parallel. It needs the same sticky routing, or use
  `--variant tep` for one shared prefix cache (then `export VBT_LLM_DP_SIZE=1`; the same for
  `tp2` and `rtxpro6000x8`). The profile runs `--trust-remote-code`: pin the reviewed Hub commit
  with `vbt local serve --profile deepseek-v4 --revision <sha>` (also sent as `--code-revision`);
  `vbt local serve` refuses to start it unpinned unless `--allow-unpinned-remote-code`.

## Security

Both services publish on `127.0.0.1` by default. Never publish either one beyond loopback.

- **vLLM has no real authentication.** `--api-key` / `VLLM_API_KEY` protects only the routes
  under `/v1`. `/invocations` (the SageMaker route, which runs chat completions), `/tokenize`,
  `/detokenize`, `/pooling`, `/score`, `/rerank` and `/metrics` stay open, so anyone who can
  reach the port can run inference on the GPU and fill the KV cache and sequence slots the
  harness depends on. A host firewall does not help with Docker either: published ports get
  their own iptables rules, which bypass ufw/firewalld.
- **Remote access.** Keep vLLM on `127.0.0.1` and reach it through an SSH tunnel
  (`ssh -N -L 8000:127.0.0.1:8000 gpu-host`, then `VBT_LLM_BASE_URL=http://localhost:8000/v1`),
  or put a reverse proxy (nginx, Caddy) in front of it that enforces authentication on every
  path and forwards only `/v1/*` and `/health`. Give the proxy's credential to the harness with
  `VBT_LLM_API_KEY` (bearer) or `provider.options.headers`.
- `vbt local serve` refuses a non-loopback `--host` / `--bind` unless you pass
  `--allow-unauthenticated`. Compose's `VLLM_BIND` is not checked: leave it at `127.0.0.1`.
- **Credentials.** The harness sends `VBT_LLM_API_KEY` (or `provider.options.api_key`) to the
  server, never `OPENAI_API_KEY` (opt in with `provider.options.api_key_env: OPENAI_API_KEY`).
- **SearxNG.** Never publish it: its bot limiter is off.
- **Remote code.** Only `deepseek-v4` uses `--trust-remote-code`; pin it with `--revision` (above).
  Containers get a private `/dev/shm` (`--shm-size 16g`), not the host's IPC namespace.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Startup OOM during CUDA-graph capture (32 GB cards) | `--variant eager` (`--enforce-eager`) |
| `model ... is not served` | The harness requests the tier model (`models.<tier>.model`, or `--model`; `provider.options.served_model_name` overrides it when set). Start vLLM with that `--served-model-name`, or pass the served name with `--model` |
| HTTP 400 `data_parallel_rank ... is out of range` / `prepare()` reports fewer data-parallel engines | `export VBT_LLM_DP_SIZE=N` with N = the server's `--data-parallel-size` (`vbt local serve` prints it) |
| HTTP 400 `Unexpected reasoning effort` | `high`, `max` or `minimal` reached Qwen3.8 (adapter bug). Use family `qwen3_8` |
| `prefix_cache` fails with `cached_tokens` 0 | Start vLLM with `--enable-prefix-caching --enable-prompt-tokens-details` (every profile does) |
| Warning about uncalibrated FP8 KV scales (H100) | Expected with the official FP8 checkpoint. Use `h200` (BF16 KV) or calibrate if long-context retrieval matters |
| `max_model_len ... < configured context window` | The server runs a smaller `--max-model-len` (for example `--bulk`). The harness compacts against the server's `max_model_len` once `prepare()` has read it; to make it explicit, lower `models.<tier>.context_window_tokens`, or restart without the variant |
| Driver 575-579 | `VLLM_TAG=v0.31.0-cu129`. Older drivers cannot run vLLM 0.31 |
