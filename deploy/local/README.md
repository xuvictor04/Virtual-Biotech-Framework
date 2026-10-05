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
| `deepseek-v4` | 4x H200 / 4x B200 | `deepseek-ai/DeepSeek-V4-Flash-0731` | 393,216 | fp8 / default / off | `local-deepseek-v4` | recipe-verified (v0.28 image) |

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
  `http://localhost:8000/v1`). The served model name and the model family are set too.
  `max_concurrency` is sized to `--max-num-seqs`.
- **Tiers.** All tiers use one model and differ only in per-request fields:

  | Tier | Effort | `thinking_token_budget` | `max_tokens` |
  |---|---|---|---|
  | CSO | `xhigh` | 24576 | 40960 |
  | Scientists | `medium` | 8192 | 32768 |
  | Support | thinking off | — | 16000 |
  | Bulk | `medium` | 3072 | 8192 |

  Effort stays fixed per agent, so prefix caches stay valid.
- **Token budgets.** `limits.max_turn_tokens` and `limits.max_item_tokens` apply because local
  runs cost 0 USD. The USD caps remain, but they only trigger when `provider.options.pricing` is
  set.
- **Bulk concurrency.** `bulk.default_concurrency` is sized to the server.
- **Web search.** `web.search` uses SearxNG (`$SEARXNG_URL`, default `http://localhost:8888`).
- **Context compaction.** It starts earlier than with Claude, because the chat template replays
  every past reasoning block.

To use Claude again, run `vbt --profile claude ...`. To reproduce the paper's Claude
Sonnet/Haiku 4.5 setup, run `vbt --profile paper ...`.

## Checking a server: `vbt local check`

`vbt local check` probes the server through the harness's own adapter. Each check passes, warns,
fails or is skipped against a threshold.

| Check | Passes when |
|---|---|
| `health`, `models`, `version`, `prepare` | The server is up. The served name is listed. `max_model_len` is at least the configured window. vLLM is 0.31 or newer (older versions only warn) |
| `tool_call`, `parallel_tool_calls` | One correct call is made. Three independent lookups produce at least 2 calls in one turn |
| `strict_schema` | `submit_result` with the Case 1 `TrialAnnotation` schema (13 anyOf, 7 $ref, 2 regex patterns, `strict: true`) validates with pydantic |
| `forced_tool_choice` | A `tool_choice` naming a tool is obeyed |
| `reasoning_effort` | `xhigh` and `medium` return reasoning, and thinking off returns none |
| `thinking_budget` | `thinking_token_budget` caps reasoning (no runaway to `max_tokens`) |
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
  one rank with the `X-data-parallel-rank` header. In `local-dp.yaml`, `data_parallel_size` must
  equal N.
- **One server per GPU.** Start each one with `CUDA_VISIBLE_DEVICES=i vbt local serve --profile
  h100 --port 800i`. Then set `provider.options.base_urls` to the server URLs and
  `provider.options.routing: urls`.
- **`deepseek-v4`.** DP4 plus expert parallel. It needs the same sticky routing, or use
  `--variant tep` for one shared prefix cache.

## Security

Both services publish on `127.0.0.1` by default, and neither has authentication.

- **Exposing vLLM beyond localhost.** Set `VLLM_BIND=0.0.0.0` (compose) or `--host 0.0.0.0`
  (bare metal). Also set `VLLM_API_KEY` on the server and the same value as `VBT_LLM_API_KEY` for
  the harness.
- **SearxNG.** Never publish it: its bot limiter is off.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Startup OOM during CUDA-graph capture (32 GB cards) | `--variant eager` (`--enforce-eager`) |
| `model ... is not served` | The harness requests `provider.options.served_model_name`. Start vLLM with the same `--served-model-name` |
| HTTP 400 `Unexpected reasoning effort` | `high`, `max` or `minimal` reached Qwen3.8 (adapter bug). Use family `qwen3_8` |
| `prefix_cache` fails with `cached_tokens` 0 | Start vLLM with `--enable-prefix-caching --enable-prompt-tokens-details` (every profile does) |
| Warning about uncalibrated FP8 KV scales (H100) | Expected with the official FP8 checkpoint. Use `h200` (BF16 KV) or calibrate if long-context retrieval matters |
| `max_model_len ... < configured context window` | The server runs a smaller `--max-model-len` (for example `--bulk`). Lower `models.<tier>.context_window_tokens` or restart without the variant |
| Driver 575-579 | `VLLM_TAG=v0.31.0-cu129`. Older drivers cannot run vLLM 0.31 |
