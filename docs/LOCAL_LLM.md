# Running the Virtual Biotech on a local model

The harness runs on a **local open-weight model** by default: **Qwen3.8-27B**, served by
**vLLM 0.31** through its OpenAI-compatible API on **one 80-96 GB NVIDIA GPU**. Claude is
optional (`--profile claude`, or `--profile paper` for the paper's Claude Sonnet/Haiku 4.5
setup).

This page explains:

1. which model was chosen and why, with the evidence;
2. what "a good GPU" means here, and what each GPU class can hold;
3. how to set up the server, web search and the harness;
4. how the harness drives the model (effort tiers, reasoning budgets, token budgets), and what
   to do when the GPU is also needed for analysis steps such as cell2location;
5. how to validate a deployment, and the risks that are still open.

Related pages:
- [deploy/local/README.md](../deploy/local/README.md): operations reference (compose services,
  every `vbt local check` probe, security, troubleshooting).
- [PROVIDERS.md](PROVIDERS.md): the adapter's internals (`src/vbt/providers/openai_compat.py`).
- [WEB_SEARCH.md](WEB_SEARCH.md): the search backends.
- [LOCAL_LLM_VERIFICATION.md](LOCAL_LLM_VERIFICATION.md): what was tested against a real
  inference engine (CPU only).

```bash
# Quick start on one H100 (other GPUs: section 3)
vbt local profiles --detect                                             # recommends a serving profile
docker compose -f deploy/local/docker-compose.yml --profile h100 up -d  # vLLM + SearxNG
vbt local check                                                         # probe the server once /health is up
vbt chat
```

## 1. The decision

**Model**: Qwen3.8-27B (Alibaba Qwen, August 2026, Apache-2.0). It is a dense 27.8B hybrid
model: 48 Gated-DeltaNet linear-attention layers and 16 gated full-attention layers. It has a
built-in multi-token-prediction (MTP) head and a native context of 262,144 tokens. It is
natively vision-language, but the harness serves it text-only (`--language-model-only`).

**Engine**: vLLM 0.31.0 (released 2026-10-05), OpenAI Chat Completions API, with the
`qwen3_coder` tool-call parser (`qwen3_xml`, a streaming parser for the same call format, on
the 32 GB profile), the `qwen3` reasoning parser and `--tool-strict-level function`. vLLM
0.30.0 is the fallback.

**One checkpoint per GPU architecture**, all with the same chat template and tool-call format,
so one adapter serves every Qwen3.8 profile:

| GPU | Checkpoint |
|---|---|
| Hopper: H100 80GB, H200 141GB | `Qwen/Qwen3.8-27B-FP8` (official) |
| Blackwell: RTX PRO 6000 96GB, B200, B300 | `RedHatAI/Qwen3.8-27B-NVFP4` (v2.0) |
| 32 GB cards (RTX 5090) | `RedHatAI/Qwen3.8-27B-INT4` |
| 2-8 GPUs, more throughput | `Qwen/Qwen3.8-27B` (BF16) as data-parallel replicas |
| 4 GPUs, maximum quality | `deepseek-ai/DeepSeek-V4-Flash-0731` (a different model) |

### How it was chosen

The research ran on 2026-10-05 and surveyed 16 open-weight model families. Three independent
judges scored each family from 0 to 10, each on one criterion:

- **Agentic reliability**: tool-call parser maturity, parallel calls, grammar-enforced
  (strict or forced) tool calls in vLLM, and behaviour in long tool loops.
- **Science**: independently re-measured GPQA-Diamond, SciCode, long-context recall and any
  biomedical signal.
- **Deployment fit**: whether it fits one 80-96 GB GPU with room for about 10 concurrent long
  agent contexts, plus quantized checkpoints, engine support and license.

A verifier then checked each serving profile against the vLLM 0.31.0 source, the model cards
and the vLLM recipe. It corrected several numbers in the first draft. Every number on this
page, and in `configs/local_models.yaml`, is the corrected value.

**Judge scores** (0-10). The judges scored for a single-GPU deployment, so models that need 2-8
GPUs score low on fit even when they are strong. That is why DeepSeek-V4 ranks near the bottom
here but is still the multi-GPU option.

| Family | Agentic | Science | Fit | Mean | Outcome |
|---|---|---|---|---|---|
| **Qwen3.8 (27B dense)** | **8** | **8.5** | **9** | **8.5** | **Chosen**: ranked first by all three judges |
| Qwen3.6 / Qwen3.5 (35B-A3B, 27B) | 7 | 6.5 | 8 | 7.2 | Bulk-throughput alternative (5090 `bulk-moe` variant) |
| Intern Agents-A1 / Intern-S2 | 5.5 | 6 | 7 | 6.2 | Not chosen |
| Google Gemma 4 (31B) | 4.5 | 6 | 6 | 5.5 | Comparison model for the in-house evaluation |
| Ornith-1.5 (Qwen3.6 derivative) | 4.5 | 4.5 | 6.5 | 5.2 | Not chosen |
| Meta Muse Glimmer (30B) | 4 | 6 | 5.5 | 5.2 | Comparison model for the in-house evaluation |
| NVIDIA Nemotron 3 / 3.5 | 4.5 | 3.5 | 6 | 4.7 | Not chosen |
| inclusionAI Ling-3.0 | 4.5 | 4.5 | 4.5 | 4.5 | Not chosen |
| OpenAI gpt-oss (120b) | 4.5 | 3.5 | 5 | 4.3 | Not chosen |
| Z.ai GLM-5.3-Flash / GLM-4.7-Flash | 5 | 3.5 | 4 | 4.2 | Alternative 4-GPU max-quality model (no shipped profile) |
| Qwen3.8-Flash-Next | 4 | 4 | 3.5 | 3.8 | Not chosen |
| Mistral Small 4 / Medium 3.5 / Devstral 2 | 3.5 | 3 | 3.5 | 3.3 | Not chosen |
| DeepSeek-V4-Flash / V4.1-Flash | 3 | 2.5 | 2 | 2.5 | Chosen for the 4-GPU max-quality profile (`deepseek-v4`) |
| Moonshot Kimi K3 / Kimi-Linear | 2 | 2 | 3 | 2.3 | Not chosen |
| MiniMax-M3 / M2.7 | 2 | 2 | 1.5 | 1.8 | Not chosen |
| Thinking Machines Inkling | 1.5 | 2 | 1 | 1.5 | Not chosen |

**Re-measured benchmarks.** Most capability numbers are vendor claims. The table keeps the
numbers that someone other than the model's vendor re-measured. They come mostly from the
evaluation tables of quantized checkpoints, for example NVIDIA's `nvidia/Qwen3.8-27B-NVFP4` and
`nvidia/DeepSeek-V4-Flash-0731-NVFP4` cards and Red Hat's `RedHatAI/Qwen3.8-27B-NVFP4` card.
These re-runs are third-party but not fully independent: the quantizers choose their own
harnesses. Unlabelled numbers are NVIDIA's. A dash means no re-measured number was found.
"vendor" marks a number the model's own vendor published.

| Model | GPQA-D | SciCode | AA-LCR (long context) | IFBench | Terminal-Bench 2.1 | Other |
|---|---|---|---|---|---|---|
| **Qwen3.8-27B** | **88.9** (NVIDIA); **89.2-89.6** (Red Hat) | **47.9** (NVIDIA) | **72.6** (NVIDIA) | **80.1** (NVIDIA) | **75.6** (NVIDIA) | SWE-bench 78.8 (Red Hat); LlamaIndex ExtractBench 89.8 (38 on long documents); protoLabs 54-case function-call suite 88.9% (FP8); third-party AA Intelligence Index 52, the highest of any model that fits one GPU |
| Qwen3.6-27B | 86.0 | 44.8 | 68.8 | — | — | HLE 21.7 |
| Qwen3.6-35B-A3B | 84.5-84.9 | 40.8 | 62.0 | ~62 | — | SWE-bench Verified 54.8 against the vendor's 73.4, BFCLv4 57.8 (Red Hat) |
| Gemma-4-31B | 85.8; 86.4 (Red Hat) | 33.6 (subtask; vendor 43) | — | — | — | Meta's bio panel: VCT 43.5, MBCT 50.6, best in its size class (run by Meta, a competitor) |
| Muse-Glimmer-30B | 83.8 | 43.6 (subtask 47.6) | 76.3 | — | 45 (vendor 51.7) | LAB-Bench ProtocolQA 80.2 (run by Meta, its vendor) |
| Nemotron 3.5 Lightning-30B-A3B | 75.4 (vendor) | — | 52 (vendor) | — | 24.6 (vendor) | SWE-bench Verified 51.6; all from NVIDIA's own card, so vendor numbers |
| gpt-oss-120b | — | — | 48.8 | — | — | RULER-128K about 51; SimpleQA hallucination rate 78% (source not recorded) |
| Ling-3.0-flash | — | — | — | — | — | Third-party AA Intelligence Index 38 |
| *Multi-GPU:* DeepSeek-V4-Flash-0731 | 91.5 | 51.7 | 72.1 | 75.8 | 74.7 | tau2-Telecom 98.7 (max effort) |
| *Multi-GPU:* GLM-5.3-Flash | 92.2 | 56.2 | 71 | 61.3 | 82.6 | |
| *Multi-GPU:* Qwen3.8-Flash-Next | 92.0 | 16.3-18.8 | 71.9-74.1 | — | — | HLE 34.7 |

### Why Qwen3.8-27B

- **It is the strongest model that fits one 80-96 GB GPU** with room for about 8-13 concurrent
  128K-token agent contexts (section 2). The re-measured numbers above agree with the vendor's:
  GPQA-D about 89, SciCode 47.9 (the best of the single-GPU class), long-context recall
  (AA-LCR) 72.6, instruction following (IFBench) 80.1 and Terminal-Bench 2.1 75.6.
- **The KV cache is small.** Only 16 of its 64 layers keep one: 32 KiB per token with FP8 KV,
  about 4.3 GB per 128K-token sequence.
- **Tool calls can be grammar-enforced.** The research confirmed in the vLLM source that the
  `qwen3_coder` parser carries an xgrammar structural tag (`qwen_3_coder`) in vLLM 0.29, 0.30
  and 0.31. A forced call, or a tool declared `strict: true`, is therefore constrained to the
  schema. Bulk annotation relies on this: the Case 1 `TrialAnnotation` schema has 13 `anyOf`,
  7 `$ref` and 2 regex patterns. Parallel tool calls are native, so the CSO's fan-out to
  several specialists works in one turn.
- **Per-request reasoning control.** One model serves every tier. Tiers differ only in
  `reasoning_effort` (`xhigh`, `medium`, `low` or `none`) and a hard `thinking_token_budget`.
- **Ecosystem.** Apache-2.0. An official vLLM recipe and SGLang cookbook. NVIDIA and Red Hat
  quantizations. About 6.8M + 4.8M downloads (base and FP8 checkpoints).

**What it was built for.** Qwen positions Qwen3.8 for "coding, professional work, research,
and long-horizon agentic tasks"; the 27B is the "compact, deployment-friendly dense" member.
Most of its agent benchmarks were run inside the Claude Code harness. It is a general agentic
model, **not a biomedical specialist**. Biomedical facts must come from the harness's MCP data
tools and claim-evidence records, and quality has to be validated in-house (section 5).

**Facts the research checked where the judges or the notes disagreed:**

1. The MTP speculative-decoding fixes for the Gated-DeltaNet layers (vLLM #51812, #51674) are
   in every vLLM release from 0.28.0 on, so MTP is usable on 0.31.0.
2. Only the server-wide `--tool-strict-level` flag is new in 0.31. vLLM 0.30.0 already enforces
   per-tool `strict: true` and a forced `tool_choice`.
3. The Qwen3.8 template accepts only `xhigh`, `medium` and `low` (plus `none`, which vLLM turns
   into `enable_thinking=false`). `high`, `max` and `minimal` make the template raise, which
   comes back as HTTP 400. The adapter never sends them.
4. `thinking_token_budget` works with `--reasoning-parser qwen3` (vLLM forces `</think>` when
   the budget is spent).
5. An FP8 KV cache is not free. A small community MRCR test (15 samples, about 220K tokens)
   scored 0.717 with FP8 KV against 0.833 with BF16. NVIDIA's AA-LCR run with FP8 KV matched
   BF16. Hence BF16 KV where capacity allows (H200), FP8 KV on H100, and calibrated FP8 KV
   scales on Blackwell (section 5).

### Runners-up and why not

| Family (checkpoint considered) | Why not |
|---|---|
| Qwen3.6-35B-A3B (`Qwen/Qwen3.6-35B-A3B-FP8`, `nvidia/Qwen3.6-35B-A3B-NVFP4`); Qwen3.6-27B (`Qwen/Qwen3.6-27B-FP8`) | The best throughput fit (3B active, about 1.3 GB of KV per 128K sequence, same parser and template family), but clearly weaker: Red Hat measured SWE-bench Verified 54.8 against the vendor's 73.4; NVIDIA measured IFBench about 62 and AA-LCR 62. Qwen3.6-27B is superseded by Qwen3.8-27B at the same size. The 35B-A3B is kept as the bulk-throughput option. |
| Gemma 4 31B (`RedHatAI/gemma-4-31B-it-FP8-dynamic`) | The strongest biomedical knowledge in its size class and good 128K retrieval. But in vLLM 0.31 the `gemma4` parser has no structural tag and does not enforce `required` or named `tool_choice`, and its template drops `anyOf`/`$ref`/`pattern`, which the `TrialAnnotation` schema uses. There are community reports of multi-round tool-call breakage. KV is heavier (about 5.8 GB per 128K at FP8). Knowledge cutoff January 2025. |
| Muse Glimmer 30B (`RedHatAI/Muse-Glimmer-30B-FP8-block`) | Only 131K native context, and 39 of its 52 layers see a 2K window. Its tool parser runs `json.loads` on every argument without the schema, so a gene ID such as `"7157"` silently becomes an integer: a data-integrity hazard for gene and trial IDs. No grammar-enforced tool calls. An open template bug turns bare tool names (`Bash`, `Read`, `mcp__*`) into namespaces. Reasoning is always on. NVIDIA re-ran Terminal-Bench 2.1 at 45 against the vendor's 51.7. |
| Intern Agents-A1 / Intern-S2-Preview (`InternScience/Agents-A1-FP8`, `internlm/Intern-S2-Preview-FP8`) | Science post-training on Qwen3.5-35B-A3B, with the same parser. But every headline number is vendor-run, tau2 is below its own base model, there are no MTP weights and no `generation_config`, adoption is tiny (408 FP8 downloads), and Intern-S2 needs `trust_remote_code`. |
| gpt-oss-120b (`openai/gpt-oss-120b`) | A mature parser with strict tags, but exactly one tool call per sampling step, so the CSO's parallel fan-out serializes. 131K context with weak retrieval (AA-LCR 48.8). 78% SimpleQA hallucination rate, June 2024 cutoff, CBRN-filtered biology pretraining. Leaves about one sequence of KV on an 80 GB H100. |
| Nemotron 3.5 Lightning-30B-A3B / Nemotron 3 Super-120B | Lightning has an ideal memory profile, but NVIDIA's own card shows weak agentic and science results (SWE-bench Verified 51.6, Terminal-Bench 2.1 24.6, GPQA 75.4, AA-LCR 52) and no effort control. Super-120B NVFP4 takes 84% of 96 GB and does not fit an H100. |
| Ling-3.0-flash (`inclusionAI/Ling-3.0-flash-fp4`) | 70.4 GB: does not fit an H100. AA Intelligence Index 38. Users report a newline-flood collapse at about 150K tokens in long agent sessions. |
| Mistral Small 4 (`mistralai/Mistral-Small-4-119B-2603-NVFP4`) | Grammar-enforced tool calls, but weak agentic tool use by Mistral's own charts (τ³-Telecom 47.1, Banking 7.0, GPQA 71). Takes 74% of 96 GB, does not fit an H100, and `reasoning_effort` is only none or high. |
| Ornith-1.5-35B-A3B | Undisclosed base model (the weights are architecturally identical to Qwen3.6-35B-A3B), no LICENSE file, vendor-only claims judged by Claude, and HF threads report failing basic read/write tool calls and looping. |
| GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) / GLM-4.7-Flash | GLM-5.3-Flash is strong (NVIDIA: GPQA 92.2, SciCode 56.2, AA-LCR 71) with a grammar-enforced `glm47` parser, but needs 4 GPUs (the smallest 4-bit build is about 190-205 GB), has weaker instruction following (IFBench 61.3), and its thinking cannot be disabled. It is the alternative max-quality option on 4 Blackwell GPUs. GLM-4.7-Flash fits one GPU but is weak (GPQA 75.2). |
| DeepSeek-V4-Flash-0731 (`deepseek-ai/DeepSeek-V4-Flash-0731`) | 167 GB of weights: no single-GPU path except a community prune calibrated on code and math only. Chosen instead for the 4-GPU max-quality profile (`deepseek-v4`). |
| Qwen3.8-Flash-Next | Does not fit one GPU usefully (about 75-84 GB stays resident; a KV pool of only 78-138K tokens on an RTX PRO 6000). Nightly-only stack, NVIDIA-measured SciCode only 16-19, reports of premature empty turns at 70-80K tokens, `qwen-community-1.0` license. |
| Inkling / Inkling-Small | Smallest build 170.7 GB. vLLM cannot run it on sm120 (vLLM #51405). Forced `tool_choice` silently degrades (0/5 in vLLM PR #54120) and `json_schema` returns HTTP 500. |
| MiniMax-M3 / M2.7 | Nothing fits one GPU (smallest 139.9 GB). M2.7 is non-commercial and M3 uses a community license. The M3 parser has no structural tag. |
| Kimi K3 / K2.7-Code / Kimi-Linear-48B | K3 needs 8x B300 or 16x H200. Kimi-Linear-48B is a late-2025 non-thinking research model without tool-calling documentation. |

## 2. What "a good GPU" means here

**Primary target: one NVIDIA GPU with 80-96 GB or more**, with native FP8 (Hopper) or native
FP8 and NVFP4 (Blackwell):

- **Hopper (sm90)**: H100 80 GB (SXM, PCIe or NVL; about 79.6 GiB usable) or H200 141 GB.
  FP8 tensor cores, no FP4. Run the official block-FP8 checkpoint. NVFP4 checkpoints would run
  only as W4A16 Marlin (saves memory, not compute), and MXFP4 builds are not an option (the
  vLLM recipe says they do not load on NVIDIA GPUs).
- **Blackwell**: RTX PRO 6000 Blackwell 96 GB (sm120, about 95.6 GiB) or B200 180 GB / B300
  (sm100/sm103). Native NVFP4 (vLLM uses the FlashInfer CUTLASS kernel on sm120). Run Red Hat's
  NVFP4 build with its calibrated FP8 KV scales.

**Host**: Linux x86_64; NVIDIA driver 575 or newer (580 or newer for the default
`vllm/vllm-openai:v0.31.0` image, which is built on CUDA 13.0); at least 128 GB of RAM;
500 GB or more of NVMe for the weights cache, plus the reference data (Open Targets is about
40 GB). Docker with the NVIDIA Container Toolkit, or a separate Python environment for vLLM.

**Why 80 GB.** The weights take 22-28 GiB. The rest of the GPU becomes the KV cache, which
holds the conversations of every running agent. A CSO session delegates to up to 8-10
specialists in parallel (`limits.max_parallel_agents`), and each specialist's context grows to
100K tokens or more over hundreds of tool calls. An 80 GB card holds about 8.5 of those at
128K; a 32 GB card holds about 2.7.

**Memory geometry** (the basis of the capacity numbers below):
- KV: 16 full-attention layers x 4 KV heads x 256 dims = 32 KiB per token with FP8 KV and 64
  KiB with BF16.
- Every running sequence also holds fp32 Gated-DeltaNet state, about 0.15 GB per copy. With
  prefix caching (vLLM's "align" mode) a running sequence pins 2 + k copies, where k is the
  number of MTP draft tokens: about 0.77 GB with k=3 and 0.31 GB without MTP (estimate). With
  many short contexts this state, not the KV, caps the number of resident sequences, so each
  profile's `--max-num-seqs` sits just below that cap.
- KV blocks are padded to the Gated-DeltaNet page: 1,568 tokens with FP8 KV, 784 with BF16.
  Prefix-cache hits land on these block boundaries.

### Serving profiles

Each serving profile (`vbt local serve --profile NAME`, `configs/local_models.yaml`) has a
matching harness profile (`vbt --profile local-NAME`, `configs/profiles/`). Capacity numbers
are estimates at `--gpu-memory-utilization 0.92`: read the real `GPU KV cache size: N tokens`
line in the vLLM startup log.

**Primary: one 80-96 GB GPU**

| Profile | GPU | Checkpoint and precision | KV cache | Context | `--max-num-seqs` | MTP | Expected capacity | Harness profile |
|---|---|---|---|---|---|---|---|---|
| `h100` | 1x H100 80GB | `Qwen/Qwen3.8-27B-FP8`: FP8 W8A8, 128x128 block scales, 30.9 GB on disk | FP8, uncalibrated (the checkpoint ships no KV scales) | 262,144 | 48 | k=3 | ~40 GiB pool, ~1.2M tokens: ~8.5 x 128K, ~15 x 64K, ~23 x 32K, ~32 x 16K (~50 x 16K without MTP); at most ~52 resident sequences with MTP-3 | `local-h100`, also the built-in default |
| `h200` | 1x H200 141GB | `Qwen/Qwen3.8-27B-FP8` | BF16 | 262,144 | 96 | k=3 | ~97 GiB pool, ~1.5M tokens: ~11 x 128K, ~56 x 16K; ~125 resident. Variant `fp8kv`: ~2.9M tokens, ~20 x 128K | `local-h200` |
| `rtxpro6000` | 1x RTX PRO 6000 Blackwell 96GB | `RedHatAI/Qwen3.8-27B-NVFP4` v2.0 (24.7 GB): NVFP4 W4A4 MLPs; FP8 attention, GDN projections and last 8 MLP layers; BF16 lm_head | FP8 with calibrated static scales | 262,144 | 48 | k=3 | ~60 GiB pool, ~1.85M tokens: ~7 x 262K or ~13 x 128K; short bulk agents ~55-65 with MTP, ~90-130 without | `local-rtxpro6000` |
| `b200` | 1x B200 180GB / B300 | `RedHatAI/Qwen3.8-27B-NVFP4` (variant `bf16`: `Qwen/Qwen3.8-27B` with BF16 KV) | FP8, calibrated | 262,144 | 128 | k=3 | ~140 GiB pool (estimate); variant `bulk`: 256 sequences without MTP | `local-h200` (raise `provider.options.max_concurrency` to 128) |

NVFP4 is not twice as fast as FP8 here. Only the MLPs of layers 0-55 are W4A4, so expect at
most about 1.4x the GEMM throughput (about 1.25x in bandwidth-bound decoding).

**Secondary: a 32 GB card**

| Profile | GPU | Checkpoint and precision | KV cache | Context | `--max-num-seqs` | MTP | Expected capacity | Harness profile |
|---|---|---|---|---|---|---|---|---|
| `5090` | 1x RTX 5090 32GB, or any 32 GB+ card (Ada; Ampere with variant `ampere`, BF16 KV) | `RedHatAI/Qwen3.8-27B-INT4`: W4A16 (AWQ+GPTQ, group 128, Marlin kernels), 19.45 GB, tokenizer `Qwen/Qwen3.8-27B` | FP8, calibrated | 131,072 | 4 | off (variant `mtp`) | ~12 GiB pool, ~360-370K tokens: ~2.7 x 128K or ~5 x 64K; ~294 MiB of Gated-DeltaNet state per running sequence | `local-5090` |

This is a development and single-investigator profile, not a machine for 10 specialists at
128K each. The harness profile runs the CSO at `medium` effort, caps parallel specialists at 3
and compacts from about 65K tokens. Red Hat reports GPQA 87.88 against 89.23 for BF16. If
startup runs out of memory during CUDA-graph capture, use `--variant eager`. For bulk
throughput on a 32 GB Blackwell card, `--variant bulk-moe` serves `nvidia/Qwen3.6-35B-A3B-NVFP4`
instead (3B active, about 28-35 concurrent 8-12K-token requests, effort tiers collapse to
thinking on or off).

**Secondary: several GPUs**

| Profile | GPUs | Checkpoint and precision | KV cache | Context | `--max-num-seqs` | MTP | Expected capacity | Harness profile |
|---|---|---|---|---|---|---|---|---|
| `dp` | 2-8x H200 / B200 (variants `h100`, `b200`, `rtxpro6000`) | `Qwen/Qwen3.8-27B`, BF16 (variant `h100`: the FP8 checkpoint with FP8 KV) | BF16 | 262,144 | 64 per rank (B200 96, RTX PRO 6000 24, H100 48) | k=3 | per GPU: H200 ~74 GiB pool, ~37-40 agents at 20K, ~10 at 100K, ~4 at 262K; B200 ~110 GiB, ~55-58 at 20K, ~15 at 100K; RTX PRO 6000 ~33 GiB, ~17 at 20K | `local-dp` |
| `deepseek-v4` | 4x H200 / 4x B200 (DP4 + expert parallel; ~49 GB of weights per GPU) | `deepseek-ai/DeepSeek-V4-Flash-0731` (MIT, 284B total / 13B active, 166.9 GB: FP4 experts + FP8) | FP8 | 393,216 | engine default | off (variant `dspark`) | KV about 3.5 KB per token: large headroom on H200 | `local-deepseek-v4` |

- **`dp`** runs the same model as the single-GPU profiles, so the harness behaves the same.
  Data parallelism beats tensor parallelism for a model that fits one GPU. Prefix caches are
  per replica, so the adapter pins each agent session to one replica with the
  `X-data-parallel-rank` header. Alternatively, run one server per GPU
  (`CUDA_VISIBLE_DEVICES=i vbt local serve --profile h100 --port 800i`) and list them in
  `provider.options.base_urls` with `routing: urls`. With replicas, scientists run at `xhigh`
  with a 16K reasoning budget.
- **`deepseek-v4`** is a different, larger model: NVIDIA measured tau2-Telecom 98.7, IFBench
  75.8, GPQA-D 91.5, AA-LCR 72.1 and SciCode 51.7 (max effort). It has its own effort
  vocabulary (none, low, high, max), which the adapter's `deepseek_v4` family maps. It runs
  `--trust-remote-code`, so `vbt local serve` refuses to start it unless the checkpoint is
  pinned with `--revision <sha>`. The vLLM recipe verified it only on the v0.28.0 image
  (`--engine-version 0.28.0`); the default v0.31.0 command is unverified. On 4 Blackwell GPUs,
  `zai-org/GLM-5.3-Flash` is the alternative (stronger at scientific coding, weaker at following
  instructions); it has no shipped profile.

**Verification status.** Every profile is marked "supported, smoke-test": expected to work,
but not run by the vLLM recipe on that exact card and checkpoint. The recipe verified
Qwen3.8-27B on GB300, on the RTX PRO 6000 (with the `nvidia/Qwen3.8-27B-NVFP4` build, the first
fallback of the `rtxpro6000` profile), on 1-2 RTX 5090s and on DGX Spark. H100/H200 FP8 is
"supported" in the recipe and verified in SGLang on H200. Red Hat evaluated its NVFP4 build on
B200, not on sm120. Run `vbt local check` on your card before production runs (section 5).

## 3. Setup

### 3.1 Install the harness

```bash
git clone --recursive <this repo> && cd Virtual-Biotech-Framework
conda env create -f environment.yml && conda activate vbt-harness
pip install -e ".[web,tools,dev]"      # no Anthropic SDK needed for the local model
cp .env.example .env                    # OPEN_TARGETS_DATA_PATH, and VBT_LLM_BASE_URL if vLLM runs elsewhere
```

### 3.2 Pick a serving profile

```bash
vbt local profiles                      # all profiles
vbt local profiles --detect             # read nvidia-smi: profile, variants, docker tag
vbt local profiles --show h100          # hardware, capacity, alternatives, variants, caveats, full command
```

### 3.3 Start vLLM

The first start downloads 20-50 GB of weights into the Hugging Face cache and compiles CUDA
graphs: allow 10-30 minutes. The server is ready when `curl -s localhost:8000/health` returns
HTTP 200.

**Docker Compose** (profiles `h100`, `h200`, `rtxpro6000` and `5090`; also starts SearxNG):

```bash
docker compose -f deploy/local/docker-compose.yml --profile h100 up -d
docker compose -f deploy/local/docker-compose.yml --profile h100 logs -f vllm-h100
```

Compose reads `VLLM_TAG` (use `v0.31.0-cu129` for drivers 575-579), `VLLM_GPU` (GPU index,
default 0), `VLLM_PORT`, `VLLM_BIND` (keep `127.0.0.1`), `VLLM_API_KEY`, `HF_CACHE` and
`HF_TOKEN`.

**Docker without Compose** (any profile, variant or alternative checkpoint):

```bash
vbt local serve --profile h100 --docker --detach            # picks the image tag from nvidia-smi
vbt local serve --profile rtxpro6000 --bulk --docker --dry-run   # print the docker run line only
docker compose -f deploy/local/docker-compose.yml up -d searxng
```

**Bare metal.** Install vLLM in its own environment and let it pin `transformers`
(`>=5.10.4,<5.18.0`); never upgrade `transformers` separately.

```bash
python3 -m venv ~/venvs/vllm && source ~/venvs/vllm/bin/activate
pip install 'vllm==0.31.0'              # or: uv pip install vllm==0.31.0 --torch-backend=auto
deploy/local/serve_vllm.sh --profile h100        # needs only Python and PyYAML (vLLM has both)
#   or, from the harness environment with vLLM on PATH: vbt local serve --profile h100
```

The default vLLM 0.31.0 wheels are built for CUDA 12.9. If pip resolves a PyTorch build for
another CUDA version, add `--extra-index-url https://download.pytorch.org/whl/cu129`.

**Useful options of `vbt local serve`**: `--dry-run` (print the command), `--bulk` (a
bulk-annotation-only server: no MTP, more sequences), `--variant NAME`, `--hf-id REPO` (a listed
alternative checkpoint), `--engine-version 0.30.0` (drops `--tool-strict-level`), `--data-parallel
N`, `--port`, `--revision SHA`, and `--vllm-arg=--flag=value` (appended after the profile's
flags; vLLM's argument parser keeps the last occurrence of a flag).

**Remote GPU host.** Keep vLLM on `127.0.0.1` and use an SSH tunnel:
`ssh -N -L 8000:127.0.0.1:8000 gpu-host`. vLLM's `--api-key` protects only the `/v1` routes, so
`vbt local serve` refuses a non-loopback `--host` or `--bind` unless you pass
`--allow-unauthenticated`. See the Security section of
[deploy/local/README.md](../deploy/local/README.md).

### 3.4 Web search (SearxNG)

The local model has no built-in web search, so the harness's `WebSearch` tool calls a search
backend itself. The shipped configs point it at a self-hosted SearxNG on
`http://localhost:8888`:

```bash
docker compose -f deploy/local/docker-compose.yml up -d searxng      # already started with --profile h100
curl -s "http://localhost:8888/search?q=PCSK9&format=json" | head -c 300
```

The Brave Search API is the alternative (`BRAVE_SEARCH_API_KEY`, `web.search.backend: brave`).
See [WEB_SEARCH.md](WEB_SEARCH.md).

### 3.5 Environment variables

| Variable | Read by | Default | Purpose |
|---|---|---|---|
| `VBT_LLM_BASE_URL` | harness | `http://localhost:8000/v1` | The inference server's OpenAI API base |
| `VBT_LLM_API_KEY` | harness | unset | Bearer token, only if the server or an auth proxy needs one. `OPENAI_API_KEY` is never sent |
| `VBT_LLM_DP_SIZE` | harness | 1 (`local-dp`, `local-deepseek-v4`: 4) | Must equal the server's `--data-parallel-size`; `vbt local serve` prints the export when it differs |
| `SEARXNG_URL` | harness | `http://localhost:8888` | SearxNG instance for `WebSearch` |
| `BRAVE_SEARCH_API_KEY` | harness | unset | Brave Search API, the alternative backend |
| `HF_TOKEN` | vLLM / compose | unset | Gated checkpoints only (not needed for Qwen3.8 or DeepSeek-V4) |
| `VLLM_TAG`, `VLLM_GPU`, `VLLM_PORT`, `VLLM_BIND`, `VLLM_API_KEY`, `HF_CACHE` | compose | `v0.31.0`, `0`, `8000`, `127.0.0.1`, unset, `~/.cache/huggingface` | Image tag, GPU index, port, bind address, `/v1` key, weights cache |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` | vLLM | `1` (set by every profile) | Keeps the strict tool-call grammar on |
| `CUDA_VISIBLE_DEVICES` | vLLM; agents' `Bash` | unset | Which GPUs a process sees (section 4.6). `CUDA_*` variables pass through to agents' commands |

`.env` holds the harness variables; `.env.example` documents each one.

### 3.6 Check the server

```bash
vbt local check                          # the built-in default (= local-h100); other cards: vbt --profile local-h200 local check
vbt local check --long-context           # adds a 200K-token needle retrieval
vbt local bench --levels 1,8,32,64       # throughput: aggregate and per-request tokens/s, latency, time to first token
vbt doctor --smoke                       # server /health and served model, reference data, MCP servers, one SearxNG query
```

`vbt local check` probes the server through the harness's own adapter: health, served models
and `max_model_len`, engine version, data-parallel routing, a single tool call, parallel tool
calls, the strict `TrialAnnotation` schema, a forced `tool_choice`, reasoning effort, the
thinking budget, needle retrieval at 32K tokens, the prefix cache, throughput and the
malformed-call rate. It writes `check.json` and `check.md` under `runs/local/` and exits 1 if
any check fails. The thresholds are listed in
[deploy/local/README.md](../deploy/local/README.md#checking-a-server-vbt-local-check).

### 3.7 Run the harness

```bash
vbt chat                                 # interactive CSO session on the local model
vbt --profile local-h200 chat            # match the harness profile to the serving profile
vbt run "Evaluate PCSK9 as a target for lowering LDL cholesterol."
vbt web                                  # browser UI
```

Without `--profile`, the harness uses `configs/default.yaml`, whose values are those of
`local-h100`. On other hardware, pass the harness profile that matches the serving profile
(`local-h200` also covers `b200`). The model name sent to the server is the tier's model
(`qwen3.8-27b`), which every serving profile sets with `--served-model-name`.

**Claude instead.** Put `ANTHROPIC_API_KEY` in `.env`, install the `anthropic` extra
(`pip install -e ".[anthropic]"`) and pass `--profile claude` (current Claude models) or
`--profile paper` (the paper's Claude Sonnet 4.5 and Haiku 4.5). Those profiles reset every
local-server option, restore the Claude context policy and budget in USD.

## 4. How the harness drives the model

### 4.1 Tiers: one model, different reasoning settings

The paper used two models: Claude Sonnet 4.5 for the CSO and the scientists, and Claude Haiku
4.5 for the Chief of Staff and the Scientific Reviewer. Locally, one model serves every tier,
and the tiers differ only in per-request fields (`configs/default.yaml`, `local-h100.yaml`,
`local-h200.yaml`, `local-rtxpro6000.yaml`):

| Tier | Agents | Harness effort | Sent to Qwen3.8 | `thinking_token_budget` | `max_tokens` | Sampling |
|---|---|---|---|---|---|---|
| `orchestrator` | CSO | `xhigh` | `reasoning_effort: "xhigh"` | 24,576 | 40,960 | thinking: T 1.0, top_p 0.95, top_k 20, min_p 0, presence 0 |
| `scientist` | the specialist scientists | `medium` | `"medium"` | 8,192 | 32,768 | thinking |
| `support` | Chief of Staff, Scientific Reviewer; WebFetch answer extraction | thinking off | `"none"` (`enable_thinking: false`) | none | 16,000 | non-thinking: T 0.7, top_p 0.8, top_k 20, min_p 0, presence 1.5 |
| `bulk` | one annotator per item (Case 1) | `medium` | `"medium"` | 3,072 | 8,192 | thinking |

- **Effort mapping** (adapter family `qwen3_8`): `low` → `low`, `medium` → `medium`, `high`,
  `xhigh` and `max` → `xhigh`, thinking off → `none`. `high`, `max` and `minimal` are never
  sent, because the Qwen3.8 template rejects them (HTTP 400).
- **Effort stays fixed per agent.** `xhigh` and `low` add a line at the top of the system
  prompt, so switching effort mid-conversation would invalidate that agent's prefix cache
  (`medium` and `none` add nothing). One exception to the tier effort: `configs/agents.yaml`
  runs the single-cell analyst at `effort: max`, as upstream does, which becomes `xhigh` with
  the scientist tier's 8,192-token budget. `agent_overrides: {single-cell-analyst: {effort:
  medium}}` keeps it on the tier.
- **The reasoning budget is always sent.** `thinking_token_budget` is a hard cap: vLLM forces
  `</think>` when it is spent. Community reports describe 20-55K-token self-validation
  loops on Qwen3.8 (HF discussion #178), and an explicit budget fixed them for one reporter.
  `max_tokens` includes the reasoning. A tier without `thinking_budget` gets
  `min(max_tokens / 2, 16384)`.
- **Compaction summaries** use the agent's own model with thinking off, so on one local model
  they are a `none` request like the support tier.
- **Sampling** follows the model card and is sent on every request, because vLLM otherwise
  applies `generation_config.json`, which holds the thinking values.
- **Reasoning is replayed.** Each earlier assistant turn's reasoning is sent back, and the
  template's `preserve_thinking` stays at its default (true), so prompts are append-only and
  the prefix cache stays valid. Contexts grow faster as a result, so compaction starts earlier
  than with Claude: above 55% of the 262K window (about 144K tokens) old tool results and old
  reasoning are cleared, and above 70% (about 183K) older spans are summarised.

**Other profiles.** `local-5090` runs the CSO at `medium` (budget 8,192) and the scientists at
`medium` (4,096), with `max_tokens` 16,384, and runs bulk with thinking off. `local-dp` runs the
scientists at `xhigh` with a 16,384-token budget. `local-deepseek-v4` uses DeepSeek's vocabulary
(CSO and scientists `high`, bulk `low`; the harness's `max` becomes `high`, because Think-Max
needs a `max_tokens` of 128K or more) and has no `thinking_token_budget`; `max_tokens` bounds
the reasoning.

### 4.2 Bulk annotation: grammar-enforced submissions

Each bulk item (one clinical-trialist agent per NCT ID in Case 1) declares `submit_result` with
`strict: true`, so vLLM constrains its arguments to the JSON Schema. Normal turns use
`tool_choice: auto`, so the agent can call lookup tools first. If the agent has not submitted
by its last allowed call, that call forces `tool_choice` to `submit_result`; an agent that ends
its turn without submitting is resumed once for a forced call. If an item's token budget runs
out first, the agent still gets one forced, budget-exempt call, so the evidence it gathered is
submitted rather than lost. The harness also validates the result with Pydantic.

### 4.3 Model-facing guards

Local models need these more than Claude does, so the runtime adds them for every provider:

- Tool calls whose arguments are not valid JSON, or miss required properties, are not run. The
  model gets an error result asking it to re-issue the call (traced as `model_error`).
- An empty reply (no text and no tool call) right after tool results gets one harness nudge
  per invocation (`empty_reply_nudge`).
- A final call where no tool may run (turn limit, budget grace) sends `tool_choice: none`.
- Every request carries a per-invocation session key, which pins an agent to one replica with
  data parallelism.

### 4.4 Budgets are in tokens

A local model costs 0 USD, so a USD cap would never stop it. Budget scopes (a user turn, a
bulk run, a bulk item) therefore also take a token limit, and a scope stops at whichever limit
it reaches first. Budget tokens are
uncached input + output + cached (prefix-cache) input x 0.1 (`limits.cached_token_weight`).

| Harness profile | Tokens per user turn (`max_turn_tokens`) | Tokens per bulk item (`max_item_tokens`) | Parallel specialists (`max_parallel_agents`) | In-flight requests (`max_concurrency`) | Bulk concurrency (`bulk.default_concurrency`) |
|---|---|---|---|---|---|
| built-in default, `local-h100` | 4,000,000 | 400,000 | 8 | 48 | 32 |
| `local-h200` | 4,000,000 | 400,000 | 10 | 96 | 64 |
| `local-rtxpro6000` | 4,000,000 | 400,000 | 10 | 48 | 32 |
| `local-5090` | 2,000,000 | 200,000 | 3 | 4 | 4 |
| `local-dp` | 8,000,000 | 400,000 | 16 | 256 | 128 |
| `local-deepseek-v4` | 8,000,000 | 400,000 | 16 | 128 | 64 |

- The USD caps (`max_turn_cost_usd: 150`) stay in the config but never trigger unless you set
  `provider.options.pricing: {input_per_mtok, cached_per_mtok, output_per_mtok}`, for example
  to charge amortised GPU-hours. Cost reports show 0 USD otherwise.
- `vbt bulk` and `vbt case1 annotate` need `--budget-tokens`: `--budget` (USD) alone is refused
  for an unpriced local model. A CSO `BulkDispatch` job needs `budget_tokens`, capped by
  `bulk.dispatch_max_budget_tokens` (20M).
- `max_concurrency` matches the server's `--max-num-seqs`. vLLM queues excess requests instead
  of returning HTTP 429, so a higher value only adds queueing. A bulk-only server
  (`vbt local serve --profile h100 --bulk`, 128 sequences, no MTP) can take
  `--concurrency 96`.

### 4.5 Capabilities that differ from Claude

- **No vision.** The server runs `--language-model-only`, so images in tool results reach the
  model as `[image: <source>]` text and PDFs as a placeholder. Dropping the flag (about 0.92 GB
  of extra weights) and setting `provider.options.vision: true` sends images as `image_url`
  parts; that path has not been exercised.
- **Web search** goes through SearxNG instead of Anthropic's server-side search (section 3.4).
- **No server-side context editing.** The harness's `context.py` clears and summarises
  history itself.

### 4.6 Sharing the GPU with analysis steps (cell2location)

Some analyses the agents run through `Bash` also want the GPU. The main one is cell2location
spatial deconvolution in case study 2 (`src/vbt/analysis/spatial.py`: a 250-epoch reference
model and a 10,000-epoch spatial model). The authors' agent report in their Zenodo archive
(`b7-h3/traces/agent_reports/single_cell_analyst_spatial_report.md`) says it ran "on a GPU node
(~2-4 hours)". Other PyTorch-based single-cell tools behave the same way.

vLLM claims `--gpu-memory-utilization` x total GPU memory **at startup and keeps it for its
lifetime**: the weights, activations and CUDA graphs, and then the KV pool takes the rest of
that share. The single-GPU Qwen3.8 profiles use 0.92, which leaves about 6 GiB of an H100 for
everything else.
Two failures follow:
- an analysis job started while vLLM runs gets CUDA out-of-memory errors;
- vLLM refuses to start when a job already holds memory ("Free memory on device ... is less
  than desired GPU memory utilization").

Options, best first:

1. **Use a second GPU for analysis.** Run vLLM on GPU 0 (`VLLM_GPU=0` for Compose,
   `CUDA_VISIBLE_DEVICES=0 vbt local serve ...` on bare metal; `vbt local serve --docker`
   passes `--gpus all`, and a single-GPU vLLM then uses the first one) and start the harness
   with `CUDA_VISIBLE_DEVICES=1`. `CUDA_*` variables pass through to agents' `Bash` commands,
   so their PyTorch jobs see only GPU 1.
2. **Lower vLLM's share on a single GPU.** Each 0.05 of `--gpu-memory-utilization` moves about
   4 GiB on an 80 GB card (about 4.8 GiB on 96 GB, 7 GiB on 141 GB) from the KV pool to other
   processes. For example, on an H100 at 0.75 the pool shrinks from about 40 GiB to about
   26 GiB (roughly 0.8M tokens instead of 1.2M), and about 20 GiB is left for analysis. Shrink
   the concurrency settings by the same ratio, or vLLM preempts sequences instead of running
   them: on that H100, about `--max-num-seqs 32`, `provider.options.max_concurrency: 32` and
   `limits.max_parallel_agents: 5`. These are estimates: read the KV cache size in the startup
   log, and measure the analysis job's peak memory with `nvidia-smi` on a pilot run.

   ```bash
   vbt local serve --profile h100 --vllm-arg=--gpu-memory-utilization=0.75 --vllm-arg=--max-num-seqs=32
   ```

   With Compose, pass a second file whose service `command` carries the new values
   (`docker compose -f deploy/local/docker-compose.yml -f gpu-share.yml --profile h100 up -d`;
   a `command` override replaces the whole list), or add `--docker` to the command above.
   vLLM's `--kv-cache-memory-bytes` is an alternative: it fixes the KV pool size and ignores
   the utilization fraction.
3. **Pause the server** for a long job on a single card: stop vLLM, run the analysis, and
   restart vLLM after the job has exited (its startup check needs its whole share free).
4. **Run the analysis on CPU or another node.** cell2location runs on CPU (`use_gpu=False` in
   `run_cell2location`), but much more slowly.

## 5. Validation plan and open risks

### 5.1 What has been verified so far

The adapter has offline tests against fake servers (`tests/test_openai_compat.py`,
`tests/test_families.py`, `tests/test_local_integration.py`, `tests/test_local_ops.py`,
`tests/test_local_review_fixes.py`, `tests/test_search_backends.py`). It was also run against **real inference engines on CPU**,
because the build machine has no GPU. Those runs used vLLM 0.30.0 (the `vllm-cpu` wheel) and
llama.cpp, both serving Qwen3.5-0.8B, a tiny model from the same family with the same hybrid
architecture, tool-call format and reasoning parser
([LOCAL_LLM_VERIFICATION.md](LOCAL_LLM_VERIFICATION.md)).

They proved the plumbing:
- vLLM accepted every field of the Qwen3.8 request dialect: `reasoning_effort`
  `xhigh`/`medium`/`none`, `thinking_token_budget`, the card's sampling, `strict` tools, a named
  `tool_choice`, `parallel_tool_calls` and `stream_options`.
- The message encoding satisfied the Qwen template: one system message first, tool results in
  call order, reasoning replay.
- Streaming decoded reasoning, text and parallel tool calls. Cached-token usage was reported.
  The thinking budget was enforced.
- A forced call with a strict schema (enum, regex, bounds, `anyOf` with null) came back
  schema-valid.
- Context-overflow recovery and unknown-model errors were mapped correctly. Five adapter bugs
  were found and fixed on the way.
- One full CSO turn with a delegation ran through `vbt run` on both engines.

They did **not** prove:
- vLLM 0.31.0, `--tool-strict-level`, or any GPU code path: CUDA graphs, FP8/NVFP4/INT4
  kernels, FP8 KV, MTP, 262K contexts, data-parallel routing.
- The Qwen3.8 template itself: its effort lines and its rejection of `high`/`max`.
- The runtime paths added when the local model became the default: the forced final
  `submit_result`, `tool_choice: none` on final calls, argument checks, the empty-reply nudge
  and token budgets. These have offline tests only.
- Model quality, the full `TrialAnnotation` schema in xgrammar, throughput and capacity.

Prefix caching in a growing agent loop was partial and irregular on vLLM 0.30's hybrid-model
cache, although the harness's requests are append-only. vLLM 0.31 contains several
hybrid-cache fixes, but this has to be measured on the GPU.

### 5.2 Validation plan for a new deployment

1. **Server smoke test** (each new card, checkpoint or engine version):
   `vbt local check --long-context`, then `vbt local bench --levels 1,8,16,32,64,128`. Watch
   for the uncalibrated-FP8-KV warning (H100), the MTP acceptance rate
   (`vllm:spec_decode_num_accepted_tokens_total` / `..._draft_tokens_total` on `/metrics`) and
   the `GPU KV cache size` line.
2. **Live adapter probes on the GPU server**:
   `VBT_LIVE_LOCAL_URL=http://localhost:8000/v1 VBT_LIVE_LOCAL_MODEL=qwen3.8-27b
   VBT_LIVE_LOCAL_HARNESS=1 python -m pytest -v tests/test_live_local.py`. This also covers the
   runtime paths that have only offline tests so far.
3. **Harness smoke test**: about 50 CSO turns with parallel `Task` fan-out, and 3-4 long
   specialist loops past 100K tokens (for example `vbt run -f turns.txt`, or the case-study
   scenarios). From the run's `logs/trace.jsonl`, measure:
   - the malformed-call rate (`tool_end` records with `model_error`);
   - the empty-turn rate (`empty_reply_nudge`);
   - reasoning-budget hits (`thinking_chars` of `model_call` records near the tier budget);
   - the prefix-cache hit rate (`usage.cache_read_tokens` against `usage.input_tokens`);
   - compactions and context overflows.
4. **Bulk sample**: 200-500 trials, then compare with the authors' released labels.
   ```bash
   vbt case1 annotate --sample 200 --budget-tokens 40000000 --out results/case1/pilot_qwen.jsonl
   vbt case1 validate --pred results/case1/pilot_qwen.jsonl --ref released
   ```
   Measure the schema-valid submission rate (completed against failed items in the run
   summary), median tokens per trial and items per hour, at concurrency 16, 64 and 128 against
   a `--bulk` server.
5. **In-house biomedical evaluation.** No candidate has a published biomedical benchmark, so
   build one: BixBench (agentic bioinformatics with scanpy and R), LAB-Bench (LitQA2, DbQA,
   SeqQA, ProtocolQA), a gold set of trial annotations (the released Case 1 labels) and a
   hallucination audit of gene, trial and compound IDs. Compare Qwen3.8-27B with
   `RedHatAI/gemma-4-31B-it-FP8-dynamic` and `RedHatAI/Muse-Glimmer-30B-FP8-block`, and with
   `--profile paper` on the same prompts. Case studies 2 and 3 have a scorer
   (`vbt scenario run b7h3 --score`, `vbt scenario run osmr --score`).

### 5.3 Open risks

- **No biomedical benchmark exists** for Qwen3.8 or any other candidate (5.2, step 5). Qwen
  models also abstain poorly: Qwen3.5-397B scored -29.8 on AA-Omniscience in Thinking Machines'
  comparison table, and Qwen3.6-27B was last in its size class on Meta's wet-lab and virology
  panel (ProtocolQA 69.1, VCT 33.7). Users report Qwen3.8 occasionally inventing tasks or
  misreading tool output (HF discussion #68). Keep claims tied to tool-returned evidence
  (`vbt verify`), and do not accept unsourced biomedical facts.
- **KV precision.** The official FP8 checkpoint ships no KV scales, so H100 runs uncalibrated
  FP8 KV. One small community MRCR test (n=15, about 220K tokens) scored 0.717 against 0.833
  with BF16 KV; NVIDIA's AA-LCR run with FP8 KV matched BF16. Options: accept it, calibrate KV
  scales on your own agent traces (HF discussion Qwen/Qwen3.8-27B-FP8 #10 has a recipe), or use
  an H200 with BF16 KV when long-context retrieval matters most. The Red Hat Blackwell builds
  ship calibrated scales, calibrated at a 4K sequence length.
- **Day-0 vLLM.** vLLM 0.31.0 was released on 2026-10-05, the day of the decision, and
  `--tool-strict-level` is new in it. If it misbehaves, try `--tool-strict-level auto` (strict
  stays on for `submit_result`), or fall back with `--engine-version 0.30.0`. vLLM 0.30.0 still
  enforces strict tools and forced calls, but lacks the hybrid prefix-cache fixes (vLLM
  #58368, #59175, #59146). It is unverified whether xgrammar compiles the full `TrialAnnotation`
  schema (`vbt local check` tests one request) and what that costs in CPU time at hundreds of
  concurrent requests. In 0.31.0 every data-parallel engine starts from the same random state
  (the fix, #59788, is unreleased), so pass a per-request seed for replicate sampling.
- **Throughput of a dense 27B model for bulk runs.** Every token reads about 27B parameters, so
  hundreds of concurrent bulk agents on one GPU are compute-bound. The paper annotated 37,075
  trials. Measure items per hour on the pilot (5.2, step 4) before planning a full run. If it is
  too slow: run a `--bulk` server (no MTP, more sequences), add replicas (`dp`), or serve
  `Qwen/Qwen3.6-35B-A3B-FP8` for the bulk run. That model has 3B active parameters and the same
  parsers, but ignores effort levels other than `none`. Pointing a bulk run at a second server
  (its `VBT_LLM_BASE_URL` and `--model <served name>`) is not yet tested end to end.
- **MTP versus throughput.** MTP with 3 draft tokens helps at about 32 concurrent sequences or
  fewer and costs throughput (and Gated-DeltaNet state memory) beyond that. That is why the
  `--bulk` variants drop it. vLLM 0.31's dynamic `num_speculative_tokens_per_batch_size` is
  untested on hybrid models.
- **Unverified hardware pairings.** Red Hat's NVFP4 build on sm120 (evaluated on B200 only;
  fall back to `nvidia/Qwen3.8-27B-NVFP4`, then `unsloth/Qwen3.8-27B-NVFP4`). H100/H200 FP8 in
  vLLM (recipe "supported", verified in SGLang only). The single-5090 INT4 command may need
  `--variant eager`. DeepSeek-V4 on vLLM 0.31 and its 2-GPU layout.
- **Paper fidelity.** Runs on Qwen3.8 are not replications of the paper, which used Claude
  Sonnet 4.5 and Haiku 4.5. Use `--profile paper` to reproduce the published setup, and treat
  local-model results as a separate configuration with its own validation.
- **Licenses.** Qwen3.8 (Apache-2.0) and DeepSeek-V4-Flash and GLM-5.3-Flash (MIT) are fine for
  research. Recheck the license of any other model before offering the harness as a hosted
  service.
