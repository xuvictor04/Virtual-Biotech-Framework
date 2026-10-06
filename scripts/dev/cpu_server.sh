#!/usr/bin/env bash
# CPU-only OpenAI-compatible inference server for checking the local-LLM adapter
# (vbt.providers.openai_compat) against a REAL engine on a machine without a GPU.
# A tiny model tests the plumbing (tool calls, reasoning fields, streaming, usage,
# overflow errors), not answer quality. See docs/LOCAL_LLM_VERIFICATION.md.
#
#   scripts/dev/cpu_server.sh --install                 # create the venv (vllm-cpu 0.30.0 + CPU torch)
#   scripts/dev/cpu_server.sh                           # serve Qwen/Qwen3.5-0.8B on 127.0.0.1:8011 (foreground)
#   scripts/dev/cpu_server.sh --dry-run                 # print the command only
#   scripts/dev/cpu_server.sh --model Qwen/Qwen3-0.6B   # Qwen3 (hermes tool format) instead
#   scripts/dev/cpu_server.sh --max-model-len 8192 --port 8012    # small window (overflow probes)
#   scripts/dev/cpu_server.sh --engine llamacpp --model /path/model.gguf   # llama.cpp llama-server
#
# Then, from the harness environment:
#   VBT_LIVE_LOCAL_URL=http://127.0.0.1:8011/v1 VBT_LIVE_LOCAL_MODEL=qwen3.5-0.8b \
#     python3 -m pytest -q tests/test_live_local.py
#
# Options:
#   --engine vllm|llamacpp   vllm (default): the community `vllm-cpu` wheel (x86_64/aarch64, a CPU
#                            build of vLLM; third-party, keep it in its own venv). llamacpp: a
#                            `llama-server` binary (LLAMA_SERVER or PATH) with --jinja.
#   --model ID|DIR|GGUF      HF id or local directory (vllm), GGUF file (llamacpp).
#                            Default Qwen/Qwen3.5-0.8B (same qwen3_coder XML tool format and qwen3
#                            reasoning parser as Qwen3.8).
#   --served-name NAME       served model name (default: derived from the model, e.g. qwen3.5-0.8b)
#   --tool-parser NAME       vLLM tool-call parser (default: qwen3_coder for Qwen3.5/3.6/3.8, hermes
#                            for Qwen3, otherwise hermes)
#   --host H / --port P      bind address (default 127.0.0.1:8011; keep it on loopback)
#   --max-model-len N        context window (default 32768)
#   --kv-cache-gb N          VLLM_CPU_KVCACHE_SPACE in GiB (default 4)
#   --batch-tokens N         vLLM --max-num-batched-tokens (default 2048). NB: vllm-cpu 0.30 still
#                            prefilled each Qwen3.5 (hybrid) prompt in ONE engine step: ~6 min for a
#                            12K-token prompt on 4 cores, during which the server answers nothing
#                            else. Keep client read timeouts above the longest prefill.
#   --slots N                llama.cpp server slots (default 1). The window is split between slots:
#                            -c 32768 with 2 slots gives 16384 tokens per request (/v1/models meta.n_ctx)
#   --threads SPEC           VLLM_CPU_OMP_THREADS_BIND (default auto)
#   --venv DIR               venv for the engine (default $VBT_CPU_VENV or ~/.cache/vbt/cpu-venv-<engine>)
#   --install                create the venv and install the engine, then exit
#   --dry-run                print the command and exit
#   -- ARGS...               extra engine arguments
set -euo pipefail

ENGINE=vllm
MODEL=Qwen/Qwen3.5-0.8B
SERVED=""
TOOL_PARSER=""
HOST=127.0.0.1
PORT=8011
MAX_LEN=32768
KV_GB=4
BATCH_TOKENS=2048
SLOTS=1
THREADS=auto
VENV="${VBT_CPU_VENV:-}"
INSTALL=0
DRY=0
EXTRA=()
VLLM_CPU_VERSION="${VLLM_CPU_VERSION:-0.30.0}"

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --served-name) SERVED="$2"; shift 2 ;;
    --tool-parser) TOOL_PARSER="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --max-model-len) MAX_LEN="$2"; shift 2 ;;
    --kv-cache-gb) KV_GB="$2"; shift 2 ;;
    --batch-tokens) BATCH_TOKENS="$2"; shift 2 ;;
    --slots) SLOTS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "error: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done

case "$ENGINE" in vllm|llamacpp) ;; *) echo "error: --engine must be vllm or llamacpp" >&2; exit 2 ;; esac
VENV="${VENV:-$HOME/.cache/vbt/cpu-venv-$ENGINE}"

if [ -z "$SERVED" ]; then
  base="$(basename "${MODEL%/}")"
  base="${base%.gguf}"
  SERVED="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
fi

lower_model="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
if [ -z "$TOOL_PARSER" ]; then
  case "$lower_model" in
    *qwen3.5*|*qwen3.6*|*qwen3.8*|*qwen3-coder*) TOOL_PARSER=qwen3_coder ;;
    *) TOOL_PARSER=hermes ;;
  esac
fi

install_vllm() {
  if command -v uv >/dev/null 2>&1; then
    [ -x "$VENV/bin/python" ] || uv venv --python "${PYTHON:-python3}" "$VENV"
    # torch==X+cpu lives on the PyTorch CPU index; vllm-cpu itself on PyPI.
    uv pip install --python "$VENV/bin/python" "vllm-cpu==$VLLM_CPU_VERSION" \
      --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
  else
    [ -x "$VENV/bin/python" ] || "${PYTHON:-python3}" -m venv "$VENV"
    "$VENV/bin/pip" install "vllm-cpu==$VLLM_CPU_VERSION" --extra-index-url https://download.pytorch.org/whl/cpu
  fi
  # vLLM's CPU kernels (_C_AVX512 / _C_AVX2) link libnuma.so.1; without it the engine dies with
  # "'_OpNamespace' '_C' object has no attribute 'init_cpu_memory_env'". Prefer the system
  # package (apt-get install libnuma1); otherwise unpack it next to the venv (no root needed).
  if ! ldconfig -p 2>/dev/null | grep -q 'libnuma\.so\.1'; then
    if command -v apt-get >/dev/null 2>&1 && command -v dpkg >/dev/null 2>&1; then
      tmp="$(mktemp -d)"
      (cd "$tmp" && apt-get download libnuma1 >/dev/null && dpkg -x libnuma1_*.deb "$VENV/libnuma")
      rm -rf "$tmp"
      echo "unpacked libnuma1 into $VENV/libnuma (added to LD_LIBRARY_PATH when serving)" >&2
    else
      echo "warning: libnuma.so.1 not found; install libnuma (e.g. libnuma1 / numactl-libs)" >&2
    fi
  fi
}

if [ "$INSTALL" = 1 ]; then
  if [ "$ENGINE" = vllm ]; then
    install_vllm
    "$VENV/bin/python" -c 'import vllm, torch, transformers; print("vllm", vllm.__version__, "torch", torch.__version__, "transformers", transformers.__version__)'
  else
    echo "llama.cpp: build llama-server from https://github.com/ggml-org/llama.cpp (cmake -B build && cmake --build build -j --target llama-server)" >&2
    echo "and set LLAMA_SERVER=/path/to/build/bin/llama-server" >&2
  fi
  exit 0
fi

if [ "$ENGINE" = vllm ]; then
  # The vllm-cpu wheel registers its distribution as `vllm-cpu`, so the `vllm` console script
  # fails with "PackageNotFoundError: No package metadata was found for vllm". The API server
  # module takes the same arguments as `vllm serve`.
  CMD=("$VENV/bin/python" -m vllm.entrypoints.openai.api_server
       --model "$MODEL" --served-model-name "$SERVED"
       --host "$HOST" --port "$PORT"
       --max-model-len "$MAX_LEN" --dtype bfloat16 --max-num-batched-tokens "$BATCH_TOKENS"
       --enable-prefix-caching --enable-prompt-tokens-details
       --reasoning-parser qwen3
       --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
  case "$lower_model" in
    *qwen3.5*|*qwen3.6*|*qwen3.8*) CMD+=(--language-model-only) ;;
  esac
  ENVS=(VLLM_CPU_KVCACHE_SPACE="$KV_GB" VLLM_CPU_OMP_THREADS_BIND="$THREADS")
  for d in "$VENV"/libnuma/usr/lib/*-linux-gnu; do
    [ -d "$d" ] && ENVS+=(LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
  done
else
  LS="${LLAMA_SERVER:-$(command -v llama-server || true)}"
  if [ -z "$LS" ] && [ "$DRY" = 0 ]; then
    echo "error: llama-server not found (set LLAMA_SERVER=/path/to/llama-server)" >&2
    exit 127
  fi
  # --jinja uses the GGUF's chat template (tool calls); --reasoning-format deepseek returns
  # reasoning in message.reasoning_content.
  CMD=("${LS:-llama-server}" -m "$MODEL" --alias "$SERVED" --host "$HOST" --port "$PORT"
       -c "$MAX_LEN" --jinja --reasoning-format deepseek -np "$SLOTS" --metrics)
  ENVS=()
fi
CMD+=(${EXTRA[@]+"${EXTRA[@]}"})

if [ "$DRY" = 1 ]; then
  printf '%s ' ${ENVS[@]+"${ENVS[@]}"}
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [ "$ENGINE" = vllm ] && [ ! -x "$VENV/bin/python" ]; then
  echo "error: no venv at $VENV; run $0 --install first (or pass --venv DIR)" >&2
  exit 1
fi

echo "serving $MODEL as '$SERVED' on http://$HOST:$PORT/v1 ($ENGINE, CPU)" >&2
exec env ${ENVS[@]+"${ENVS[@]}"} "${CMD[@]}"
