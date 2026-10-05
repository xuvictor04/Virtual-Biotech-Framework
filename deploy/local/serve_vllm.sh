#!/usr/bin/env bash
# Start (or print) the bare-metal vLLM server for a serving profile of configs/local_models.yaml.
#
#   deploy/local/serve_vllm.sh --profile h100 --dry-run       # print the vllm serve command
#   deploy/local/serve_vllm.sh --profile h100                 # run it (exec, foreground)
#   deploy/local/serve_vllm.sh --profile h100 --bulk          # bulk-annotation-only server
#   deploy/local/serve_vllm.sh --profile 5090 --variant eager --dry-run
#   deploy/local/serve_vllm.sh --profile h100 --docker --dry-run   # the docker run line instead
#   deploy/local/serve_vllm.sh --detect                       # recommend a profile from nvidia-smi
#   deploy/local/serve_vllm.sh --list                         # list the profiles
#
# Without --profile the profile is picked from nvidia-smi. Other options: --variant NAME, --hf-id REPO,
# --engine-version 0.30.0 (drops --tool-strict-level), --data-parallel N, --host, --port,
# --vllm-arg=--extra-flag; see `--help`.
#
# The command is rendered by vbt.local.serve, which needs only Python 3.10+ and PyYAML, so this works
# from a vLLM environment without installing the harness: PYTHON selects the interpreter (default
# python3). Install vLLM itself in its own environment (pip install 'vllm==0.31.0'); let it pin
# transformers (>=5.10.4,<5.18.0) and never upgrade transformers separately.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: $PY not found (set PYTHON=/path/to/python3)" >&2
  exit 127
fi
if ! "$PY" -c 'import yaml' >/dev/null 2>&1; then
  echo "error: PyYAML is missing for $PY (pip install pyyaml)" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m vbt.local.serve "$@"
