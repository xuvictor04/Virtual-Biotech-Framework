#!/usr/bin/env bash
# Bring up the full harness deployment with Docker (docs/DEPLOYMENT.md): vLLM serving the local model on the
# GPUs, SearxNG, the harness web UI, and the data / runs / projects / state volumes under $VBT_HOME.
#
#   export VBT_HOME=/srv/vbt                 # host directory: data/ runs/ projects/ state/ models/ (created)
#   mkdir -p "$VBT_HOME" && install -m 600 /dev/null "$VBT_HOME/secrets.env"
#   printf 'VBT_WEB_PASSWORD=...\n' >> "$VBT_HOME/secrets.env"
#   deploy/full/deploy.sh up                 # build (if needed) + setup + compose up + smoke: one command
#
# Subcommands:
#   build              build the harness image ($VBT_IMAGE, default vbt-harness:<git revision>, also :latest)
#   lock               regenerate deploy/full/conda-linux-64.lock and requirements.lock (Dockerfile target `lock`)
#   setup [ARGS...]    `vbt setup ARGS` in the harness image: probe, serving profile, host config, data,
#                      indexes, readiness checks, calibration (resumable; `setup --plan` prints the plan)
#   up                 build if the image is missing, setup, docker compose up -d, then the smoke step
#   smoke              `vbt setup --only smoke` in the running harness container
#   down | ps | logs [SERVICE] | config      docker compose passthrough (config: the merged configuration)
#   backup DEST        archive runs/, projects/ and state/ into DEST/vbt-backup-<UTC time>.tar.gz
#                      (data/ and models/ are re-acquirable: `vbt setup` fetches them again)
#   upgrade            git pull --recurse-submodules, build, setup (resumes), compose up -d, smoke
#
# Environment:
#   VBT_HOME           required: the host directory mounted at /srv/vbt in the containers
#   VBT_IMAGE          harness image (default vbt-harness:latest)
#   VBT_SECRETS_FILE   KEY=VALUE secrets for the harness container (default $VBT_HOME/secrets.env):
#                      VBT_WEB_PASSWORD, and as needed ANTHROPIC_API_KEY, HF_TOKEN, VLLM_API_KEY, NCBI_API_KEY ...
#   VBT_WEB_BIND / VBT_WEB_PORT   where the web UI is published (default 127.0.0.1:7860)
#   build only: HTTPS_PROXY, VBT_BUILD_CA_FILE (CA of a TLS-intercepting proxy, a build secret),
#               VBT_BUILD_NETWORK (e.g. host, when the proxy listens on the host's loopback),
#               BASE_IMAGE, MICROMAMBA_IMAGE (another registry or digest-pinned bases)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
COMPOSE_FILE="$HERE/compose.yaml"
VBT_IMAGE="${VBT_IMAGE:-vbt-harness:latest}"
export VBT_IMAGE

die() { echo "deploy.sh: $*" >&2; exit 1; }

need_home() {
  [ -n "${VBT_HOME:-}" ] || die "set VBT_HOME to the host directory for data, runs, projects and state (e.g. /srv/vbt)"
  case "$VBT_HOME" in /*) ;; *) die "VBT_HOME must be an absolute path (got $VBT_HOME)" ;; esac
  mkdir -p "$VBT_HOME"/data "$VBT_HOME"/runs "$VBT_HOME"/projects "$VBT_HOME"/state "$VBT_HOME"/models
  # the harness containers run as VBT_UID (default 10001, the image's user): it must own these directories
  local uid="${VBT_UID:-10001}" gid="${VBT_GID:-10001}" d
  for d in data runs projects state; do
    if [ "$(stat -c %u "$VBT_HOME/$d")" != "$uid" ]; then
      if [ "${READ_ONLY:-0}" = 1 ]; then
        :                                    # setup --plan/--probe/--status changes no ownership
      elif [ "$(id -u)" = 0 ]; then
        chown "$uid:$gid" "$VBT_HOME/$d"
      else
        echo "deploy.sh: note: $VBT_HOME/$d is not owned by uid $uid (the harness user); chown it or set VBT_UID" >&2
      fi
    fi
  done
  export VBT_HOME
  export VBT_SECRETS_FILE="${VBT_SECRETS_FILE:-$VBT_HOME/secrets.env}"
  if [ ! -f "$VBT_SECRETS_FILE" ]; then
    echo "deploy.sh: note: $VBT_SECRETS_FILE does not exist; the web UI needs VBT_WEB_PASSWORD there" >&2
  fi
}

compose_files() {
  local args=(-f "$COMPOSE_FILE")
  if [ -f "$VBT_HOME/state/compose.vllm.yaml" ]; then
    args+=(-f "$VBT_HOME/state/compose.vllm.yaml")
  fi
  printf '%s\n' "${args[@]}"
}

secret_value() {  # KEY's value in $VBT_SECRETS_FILE (KEY=VALUE lines, optionally quoted; never evaluated)
  [ -r "${VBT_SECRETS_FILE:-}" ] || return 0
  sed -n -e "s/^[[:space:]]*$1=//p" -e "s/^[[:space:]]*export[[:space:]][[:space:]]*$1=//p" "$VBT_SECRETS_FILE" |
    tail -n 1 | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

compose() {
  local files=() line key
  while IFS= read -r line; do files+=("$line"); done < <(compose_files)
  # the vllm service (state/compose.vllm.yaml) takes HF_TOKEN and VLLM_API_KEY from compose's environment: read
  # just these two from the secrets file, so that service never sees the harness's other secrets
  for key in HF_TOKEN VLLM_API_KEY; do
    [ -n "${!key:-}" ] || export "$key=$(secret_value "$key")"
  done
  docker compose --project-directory "$HERE" "${files[@]}" "$@"
}

cmd_build() {
  local rev upstream status
  rev="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
  upstream="$(git -C "$ROOT" ls-tree HEAD third_party/TheVirtualBiotech 2>/dev/null | awk '{print $3}')"
  status="$(git -C "$ROOT" submodule status third_party/TheVirtualBiotech 2>/dev/null || true)"
  case "$status" in
    -*) die "the upstream submodule is not checked out: git -C $ROOT submodule update --init" ;;
    +*) [ "${VBT_ALLOW_UPSTREAM_DRIFT:-0}" = 1 ] || \
          die "third_party/TheVirtualBiotech is not at the pinned commit $upstream: git submodule update (or VBT_ALLOW_UPSTREAM_DRIFT=1)" ;;
  esac
  [ -f "$ROOT/third_party/TheVirtualBiotech/src/mcp_servers/target_mcp/server.py" ] || \
    die "third_party/TheVirtualBiotech is empty: git -C $ROOT submodule update --init"
  local args=(build -f "$HERE/Dockerfile" -t "$VBT_IMAGE" --build-arg "VCS_REF=$rev"
              --build-arg "UPSTREAM_COMMIT=${upstream:-unknown}")
  [ "$VBT_IMAGE" = "vbt-harness:latest" ] && args+=(-t "vbt-harness:$rev")
  build_opts args
  docker "${args[@]}" "$ROOT"
}

build_opts() {
  local -n _a="$1"
  [ -n "${HTTPS_PROXY:-}" ] && _a+=(--build-arg "HTTPS_PROXY=$HTTPS_PROXY")
  [ -n "${VBT_BUILD_CA_FILE:-}" ] && _a+=(--secret "id=extra_ca,src=$VBT_BUILD_CA_FILE")
  [ -n "${VBT_BUILD_NETWORK:-}" ] && _a+=(--network "$VBT_BUILD_NETWORK")
  [ -n "${BASE_IMAGE:-}" ] && _a+=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
  [ -n "${MICROMAMBA_IMAGE:-}" ] && _a+=(--build-arg "MICROMAMBA_IMAGE=$MICROMAMBA_IMAGE")
  return 0
}

cmd_lock() {
  local out="$HERE/out/lock"
  rm -rf "$out"
  local args=(build -f "$HERE/Dockerfile" --target lock --build-arg LOCKED=0 --output "type=local,dest=$out")
  build_opts args
  docker "${args[@]}" "$ROOT"
  cp "$out/conda-linux-64.lock" "$out/requirements.lock" "$HERE/"
  rm -rf "$HERE/out"
  echo "updated deploy/full/conda-linux-64.lock and deploy/full/requirements.lock; build and test before committing"
}

gpu_info() {
  # The harness container sees no GPU; `vbt setup` reads the host's nvidia-smi output from the state volume.
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits \
      > "$VBT_HOME/state/nvidia-smi.csv" 2>/dev/null || rm -f "$VBT_HOME/state/nvidia-smi.csv"
  else
    rm -f "$VBT_HOME/state/nvidia-smi.csv"
  fi
}

read_only_setup() {  # setup --plan / --probe / --status only look: no service is started, no ownership changed
  local a
  for a in "$@"; do
    case "$a" in --plan|--probe|--status) return 0 ;; esac
  done
  return 1
}

cmd_setup() {
  if read_only_setup "$@"; then READ_ONLY=1; fi
  need_home
  # the host's nvidia-smi output in state/ is the one file a --plan writes: the harness container sees no GPU,
  # and the plan's serving profile depends on it
  gpu_info
  if [ "${READ_ONLY:-0}" != 1 ]; then
    # SearxNG first, so the probe's reachability check of SEARXNG_URL sees the real service (search is optional:
    # setup goes on without it)
    compose up -d --wait --wait-timeout 120 searxng ||
      echo "deploy.sh: note: searxng did not report healthy; WebSearch will not work until it does" >&2
  fi
  compose --profile setup run --rm --no-deps setup setup --deploy compose "$@"
}

cmd_up() {
  need_home
  if ! docker image inspect "$VBT_IMAGE" >/dev/null 2>&1; then
    cmd_build
  fi
  # the smoke step runs after `up`, against the running model server
  cmd_setup --skip smoke
  compose up -d
  cmd_smoke
}

wait_model() {
  # the first start downloads the weights and compiles CUDA graphs: wait for the healthcheck
  [ -f "$VBT_HOME/state/compose.vllm.yaml" ] || return 0
  local wait_s="${VBT_MODEL_WAIT_S:-3600}" start id status
  start="$(date +%s)"
  echo "deploy.sh: waiting up to ${wait_s} s for the vllm service to report healthy" >&2
  while :; do
    id="$(compose ps -q vllm 2>/dev/null || true)"
    status="$(docker inspect -f '{{.State.Health.Status}}' "$id" 2>/dev/null || true)"
    [ "$status" = healthy ] && return 0
    if [ $(( $(date +%s) - start )) -ge "$wait_s" ]; then
      echo "deploy.sh: vllm is '${status:-absent}' after ${wait_s} s; run 'deploy.sh smoke' once it is healthy" >&2
      return 1
    fi
    sleep 15
  done
}

cmd_smoke() {
  need_home
  wait_model || true
  compose exec -T vbt /opt/vbt/deploy/full/vbt-host setup --only smoke
}

cmd_backup() {
  need_home
  local dest="${1:-}"
  [ -n "$dest" ] || die "usage: deploy.sh backup DEST_DIR"
  mkdir -p "$dest"
  local file
  file="$dest/vbt-backup-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
  tar -C "$VBT_HOME" -czf "$file" runs projects state
  echo "$file"
}

cmd_upgrade() {
  need_home
  git -C "$ROOT" pull --recurse-submodules
  git -C "$ROOT" submodule update --init
  cmd_build
  cmd_setup --skip smoke
  compose up -d
  cmd_smoke
}

sub="${1:-}"
[ $# -gt 0 ] && shift
case "$sub" in
  build) cmd_build ;;
  lock) cmd_lock ;;
  setup) cmd_setup "$@" ;;
  up) cmd_up ;;
  smoke) cmd_smoke ;;
  down|ps|logs|config) need_home; compose "$sub" "$@" ;;
  backup) cmd_backup "$@" ;;
  upgrade) cmd_upgrade ;;
  ""|-h|--help|help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' ;;
  *) die "unknown subcommand $sub (see deploy.sh --help)" ;;
esac
