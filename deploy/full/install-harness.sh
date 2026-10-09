#!/usr/bin/env bash
# Install the harness into an environment made by install-env.sh. Shared by deploy/full/Dockerfile and
# deploy/full/vbt-harness.def.
#
#   install-harness.sh --prefix /opt/conda/envs/vbt --src /opt/vbt
#
# The harness is installed editable: it finds configs/, skills/ and third_party/TheVirtualBiotech next to src/.
# No dependency is installed here (the environment's locks hold them all); `pip check` and an import check of
# what the harness and the unmodified upstream MCP servers load fail the build when the environment is incomplete.
set -euo pipefail

PREFIX=/opt/conda/envs/vbt
SRC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    *) echo "install-harness.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$SRC" ] || { echo "install-harness.sh: --src is required" >&2; exit 2; }
PY="$PREFIX/bin/python"

for f in pyproject.toml src/vbt/cli.py third_party/TheVirtualBiotech/src/mcp_servers/target_mcp/server.py; do
  [ -f "$SRC/$f" ] || { echo "install-harness.sh: $SRC/$f is missing (is the upstream submodule checked out? git submodule update --init)" >&2; exit 1; }
done
"$PY" -m pip install --no-deps --no-build-isolation --no-input -e "$SRC"
"$PY" -m pip check
"$PY" - <<'EOF'
import importlib
mods = ["vbt.cli", "vbt.setup", "vbt.datalayer.service", "fastmcp", "mcp", "pyarrow", "pandas", "numpy", "requests",
        "pybioportal", "pronto", "cellxgene_census", "tiledbsoma", "anndata", "scanpy", "harmonypy", "decoupler",
        "liana", "lifelines", "pydeseq2", "gseapy", "rpy2", "anthropic", "starlette", "uvicorn", "jsonschema",
        "pypdf", "PIL", "yaml", "httpx", "pydantic"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{m}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("imports failed:\n  " + "\n  ".join(missing))
print(f"{len(mods)} imports OK")
EOF
