"""Environment policy for child processes (Bash, MCP servers) and secret redaction.

Agent-written code and third-party MCP servers must not see provider
credentials. ``child_env`` builds a child environment from an allow-list
instead of inheriting ``os.environ``; anything that looks like a secret is
dropped unless the caller names it explicitly in ``passthrough`` (for example
``NCBI_API_KEY`` for the PubMed server only). ``redact`` masks the values of
secret-looking variables in text before it is shown to a model or written to a
shareable record (trace, tool output).
"""

from __future__ import annotations

import fnmatch
import glob
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

#: Variables (exact names or fnmatch patterns) copied from the parent environment.
ALLOW: tuple[str, ...] = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "TMPDIR", "VIRTUAL_ENV", "JAVA_HOME",
    "LANG", "LANGUAGE", "LC_*", "TZ",
    "CONDA_*", "R_*", "RHOME",
    "PYTHONHASHSEED",
    "OMP_*", "MKL_*", "OPENBLAS_*", "NUMBA_*", "NUMEXPR_*", "VECLIB_MAXIMUM_THREADS",
    "LD_LIBRARY_PATH", "CUDA_*", "NVIDIA_*", "XDG_*", "MPLBACKEND", "MPLCONFIGDIR",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "no_proxy", "all_proxy",
)

#: Secret-looking names, matched case-insensitively. These win over ALLOW
#: (e.g. a hypothetical ``R_API_KEY`` or ``CONDA_TOKEN``) but lose to an
#: explicit ``passthrough`` entry.
SECRET: tuple[str, ...] = (
    "ANTHROPIC_*", "OPENAI_*", "AWS_*", "GOOGLE_*", "AZURE_*", "GEMINI_*", "HF_TOKEN",
    "*_API_KEY", "*APIKEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*", "*PASSWD*", "*CREDENTIAL*",
    "*_PRIVATE_KEY", "*ACCESS_KEY*",
)

#: Values shorter than this are never redacted (too likely to be ordinary text).
_MIN_SECRET_LEN = 8


def _match(name: str, patterns: Iterable[str], *, case_insensitive: bool = False) -> bool:
    key = name.upper() if case_insensitive else name
    for pat in patterns:
        if fnmatch.fnmatchcase(key, pat.upper() if case_insensitive else pat):
            return True
    return False


def is_secret_name(name: str) -> bool:
    """True when ``name`` looks like a credential variable."""
    return _match(name, SECRET, case_insensitive=True)


def child_env(base: Mapping[str, str] | None = None, *, passthrough: Iterable[str] = (),
              extra: Mapping[str, str] | None = None, home: str | os.PathLike | None = None,
              tmp: str | os.PathLike | None = None) -> dict[str, str]:
    """Return an allow-listed environment for a child process.

    Args:
        base: the parent environment (default ``os.environ``).
        passthrough: extra variable names or fnmatch patterns copied from
            ``base`` even when they look like secrets (``NCBI_API_KEY``).
        extra: explicit values set last (harness tool_env such as
            ``OPEN_TARGETS_DATA_PATH``, ``MCP_OUTPUT_DIR``); never filtered.
        home: if given, ``HOME`` points here (created). Python user
            site-packages and R user libraries under the original home stay
            importable through ``PYTHONUSERBASE``/``R_LIBS_USER``.
        tmp: if given, ``TMPDIR``/``TMP``/``TEMP`` point here (created).
    """
    base = os.environ if base is None else base
    passthrough = [p for p in passthrough if p]
    env: dict[str, str] = {}
    for name, value in base.items():
        if not isinstance(value, str) or value.startswith("()"):  # skip exported shell functions
            continue
        if _match(name, passthrough):
            env[name] = value
        elif _match(name, ALLOW) and not is_secret_name(name):
            env[name] = value

    if home is not None:
        home_path = Path(home)
        home_path.mkdir(parents=True, exist_ok=True)
        orig_home = base.get("HOME") or os.path.expanduser("~")
        env["HOME"] = str(home_path)
        if orig_home and Path(orig_home) != home_path:
            user_base = Path(orig_home) / ".local"
            if "PYTHONUSERBASE" not in env and "PYTHONUSERBASE" not in base and user_base.is_dir():
                env["PYTHONUSERBASE"] = str(user_base)
            if "R_LIBS_USER" not in env:
                libs = sorted(glob.glob(str(Path(orig_home) / "R" / "*-library" / "*")))
                if libs:
                    env["R_LIBS_USER"] = os.pathsep.join(libs)
    if tmp is not None:
        tmp_path = Path(tmp)
        tmp_path.mkdir(parents=True, exist_ok=True)
        for k in ("TMPDIR", "TMP", "TEMP"):
            env[k] = str(tmp_path)
    for k, v in (extra or {}).items():
        if v is not None:
            env[str(k)] = str(v)
    return env


def secret_values(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{value: name}`` for every secret-looking variable with a non-trivial value."""
    env = os.environ if env is None else env
    out: dict[str, str] = {}
    for name, value in env.items():
        if isinstance(value, str) and len(value.strip()) >= _MIN_SECRET_LEN and is_secret_name(name):
            out[value.strip()] = name
    return out


def redact(text, env: Mapping[str, str] | None = None):
    """Mask the values of secret-looking environment variables in ``text``.

    Non-string input is returned unchanged. Longer values are replaced first
    so a secret that contains another one is masked whole.
    """
    if not isinstance(text, str) or not text:
        return text
    for value, name in sorted(secret_values(env).items(), key=lambda kv: -len(kv[0])):
        if value in text:
            text = text.replace(value, f"[redacted:{name}]")
    return text


__all__ = ["ALLOW", "SECRET", "child_env", "is_secret_name", "redact", "secret_values"]
