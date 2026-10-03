"""One path and command policy for the built-in file tools and Bash.

:class:`PathPolicy` decides what an agent may read and write:

* **read**: the run directory plus ``runtime.read_roots`` (Bash additionally
  gets ``bash.system_read_roots``: the interpreter prefixes, ``/usr``, ...).
* **blocked_read** (``paths.blocked_read``): credentials and the project/upstream
  ``.env`` and ``.git``. It takes precedence over every allow.
* **write**: inside the run directory, never into harness records
  (``MANIFEST.json``, ``session_report.json``, ``README.md``, ``audit.html``,
  ``logs/``, ``evidence/``, ``inputs/``, ``report/``, ``.claude/``,
  ``memory/``) and, with ``bash.sandbox.protect_other_workspaces`` (default),
  only under the agent's own ``work/<agent>/`` (``work/_cso/`` for agents whose
  workspace is the run root) or the run's scratch ``.tmp``/``.home``.

Every root is stored in both its lexical and its symlink-resolved form, and a
candidate path is checked in both forms: a symlinked read root works, while a
symlink inside an allowed root that points elsewhere does not escape.

The Bash command policy (:func:`check_command`) ports the upstream
``agent_hooks`` guardrails: heredoc bodies are stripped before pattern matching
(and scanned only for package installs), the pattern groups are compiled
case-insensitively, ``rm``/``mv``/``ln``/``chmod``/... are allowed only inside the
agent's own workspace, every path token must pass the Bash read policy and
every redirect target the write policy. Command substitutions, ``bash -c``
and ``eval`` strings and heredocs fed to a shell are checked again as commands;
``NAME=value`` assignments and ``for`` lists are tracked and an unresolvable
expansion is denied where it may name a file; in-place editors, ``tar``,
``unzip``, ``cp -t`` and ``dd of=`` are write targets; the system/network
groups are also matched against each command word by basename
(``/bin/kill``). A command that cannot be parsed is denied with a pointer to
the Read/Write tools.

This is a guardrail on the command text, not an OS sandbox: interpreter code
(``python -c``, scripts) can still open files. ``bash.sandbox.os: bwrap``
(:func:`vbt.tools.builtin.bwrap_argv`) enforces the write and blocked-read
rules at the OS level.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:  # pragma: no cover
    from .base import ToolContext

#: Relative paths starting with one of these resolve against the run directory.
RUN_PREFIXES = ("work", "inputs", "evidence", "report", "logs", ".claude")

#: Run-root files no agent may write.
PROTECTED_FILES = frozenset({"MANIFEST.json", "session_report.json", "README.md", "audit.html"})

#: Run-level directories no agent may write into.
PROTECTED_DIRS = frozenset({"logs", "evidence", "inputs", "report", ".claude", "memory"})

#: Run-level scratch directories (Bash HOME and TMPDIR) any agent may write.
SCRATCH_DIRS = frozenset({".tmp", ".home"})

#: Work directory of agents whose workspace is the run root (CSO, reviewer).
RUN_ROOT_WORK = "_cso"

#: Lexical-only special files Bash may name (never resolved: /dev/stdout -> /proc/<pid>/fd/1).
SPECIAL_FILES = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/stdin", "/dev/zero",
                           "/dev/urandom", "/dev/random", "/dev/tty"})
_PROC_SELF_OK = re.compile(r"/proc/self/(?:status|stat|statm|limits|cgroup|cmdline|environ|fd(?:/\d+)?|"
                           r"meminfo|cpuinfo)\Z")


def default_blocked_read(config: Mapping[str, Any] | None = None) -> list[str]:
    """``paths.blocked_read`` default: provider credentials and repository secrets."""
    from ..config import PROJECT_ROOT

    project = Path(PROJECT_ROOT)
    out = [project / ".env", project / ".git"]
    upstream = ((config or {}).get("vars") or {}).get("upstream")
    if upstream:
        out += [Path(upstream) / ".env", Path(upstream) / ".git"]
    home = Path(os.path.expanduser("~"))
    out += [home / ".ssh", home / ".aws", home / ".config" / "gcloud", home / ".claude" / ".credentials.json",
            home / ".claude.json", home / ".netrc"]
    return [str(p) for p in out]


def default_system_read_roots(config: Mapping[str, Any] | None = None) -> list[str]:
    """``bash.system_read_roots`` default (the run's ``.tmp`` is always added)."""
    roots = ["/usr", "/bin", "/lib", "/lib64", "/etc/ssl", sys.prefix, sys.base_prefix]
    mcp_python = ((config or {}).get("vars") or {}).get("mcp_python")
    if mcp_python:
        exe = shutil.which(str(mcp_python)) or (str(mcp_python) if os.path.isabs(str(mcp_python)) else None)
        if exe:
            try:
                roots.append(str(Path(exe).resolve().parent.parent))
            except OSError:
                pass
    return roots


def _norm(p: str | os.PathLike) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(p))))


def _real(p: str | os.PathLike) -> str:
    try:
        return os.path.realpath(os.path.expanduser(str(p)))
    except (OSError, ValueError):
        return _norm(p)


def _forms(p: str | os.PathLike) -> tuple[str, ...]:
    a, b = _norm(p), _real(p)
    return (a,) if a == b else (a, b)


def _within(path: str, root: str) -> bool:
    if root == os.sep:
        return True
    return path == root or path.startswith(root + os.sep)


def _within_any(path: str, roots: Iterable[str]) -> bool:
    return any(_within(path, r) for r in roots)


def _root_forms(paths: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if p in (None, ""):
            continue
        for f in _forms(p):
            if f not in out:
                out.append(f)
    return out


def _cfg(config: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = config or {}
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


@dataclass
class PathPolicy:
    """Read/write decisions for one agent in one run (see the module docstring)."""

    run_dir: Path
    workspace: Path
    agent: str = ""
    read_roots: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    system_roots: list[str] = field(default_factory=list)
    protect_other_workspaces: bool = True

    def __post_init__(self) -> None:
        self.run_dir = Path(_norm(self.run_dir))
        self.workspace = Path(_norm(self.workspace))
        self._run_forms = _root_forms([self.run_dir])
        self._read = _root_forms([self.run_dir, *self.read_roots])
        self._blocked = _root_forms(self.blocked)
        self._system = _root_forms([*self.system_roots, self.run_dir / ".tmp"])
        own = RUN_ROOT_WORK if _forms(self.workspace)[-1] in self._run_forms else (self.agent or "_agent")
        self.own_dir_name = own

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_ctx(cls, ctx: "ToolContext") -> "PathPolicy":
        runtime = ctx.runtime
        config = getattr(runtime, "config", None) or {}
        blocked = _cfg(config, "paths", "blocked_read")
        if blocked is None or blocked == []:
            blocked = default_blocked_read(config) if blocked is None else []
        system = _cfg(config, "bash", "system_read_roots")
        if system is None:
            system = default_system_read_roots(config)
        protect = bool(_cfg(config, "bash", "sandbox", "protect_other_workspaces", default=True))
        return cls(run_dir=Path(ctx.run.dir), workspace=Path(ctx.workspace), agent=ctx.agent,
                   read_roots=[str(r) for r in (getattr(runtime, "read_roots", None) or [])],
                   blocked=[str(b) for b in blocked], system_roots=[str(s) for s in system],
                   protect_other_workspaces=protect)

    # ------------------------------------------------------------------ resolution

    @property
    def own_dir(self) -> Path:
        """The directory this agent may write in (``work/<agent>`` or ``work/_cso``)."""
        return self.run_dir / "work" / self.own_dir_name

    def resolve(self, raw: str | os.PathLike, *, for_read: bool = True, cwd: Path | None = None) -> Path:
        """Lexically normalised absolute path for a tool argument.

        ``~`` is expanded. A relative path whose first component is a run
        prefix (work, inputs, evidence, report, logs, .claude) resolves against
        the run directory; any other relative path against ``cwd`` (default the
        workspace), falling back to the run directory for reads when the
        workspace path does not exist.
        """
        s = os.path.expanduser(str(raw).strip())
        p = Path(s)
        if p.is_absolute():
            return Path(_norm(p))
        if cwd is None and p.parts and p.parts[0] in RUN_PREFIXES:
            return Path(_norm(self.run_dir / p))
        base = cwd or self.workspace
        cand = Path(_norm(base / p))
        if for_read and cwd is None and not os.path.lexists(cand):
            alt = Path(_norm(self.run_dir / p))
            if os.path.lexists(alt):
                return alt
        return cand

    # ------------------------------------------------------------------ reads

    def is_blocked(self, path: str | os.PathLike) -> bool:
        return any(_within_any(f, self._blocked) for f in _forms(path))

    def read_denial(self, path: str | os.PathLike, *, bash: bool = False) -> str | None:
        """None when ``path`` may be read, else the reason."""
        s = str(path)
        if bash and (s in SPECIAL_FILES or _PROC_SELF_OK.match(s)):
            return None
        if self.is_blocked(s):
            return f"access to {s} is blocked by policy (credentials and repository secrets are never readable)"
        roots = self._read + (self._system if bash else [])
        if all(_within_any(f, roots) for f in _forms(s)):
            return None
        return (f"read outside permitted roots: {s}. Allowed: the run directory {self.run_dir} and the read roots "
                f"{[r for r in self.read_roots]}")

    def allows_read(self, path: str | os.PathLike, *, bash: bool = False) -> bool:
        return self.read_denial(path, bash=bash) is None

    # ------------------------------------------------------------------ writes

    def _rel_parts(self, form: str) -> tuple[str, ...] | None:
        for root in self._run_forms:
            if _within(form, root):
                rel = os.path.relpath(form, root)
                return () if rel == "." else tuple(Path(rel).parts)
        return None

    def write_denial(self, path: str | os.PathLike, *, own_only: bool | None = None) -> str | None:
        """None when ``path`` may be written, else the reason.

        ``own_only``: require the agent's own work directory even when
        ``protect_other_workspaces`` is off (destructive Bash commands).
        """
        s = str(path)
        if self.is_blocked(s):
            return f"writing {s} is blocked by policy"
        protect = self.protect_other_workspaces if own_only is None else (own_only or self.protect_other_workspaces)
        for form in _forms(s):
            parts = self._rel_parts(form)
            if parts is None:
                return f"writes must stay inside the run directory {self.run_dir}; got {s}"
            if not parts:
                return f"cannot write the run directory itself ({self.run_dir})"
            if parts[0] in SCRATCH_DIRS:
                continue
            if len(parts) == 1 and parts[0] in PROTECTED_FILES or parts[0] in PROTECTED_DIRS:
                return (f"{'/'.join(parts)} is a harness record and cannot be written by agents; "
                        f"put your files under {self.own_dir}")
            if protect and not (len(parts) >= 2 and parts[0] == "work" and parts[1] == self.own_dir_name):
                return (f"agents write only inside their own workspace ({self.own_dir}); "
                        f"{s} belongs elsewhere (other agents' outputs are read-only)")
        return None

    def allows_write(self, path: str | os.PathLike, **kw: Any) -> bool:
        return self.write_denial(path, **kw) is None


# ===========================================================================
# Bash command policy
# ===========================================================================

_HEREDOC_RE = re.compile(r"<<(-?)[ \t]*(?:'([^'\n]+)'|\"([^\"\n]+)\"|\\?([A-Za-z_][\w.\-]*))")


def split_heredocs(command: str) -> tuple[str, list[str]]:
    """Return (shell text without heredoc bodies and shell comments, heredoc bodies).

    Quote-aware: ``<<`` inside quotes or comments is not a heredoc. An
    unterminated heredoc is left in the shell text (checked, not hidden).
    """
    out: list[str] = []
    bodies: list[str] = []
    pending: list[tuple[bool, str]] = []
    i, n, q = 0, len(command), None
    while i < n:
        c = command[i]
        if q == "'":
            out.append(c)
            if c == "'":
                q = None
            i += 1
            continue
        if c == "\\":
            out.append(command[i:i + 2])
            i += 2
            continue
        if q == '"':
            out.append(c)
            if c == '"':
                q = None
            i += 1
            continue
        if c in "'\"":
            q = c
            out.append(c)
            i += 1
            continue
        if c == "#" and (i == 0 or command[i - 1] in " \t\n;&|("):
            j = command.find("\n", i)
            i = n if j < 0 else j
            continue
        if command.startswith("<<", i) and not command.startswith("<<<", i):
            m = _HEREDOC_RE.match(command, i)
            if m:
                pending.append((m.group(1) == "-", m.group(2) or m.group(3) or m.group(4)))
                out.append(m.group(0))
                i = m.end()
                continue
        if c == "\n" and pending:
            out.append("\n")
            i += 1
            for strip_tabs, delim in pending:
                body: list[str] = []
                start = i
                found = False
                while i < n:
                    j = command.find("\n", i)
                    line = command[i:] if j < 0 else command[i:j]
                    i = n if j < 0 else j + 1
                    if (line.lstrip("\t") if strip_tabs else line) == delim:
                        found = True
                        break
                    body.append(line)
                if found:
                    bodies.append("\n".join(body))
                else:  # unterminated: keep it visible to the checks
                    i = start
                    break
            pending = []
            continue
        out.append(c)
        i += 1
    return "".join(out), bodies


def strip_heredocs(command: str) -> str:
    """Shell text without heredoc bodies (port of upstream ``_strip_heredocs``)."""
    return split_heredocs(command)[0]


def _newlines_to_separators(text: str) -> str:
    """Replace unquoted newlines with ' ; ' so shlex keeps command boundaries."""
    out: list[str] = []
    q = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and q != "'":
            if i + 1 < n and text[i + 1] == "\n":
                i += 2  # line continuation
                continue
            out.append(text[i:i + 2])
            i += 2
            continue
        if q:
            if c == q:
                q = None
        elif c in "'\"":
            q = c
        elif c == "\n":
            out.append(" ; ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def tokenize(shell_text: str) -> list[str]:
    """shlex tokens with ``;&|<>`` runs as separate punctuation tokens (raises ValueError)."""
    lexer = shlex.shlex(_newlines_to_separators(shell_text), posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


# ---------------------------------------------------------------- pattern groups

_GUIDANCE = {
    "pkg_install": "Package installation ({name}) is not allowed. All required packages are pre-installed in "
                   "the environment.",
    "destructive_fs": "Destructive filesystem command ({name}) is not allowed. Use Write/Edit tools for file "
                      "operations within your workspace.",
    "destructive_db": "Destructive database operation ({name}) is not allowed. Only read-only database queries "
                      "are permitted.",
    "system": "System command ({name}) is not allowed in this environment.",
    "network": "Network access ({name}) is disabled for this run (bash.network: false, e.g. to prevent "
               "information leakage). Use the provided data tools instead.",
    "custom": "Command blocked by policy ({name}).",
}

_CMD_START = r"(?<![\w./-])"
_CMD_END = r"(?![\w.-])"

DEFAULT_GROUPS: dict[str, dict[str, Any]] = {
    "pkg_install": {"enabled": True, "patterns": [
        [r"\bpip(?:3(?:\.\d+)?)?\s+install\b", "pip install"],
        [r"\bpython(?:3(?:\.\d+)?)?\s+-m\s+pip\s+install\b", "python -m pip install"],
        [r"\bpipx\s+install\b", "pipx install"],
        [r"\buv\s+(?:pip\s+(?:install|sync)|add|tool\s+install)\b", "uv pip/add"],
        [r"\b(?:micro)?mamba\s+(?:install|create|update)\b", "mamba/micromamba install"],
        [r"\bconda\s+(?:install|create|update|env\s+(?:create|update))\b", "conda install"],
        [r"\bapt(?:-get)?\s+install\b", "apt install"],
        [r"\byum\s+install\b", "yum install"],
        [r"\bdnf\s+install\b", "dnf install"],
        [r"\bnpm\s+(?:install|i|add|ci)\b", "npm install"],
        [r"\byarn\s+add\b", "yarn add"],
        [r"\bpnpm\s+(?:add|install|i)\b", "pnpm add"],
        [r"\bbrew\s+install\b", "brew install"],
        [r"\bgem\s+install\b", "gem install"],
        [r"\bcargo\s+install\b", "cargo install"],
        [r"\binstall\.packages\s*\(", "R install.packages"],
        [r"\bBiocManager::install\b", "BiocManager::install"],
        [r"\b(?:remotes|devtools)::install_\w+", "remotes/devtools install"],
    ]},
    "destructive_fs": {"enabled": True, "mode": "workspace_only",
                       "workspace_commands": ["rm", "rmdir", "unlink", "mv", "ln", "chmod", "chown"],
                       "patterns": [
                           [r"\bshred\b", "shred"],
                           [r"\bmkfs(?:\.\w+)?\b", "mkfs"],
                           [_CMD_START + r"dd\s+(?:\S+\s+)*?(?:if|of)=", "dd"],
                           [r"\bcurl\b.*\|\s*(?:sudo\s+)?(?:bash|sh|zsh|dash)\b", "curl pipe to shell"],
                           [r"\bwget\b.*\|\s*(?:sudo\s+)?(?:bash|sh|zsh|dash)\b", "wget pipe to shell"],
                           [r">\s*/dev/(?!(?:null|stdout|stderr|fd/[0-2])\b)", "write to /dev"],
                       ]},
    "destructive_db": {"enabled": True, "patterns": [
        [r"\bDROP\s+TABLE\b", "DROP TABLE"],
        [r"\bDROP\s+DATABASE\b", "DROP DATABASE"],
        [r"\bDROP\s+SCHEMA\b", "DROP SCHEMA"],
        [r"\bTRUNCATE\s+(?:TABLE\s+)?[\w.\"`]+\s*;", "TRUNCATE"],
        [r"\bDELETE\s+FROM\s+\S+\s*;", "DELETE FROM without WHERE clause"],
        [r"\bALTER\s+TABLE\s+\S+\s+DROP\b", "ALTER TABLE DROP"],
    ]},
    "system": {"enabled": True, "patterns": [
        [_CMD_START + r"kill" + _CMD_END, "kill"],
        [_CMD_START + r"killall" + _CMD_END, "killall"],
        [_CMD_START + r"pkill" + _CMD_END, "pkill"],
        [_CMD_START + r"sudo" + _CMD_END, "sudo"],
        [_CMD_START + r"shutdown" + _CMD_END, "shutdown"],
        [_CMD_START + r"reboot" + _CMD_END, "reboot"],
        [_CMD_START + r"init\s+[06]\b", "init (shutdown/reboot)"],
        [_CMD_START + r"crontab" + _CMD_END, "crontab"],
        [_CMD_START + r"ssh" + _CMD_END, "ssh"],
        [_CMD_START + r"scp" + _CMD_END, "scp"],
        [_CMD_START + r"rsync" + _CMD_END, "rsync"],
        [_CMD_START + r"nc" + _CMD_END, "netcat"],
        [_CMD_START + r"ncat" + _CMD_END, "ncat"],
    ]},
}

#: Package installs hidden in heredoc bodies (the only thing bodies are scanned for).
HEREDOC_PKG_PATTERNS: list[list[str]] = [
    [r"(?:subprocess\.\w+|os\.system|os\.popen|system2?|check_call|check_output|run)\s*\([^\n]*\bpip"
     r"(?:3(?:\.\d+)?)?\b[^\n]*\binstall\b", "pip install via subprocess/os.system"],
    [r"""['"]pip(?:3(?:\.\d+)?)?['"]\s*,\s*['"]install['"]""", "pip install via subprocess"],
    [r"""['"]-m['"]\s*,\s*['"]pip['"]\s*,\s*['"]install['"]""", "python -m pip install via subprocess"],
    [r"^\s*[!%]\s*pip\s+install\b", "pip install magic"],
    [r"\binstall\.packages\s*\(", "R install.packages"],
    [r"\bBiocManager::install\b", "BiocManager::install"],
    [r"\b(?:remotes|devtools)::install_\w+", "remotes/devtools install"],
]

#: bash.network: false.
NETWORK_PATTERNS: list[list[str]] = [
    [_CMD_START + r"(?:curl|wget|aria2c|ftp|telnet|lynx)" + _CMD_END, "curl/wget"],
    [r"\bhttps?://", "http(s) URL"],
    [r"\b(?:import|from)\s+(?:requests|httpx|urllib\d?|aiohttp|http\.client|socket|ftplib)\b",
     "requests/httpx/urllib import"],
    [r"\b(?:requests|httpx|urllib\.request|urllib3|aiohttp)\.\w+\s*\(", "requests/httpx/urllib call"],
    [r"\bdownload\.file\s*\(", "R download.file"],
    [r"\b(?:httr|httr2|RCurl|curl)::", "R http client"],
]


@dataclass
class _Rule:
    rx: re.Pattern
    name: str
    group: str
    message: str


def _compile(entries: Iterable[Any], group: str, message: str | None) -> list[_Rule]:
    out: list[_Rule] = []
    for e in entries or []:
        if isinstance(e, Mapping):
            pat, name = e.get("pattern"), e.get("name") or e.get("pattern")
        elif isinstance(e, (list, tuple)):
            pat, name = e[0], (e[1] if len(e) > 1 else e[0])
        else:
            pat, name = e, e
        if not pat:
            continue
        try:
            rx = re.compile(str(pat), re.IGNORECASE | re.MULTILINE)
        except re.error:
            continue
        out.append(_Rule(rx, str(name), group, (message or _GUIDANCE.get(group, _GUIDANCE["custom"]))))
    return out


@dataclass
class CommandPolicy:
    """Compiled ``bash.block.*`` groups plus the network switch."""

    shell_rules: list[_Rule]
    heredoc_rules: list[_Rule]
    network_rules: list[_Rule]
    fs_mode: str = "workspace_only"
    workspace_commands: frozenset[str] = frozenset()
    fs_enabled: bool = True
    paths: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "CommandPolicy":
        bash = _cfg(config, "bash", default={}) or {}
        block = bash.get("block") or {}
        shell: list[_Rule] = []
        heredoc: list[_Rule] = []
        fs_mode, wcmds, fs_enabled = "workspace_only", frozenset(), True
        for group, default in DEFAULT_GROUPS.items():
            user = block.get(group)
            if user is False:
                user = {"enabled": False}
            g = {**default, **(user or {})}
            enabled = bool(g.get("enabled", True))
            if group == "destructive_fs":
                fs_enabled = enabled
                fs_mode = str(g.get("mode") or "workspace_only")
                wcmds = frozenset(str(c) for c in (g.get("workspace_commands") or []))
            if not enabled:
                continue
            shell += _compile(g.get("patterns") or [], group, g.get("message"))
            if group == "destructive_fs" and fs_mode == "block":
                shell += _compile([[rf"{_CMD_START}{re.escape(c)}{_CMD_END}", c] for c in sorted(wcmds)], group,
                                  g.get("message"))
            if group == "pkg_install":
                heredoc += _compile(g.get("heredoc_patterns") or HEREDOC_PKG_PATTERNS, group, g.get("message"))
        # legacy flat list (configs/default.yaml bash.blocked_patterns)
        shell += _compile(bash.get("blocked_patterns") or [], "custom",
                          "Command blocked by policy ({name}): do not install packages or leave the workspace.")
        network = [] if bash.get("network", True) else _compile(
            bash.get("network_patterns") or NETWORK_PATTERNS, "network", None)
        paths = bool(_cfg(bash, "sandbox", "paths", default=True))
        return cls(shell, heredoc, network, fs_mode, wcmds if fs_mode == "workspace_only" else frozenset(),
                   fs_enabled, paths)


class CommandDenied(Exception):
    """A Bash command rejected by policy (message is shown to the model)."""

    def __init__(self, message: str, group: str = "", rule: str = ""):
        super().__init__(message)
        self.group = group
        self.rule = rule


_REDIRECTS_WRITE = (">", ">>", "&>", "&>>", ">|", "<>", ">&", "&>|")
_REDIRECTS_READ = ("<",)
_SEPARATORS = (";", ";;", "&&", "||", "|", "&", "|&")
_WRAPPERS = frozenset({"sudo", "env", "nohup", "time", "nice", "command", "exec", "builtin", "stdbuf", "ionice",
                       "then", "do", "else", "elif", "if", "while", "until", "!", "{", "xargs"})
#: Commands whose path arguments are written ("all") or whose last argument is ("last").
WRITE_COMMANDS = {"tee": "all", "touch": "all", "mkdir": "all", "truncate": "all", "mkfifo": "all",
                  "cp": "last", "install": "last", "rsync": "last",
                  "gzip": "all", "gunzip": "all", "bzip2": "all", "bunzip2": "all", "xz": "all", "unxz": "all",
                  "zstd": "all", "unzstd": "all"}
#: Compressors write next to (or replace) their inputs unless they only list, test or write stdout.
_COMPRESSORS = frozenset({"gzip", "gunzip", "bzip2", "bunzip2", "xz", "unxz", "zstd", "unzstd"})
_COMPRESS_READONLY = frozenset({"-c", "--stdout", "--to-stdout", "-l", "--list", "-t", "--test"})
#: Commands taking ``-t DIR`` / ``--target-directory=DIR`` as the destination.
_TARGET_DIR_COMMANDS = frozenset({"cp", "mv", "install", "ln"})

#: Marks a ``$`` or backtick the shell does not expand (single-quoted or backslash-escaped).
_LIT_DOLLAR = ""
_LIT_TICK = ""


def _protect_literals(text: str) -> str:
    """Replace ``$``/backticks that the shell leaves literal by sentinels (shlex drops the quotes)."""
    out: list[str] = []
    i, n, q = 0, len(text), None
    while i < n:
        c = text[i]
        if q == "'":
            if c == "'":
                q = None
            out.append(_LIT_DOLLAR if c == "$" else _LIT_TICK if c == "`" else c)
            i += 1
            continue
        if c == "\\" and i + 1 < n and text[i + 1] in "$`":
            out.append(_LIT_DOLLAR if text[i + 1] == "$" else _LIT_TICK)
            i += 2
            continue
        if c == "\\":
            out.append(text[i:i + 2])
            i += 2
            continue
        if c == '"':
            q = None if q == '"' else '"'
        elif c == "'" and q is None:
            q = "'"
        out.append(c)
        i += 1
    return "".join(out)


def _unprotect(s: str) -> str:
    return s.replace(_LIT_DOLLAR, "$").replace(_LIT_TICK, "`")


def _shell_c_string(rest: list[str]) -> tuple[bool, str | None]:
    """For a shell's arguments: (has -c, the -c command string or the script path or None)."""
    has_c, skip = False, False
    for t in rest:
        if skip:
            skip = False
            continue
        if t in _REDIRECTS_WRITE or t in _REDIRECTS_READ or t in ("<<", "<<-", "<<<"):
            skip = True
            continue
        if t == "--":
            continue
        if t.startswith(("-", "+")) and len(t) > 1 and not t.startswith("--"):
            if "c" in t[1:]:
                has_c = True
            if t in ("-o", "+o", "-O", "+O"):
                skip = True
            continue
        if t.startswith("--"):
            continue
        return has_c, t
    return has_c, None


def _inplace_files(word: str, args: list[str]) -> list[str] | None:
    """Files edited in place by sed -i / perl -i / awk -i inplace; None when not in-place."""
    takes_value = {"sed": {"-e", "-f", "-l", "--expression", "--file", "--line-length"},
                   "perl": {"-e", "-E", "-I", "-M", "-m"},
                   "awk": {"-f", "-v", "-F", "-e", "-i", "-l", "--file", "--assign", "--field-separator",
                           "--source", "--include", "--load"}}[word]
    script_opts = {"sed": {"-e", "-f", "--expression", "--file"}, "perl": {"-e", "-E"},
                   "awk": {"-f", "-e", "--file", "--source"}}[word]
    inplace, has_script, skip_for = False, False, None
    pos: list[str] = []
    end_opts = False
    for t in args:
        if skip_for is not None:
            if word == "awk" and skip_for in ("-i", "--include") and t in ("inplace", "inplace.awk"):
                inplace = True
            skip_for = None
            continue
        if not end_opts and t == "--":
            end_opts = True
            continue
        if not end_opts and t.startswith("-") and t != "-":
            name = t.split("=", 1)[0]
            if name in script_opts:
                has_script = True
            if word == "awk" and name in ("--include",) and t.endswith(("=inplace", "=inplace.awk")):
                inplace = True
            if word in ("sed", "perl") and (name in ("--in-place",) or
                                            (not t.startswith("--") and "i" in t[1:].split("e")[0])):
                inplace = True
            if word == "perl" and not t.startswith("--") and "e" in t[1:]:
                has_script = True
            if name in takes_value and "=" not in t:
                skip_for = name
            continue
        pos.append(t)
    if not inplace:
        return None
    return pos if has_script else pos[1:]


def _tar_writes(args: list[str]) -> list[str]:
    """Paths a tar invocation writes: the extraction directory or the created archive."""
    extract = create = False
    archive = directory = None
    i = 0
    while i < len(args):
        t = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else None
        if t.startswith("--"):
            name, _, val = t.partition("=")
            if name in ("--extract", "--get"):
                extract = True
            elif name in ("--create", "--append", "--update", "--concatenate"):
                create = True
            elif name == "--directory":
                directory = val or nxt
                i += 0 if val else 1
            elif name == "--file":
                archive = val or nxt
                i += 0 if val else 1
        elif t.startswith("-") or i == 0:
            flags = t.lstrip("-")
            if t.startswith("-") or re.fullmatch(r"[A-Za-z]+", flags or "-"):
                for k, ch in enumerate(flags):
                    if ch == "x":
                        extract = True
                    elif ch in "cruA":
                        create = True
                    elif ch in "fC":
                        val = flags[k + 1:] or nxt
                        if ch == "f":
                            archive = val
                        else:
                            directory = val
                        if not flags[k + 1:]:
                            i += 1
                        break
        i += 1
    out: list[str] = []
    if extract:
        out.append(directory or ".")
    if create and archive and archive != "-":
        out.append(archive)
    return out


def _target_dir(args: list[str]) -> str | None:
    for i, t in enumerate(args):
        if t in ("-t", "--target-directory"):
            return args[i + 1] if i + 1 < len(args) else ""
        if t.startswith("--target-directory="):
            return t.split("=", 1)[1]
        if t.startswith("-t") and not t.startswith("--") and len(t) > 2:
            return t[2:]
    return None
#: Shells whose ``-c`` string is checked again as a command.
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
#: Commands whose arguments never name files (an unresolvable $VAR there is harmless).
_NO_FILE_ARGS = frozenset({"echo", "printf", "true", "false", ":", "test", "[", "[[", "expr", "let", "return",
                           "exit", "shift", "set", "unset", "export", "local", "declare", "readonly", "typeset"})
#: Special parameters that expand to numbers or flags, never to paths.
_SPECIAL_PARAMS = re.compile(r"\$(?:[?#$!-]|\{[?#$!-]\})")
_VAR_RE = re.compile(r"\$(?:\{([A-Za-z_]\w*)(?:(:?[-=])([^}]*))?\}|([A-Za-z_]\w*))")
_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
_MAX_DEPTH = 4


def _expand_vars(token: str, env: Mapping[str, str]) -> str | None:
    """Expand $VAR, ${VAR} and ${VAR:-word} from ``env``; None when something stays unresolvable."""
    if "`" in token or "$(" in token:
        return None
    token = _SPECIAL_PARAMS.sub("0", token)

    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(4)
        if name in env and not (m.group(2) and m.group(2).startswith(":") and env[name] == ""):
            return env[name]
        if m.group(2):  # ${NAME:-word} / ${NAME-word} / ${NAME:=word}: the default is what expands
            return m.group(3) or ""
        raise KeyError(name)

    try:
        out = _VAR_RE.sub(sub, token)
    except KeyError:
        return None
    return None if "$" in out else out


def command_substitutions(text: str) -> list[str]:
    """Inner command text of every ``$(...)``, backtick span and ``<(...)``/``>(...)`` outside single quotes.

    ``$((...))`` arithmetic is skipped. Nested substitutions are returned by
    the recursive check of their parent's text.
    """
    out: list[str] = []
    i, n, q = 0, len(text), None
    while i < n:
        c = text[i]
        if q == "'":
            if c == "'":
                q = None
            i += 1
            continue
        if c == "\\":
            i += 2
            continue
        if c == "'" and q is None:
            q = "'"
            i += 1
            continue
        if c == '"':
            q = None if q == '"' else '"'
            i += 1
            continue
        opener = (c == "$" or (c in "<>" and q is None)) and text.startswith("(", i + 1)
        if opener and not text.startswith("((", i + 1):
            depth, j, iq = 1, i + 2, None
            while j < n and depth:
                d = text[j]
                if iq:
                    if d == iq:
                        iq = None
                elif d in "'\"":
                    iq = d
                elif d == "\\":
                    j += 1
                elif d == "(":
                    depth += 1
                elif d == ")":
                    depth -= 1
                j += 1
            out.append(text[i + 2:j - 1] if depth == 0 else text[i + 2:])
            i = j
            continue
        if c == "`":
            j = text.find("`", i + 1)
            out.append(text[i + 1:] if j < 0 else text[i + 1:j])
            i = n if j < 0 else j + 1
            continue
        i += 1
    return out


def _expand_home(token: str, env: Mapping[str, str]) -> str:
    if token == "~" or token.startswith("~/"):
        home = env.get("HOME") or os.path.expanduser("~")
        return home + token[1:]
    return token


def _is_pathlike(tok: str) -> bool:
    return "/" in tok or tok.startswith(".") or tok.startswith("~")


def _segments(tokens: list[str]) -> list[list[str]]:
    segs: list[list[str]] = [[]]
    for t in tokens:
        if t in _SEPARATORS:
            segs.append([])
        else:
            segs[-1].append(t)
    return [s for s in segs if s]


def _command_word(seg: list[str]) -> tuple[int, str]:
    """Index and basename of the segment's command word (wrappers and assignments skipped)."""
    i = 0
    while i < len(seg):
        t = seg[i].lstrip("(")
        if not t or _ASSIGN_RE.match(t) or t in _WRAPPERS or ("/" in t and os.path.basename(t) in _WRAPPERS):
            i += 1
            continue
        if t == "timeout":
            i += 1
            while i < len(seg) and seg[i].startswith("-"):
                i += 1
            i += 1  # the duration
            continue
        if t.startswith("-") and i > 0:
            i += 1
            continue
        return i, os.path.basename(t)
    return -1, ""


def _segments_sep(tokens: list[str]) -> list[tuple[str, list[str]]]:
    """Segments with the separator that precedes each ("" for the first)."""
    out: list[tuple[str, list[str]]] = [("", [])]
    for t in tokens:
        if t in _SEPARATORS:
            out.append((t, []))
        else:
            out[-1][1].append(t)
    return [(s, seg) for s, seg in out if seg]


#: Interpreters whose script argument is scanned against the network rules (bash.network: false).
_SCRIPT_INTERPRETERS = re.compile(r"^(?:python(?:\d+(?:\.\d+)?)?|pypy3?|ipython3?|Rscript|R|node|perl|ruby|julia|"
                                  r"bash|sh|zsh|dash|ksh|source|\.)$")
_SCRIPT_SCAN_MAX = 2_000_000


def _script_operand(word: str, args: list[str]) -> str | None:
    """The script file an interpreter command runs, or None (inline code, module, stdin)."""
    if word == "R":
        for k, a in enumerate(args):
            if a in ("-f", "--file") and k + 1 < len(args):
                return args[k + 1]
            if a.startswith("--file="):
                return a.split("=", 1)[1]
        return None
    if word.startswith(("python", "pypy", "ipython")):
        no_script, with_value = {"-c", "-m"}, {"-W", "-X", "-Q"}
    elif word in ("node",):
        no_script, with_value = {"-e", "--eval", "-p", "--print"}, {"-r", "--require"}
    elif word in ("perl", "ruby"):
        no_script, with_value = {"-e", "-E"}, set()
    elif word in ("Rscript", "julia"):
        no_script, with_value = {"-e", "--eval"}, {"--default-packages"}
    else:  # shells, source, .
        no_script, with_value = {"-c", "-s"}, {"-o", "-O"}
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a in no_script:
            return None
        if a == "--":
            continue
        if a.startswith("-"):
            skip = a in with_value
            continue
        return a
    return None

def _deny(rule: _Rule) -> CommandDenied:
    return CommandDenied(f"[SECURITY] Command blocked: {rule.message.format(name=rule.name)}", rule.group, rule.name)


def check_command(command: str, policy: PathPolicy, cmd_policy: CommandPolicy, *,
                  env: Mapping[str, str] | None = None, cwd: Path | None = None, _depth: int = 0) -> None:
    """Raise :class:`CommandDenied` when ``command`` breaks the Bash policy.

    Command substitutions, ``bash/sh -c`` strings, ``eval`` arguments and
    heredocs fed to a shell are checked again as commands of their own.
    ``NAME=value`` assignments and ``for NAME in ...`` lists are tracked so a
    path held in a variable is still checked; an expansion that cannot be
    resolved is denied wherever it may name a file. This is a guardrail on the
    command text: interpreter code (``python -c``, scripts) can still open
    files, so ``bash.sandbox.os: bwrap`` adds OS enforcement.
    """
    if _depth > _MAX_DEPTH:
        raise CommandDenied("Command nests shells or substitutions too deeply; write a script file in your "
                            "workspace and run it instead.", "paths", "nesting")
    env = dict(env or {})
    shell_text, bodies = split_heredocs(command)

    for rule in cmd_policy.shell_rules:
        if rule.rx.search(shell_text):
            raise _deny(rule)
    for body in bodies:
        for rule in cmd_policy.heredoc_rules:
            if rule.rx.search(body):
                raise _deny(rule)
    for rule in cmd_policy.network_rules:
        if rule.rx.search(command):
            raise _deny(rule)

    def nested(text: str, at: Path | None) -> None:
        if at is None:
            raise CommandDenied("Nested command after an unverifiable cd; use absolute paths", "paths", "cwd")
        check_command(text, policy, cmd_policy, env=venv, cwd=at, _depth=_depth + 1)

    check_paths = cmd_policy.paths
    destructive = cmd_policy.fs_enabled and bool(cmd_policy.workspace_commands)
    try:
        tokens = tokenize(_protect_literals(shell_text))
    except ValueError:
        if not (check_paths or destructive):
            return
        raise CommandDenied("Cannot safely parse shell command; use Read/Write tools for file operations.",
                            "paths", "parse") from None

    cwd0 = Path(_norm(cwd or policy.workspace))
    state: dict[str, Any] = {"cwd": cwd0}
    venv: dict[str, str] = {**env, "PWD": str(cwd0)}
    loops: dict[str, list[str]] = {}

    for inner in command_substitutions(shell_text):
        nested(inner, cwd0)

    def resolve_tok(tok: str) -> Path | None:
        t = _expand_vars(tok, {**venv, "PWD": str(state["cwd"] or "")})
        if t is None:
            return None
        t = _unprotect(_expand_home(t, venv))
        p = Path(t)
        if p.is_absolute():
            return Path(_norm(p))
        if state["cwd"] is None:
            return None
        return Path(_norm(state["cwd"] / p))

    def resolve_all(tok: str) -> list[Path] | None:
        """Every path ``tok`` can name (one per loop item); None when an expansion is unknown."""
        names = [n for n in loops if re.search(r"\$(?:\{%s\}|%s(?!\w))" % (n, n), tok)]
        variants = [tok]
        for n in names:
            rx = re.compile(r"\$(?:\{%s\}|%s(?!\w))" % (n, n))
            variants = [rx.sub(lambda _m, it=it: it, v) for v in variants for it in loops[n]][:64]
        out: list[Path] = []
        for v in variants:
            p = resolve_tok(v)
            if p is None:
                return None
            out.append(p)
        return out

    def has_expansion(tok: str) -> bool:
        return "$" in tok or "`" in tok

    def read_check(tok: str, word: str = "") -> None:
        if not check_paths:
            return
        ps = resolve_all(tok)
        if ps is None:
            if has_expansion(tok):
                if word in _NO_FILE_ARGS:
                    return
                raise CommandDenied(f"Command uses an expansion whose value cannot be verified: {_unprotect(tok)}; "
                                    f"use literal paths (or a for-loop over literal paths)", "paths", "expansion")
            raise CommandDenied(f"Command references a relative path after an unverifiable cd: {_unprotect(tok)}; "
                                f"use absolute paths", "paths", "cwd")
        for p in ps:
            if str(p) in SPECIAL_FILES:
                continue
            why = policy.read_denial(p, bash=True)
            if why:
                raise CommandDenied(f"Command references path outside allowed directories: {p} ({why})",
                                    "paths", "read")

    def write_check(tok: str, *, own_only: bool = False, what: str = "Shell output") -> None:
        if tok in SPECIAL_FILES:
            return
        ps = resolve_all(tok)
        if ps is None:
            raise CommandDenied(f"{what} target cannot be verified ({_unprotect(tok)}); use a literal path inside "
                                f"your workspace {policy.own_dir}", "paths", "write")
        for p in ps:
            if str(p) in SPECIAL_FILES:
                continue
            why = policy.write_denial(p, own_only=own_only or None)
            if why is None and own_only and _forms(p)[0] == _norm(policy.own_dir):
                why = "refusing to remove or move your whole workspace"
            if why:
                raise CommandDenied(f"{what} is restricted to your workspace: {p} ({why})", "paths", "write")

    def scan_script(word: str, cmd_tok: str, args: list[str]) -> None:
        """bash.network: false also applies to a script file the command runs (``python fetch.py``,
        ``Rscript x.R``, ``./fetch.sh``), not only to the command text. Still a guardrail: code the
        script imports from elsewhere is not followed; bash.network_isolation is the OS control."""
        cands: list[str] = []
        if _SCRIPT_INTERPRETERS.match(word):
            op = _script_operand(word, [_unprotect(a) for a in args])
            if op:
                cands.append(op)
        if "/" in _unprotect(cmd_tok.lstrip("(")):
            cands.append(_unprotect(cmd_tok.lstrip("(")))  # ./fetch.py run directly
        for c in cands:
            p = resolve_tok(c)
            if p is None:
                continue
            try:
                if not p.is_file() or p.stat().st_size > _SCRIPT_SCAN_MAX:
                    continue
                text = p.read_text(errors="replace")
            except OSError:
                continue
            for rule in cmd_policy.network_rules:
                if rule.rx.search(text):
                    exc = _deny(rule)
                    raise CommandDenied(f"{exc} (in script {c})", exc.group, exc.rule)

    for sep, seg in _segments_sep(tokens):
        idx, word = _command_word(seg)
        # rules again on the command word itself: /bin/kill, /usr/bin/ssh, k''ill, env /bin/kill ...
        upto = len(seg) if idx < 0 else idx + 1
        rebuilt = shlex.join([os.path.basename(_unprotect(t.lstrip("("))) if k < upto and "/" in t
                              else _unprotect(t) for k, t in enumerate(seg)])
        for rule in (*cmd_policy.shell_rules, *cmd_policy.network_rules):
            if rule.rx.search(rebuilt):
                raise _deny(rule)
        args = [x.strip("()") for x in seg[idx + 1:]] if idx >= 0 else []
        if cmd_policy.network_rules and idx >= 0:
            scan_script(word, seg[idx], args)

        # assignments (NAME=value, export NAME=value) and for-loop lists feed later expansions
        assigns = seg[:idx] if idx >= 0 else seg
        if word in ("export", "declare", "local", "readonly", "typeset"):
            assigns = list(assigns) + args
        for a in assigns:
            a = a.lstrip("(")
            if _ASSIGN_RE.match(a):
                name, value = a.split("=", 1)
                v = _expand_vars(value, {**venv, "PWD": str(state["cwd"] or "")})
                loops.pop(name, None)
                if v is None:
                    venv.pop(name, None)
                else:
                    venv[name] = _expand_home(v, venv)
        if word == "for" and len(args) >= 2:
            name = args[0]
            items = args[2:] if len(args) > 2 and args[1] == "in" else []
            venv.pop(name, None)
            loops[name] = items or ["$" + "@"]
        elif word in ("read", "mapfile", "readarray"):
            for n in args:
                if re.fullmatch(r"[A-Za-z_]\w*", n):
                    venv.pop(n, None)
                    loops.pop(n, None)

        # nested shells and eval run their string as a command
        if word == "eval":
            nested(" ".join(_unprotect(a) for a in args), state["cwd"])
        for k, t in enumerate(seg):
            if os.path.basename(_unprotect(t.lstrip("("))) in _SHELLS:
                has_c, arg = _shell_c_string(seg[k + 1:])
                if has_c and arg is not None:
                    nested(_unprotect(arg), state["cwd"])
                elif k == idx and arg is None:  # the shell reads its commands from stdin
                    for m, h in enumerate(seg[k + 1:-1], start=k + 1):
                        if h == "<<<":
                            nested(_unprotect(seg[m + 1]), state["cwd"])
                    for body in bodies:
                        nested(body, state["cwd"])
                    if sep in ("|", "|&"):
                        raise CommandDenied("Piping text into a shell is not allowed; write the commands to a "
                                            "script in your workspace or run them directly.", "paths", "nesting")

        # --- path tokens and redirect targets (upstream _bash_path_denial)
        skip_next = False
        for j, raw in enumerate(seg):
            if skip_next:
                skip_next = False
                continue
            tok = raw.strip("()")
            if not tok:
                continue
            if raw in ("<<", "<<-", "<<<"):
                skip_next = True  # heredoc delimiter or here-string
                continue
            if raw in _REDIRECTS_WRITE:
                target = seg[j + 1] if j + 1 < len(seg) else ""
                skip_next = True
                if raw == ">&" and (target.isdigit() or target == "-"):
                    continue
                if not target:
                    raise CommandDenied("Cannot safely parse shell redirection; use the Write tool.", "paths",
                                        "parse")
                if check_paths:
                    write_check(target)
                    read_check(target)
                continue
            if raw in _REDIRECTS_READ:
                continue
            if raw.isdigit() and j + 1 < len(seg) and seg[j + 1] in _REDIRECTS_WRITE + _REDIRECTS_READ:
                continue  # fd number of a redirection (2>file)
            if _ASSIGN_RE.match(tok) and (idx < 0 or j < idx or word in ("export", "declare", "local",
                                                                          "readonly", "typeset")):
                continue  # NAME=value: the value is checked where the variable is used
            if tok.startswith("-"):
                if "=" in tok:
                    val = tok.split("=", 1)[1]
                    if val.startswith(("/", "~")) or has_expansion(val):
                        read_check(val, word)
                continue
            if _is_pathlike(tok) or (has_expansion(tok) and word not in _NO_FILE_ARGS):
                read_check(tok, word)

        if check_paths and idx >= 0:
            # file-creating commands: their targets obey the write policy like redirects
            if word in WRITE_COMMANDS and not (word in _COMPRESSORS and any(a in _COMPRESS_READONLY for a in args)):
                targets = _args_only(args)
                tdir = _target_dir(args) if word in _TARGET_DIR_COMMANDS else None
                if tdir is not None:
                    targets = [tdir]
                elif WRITE_COMMANDS[word] == "last":
                    targets = targets[-1:] if len(targets) >= 2 else []
                for t in targets:
                    write_check(t, what=word)
            elif word in ("mv", "ln") and _target_dir(args) is not None:
                write_check(_target_dir(args) or "", own_only=True, what=word)
            if word in ("sed", "perl", "awk", "gawk"):
                files = _inplace_files("awk" if word == "gawk" else word, args)
                for t in files or []:
                    write_check(t, what=f"{word} in-place edit")
            elif word in ("tar", "bsdtar"):
                for t in _tar_writes(args):
                    write_check(t, what="tar")
            elif word == "unzip" and not any(a in ("-l", "-t", "-p", "-Z", "-v") for a in args):
                dest = args[args.index("-d") + 1] if "-d" in args and args.index("-d") + 1 < len(args) else "."
                write_check(dest, what="unzip")
            elif word == "dd":
                for a in args:
                    if a.startswith("of="):
                        write_check(a[3:], what="dd")
            elif word in ("curl", "wget"):
                for i, a in enumerate(args):
                    opt, _, val = a.partition("=")
                    if word == "curl" and a in ("-o", "--output") or word == "wget" and a in (
                            "-O", "--output-document", "-P", "--directory-prefix"):
                        val = args[i + 1] if i + 1 < len(args) else ""
                    elif not (val and opt in ("--output", "--output-document", "--directory-prefix")):
                        continue
                    if val and val != "-":
                        write_check(val, what=word)
        # cd changes the base for later relative paths
        if word in ("cd", "pushd") and idx >= 0:
            cargs = [a for a in seg[idx + 1:] if not a.startswith("-")]
            if not cargs:
                state["cwd"] = Path(_norm(venv.get("HOME") or policy.workspace))
            else:
                state["cwd"] = resolve_tok(cargs[0])
            venv["PWD"] = str(state["cwd"] or "")
        elif word == "popd":
            state["cwd"] = None

    # --- destructive commands only inside the agent's own workspace
    if destructive:
        wcmds = cmd_policy.workspace_commands
        for seg in _segments(tokens):
            idx, word = _command_word(seg)
            if idx < 0:
                continue
            raw0 = seg[idx].lstrip("(")
            if word in wcmds:
                args = [a.strip("()") for a in seg[idx + 1:]]
                tdir = _target_dir(args) if word in _TARGET_DIR_COMMANDS else None
                args = _args_only(args) + ([tdir] if tdir else [])
                if word in ("chmod", "chown") and args:
                    args = args[1:]
                if not args:
                    continue
                for a in args:
                    write_check(a, own_only=True, what=f"{word}")
            elif word in ("find", "xargs", "parallel") or os.path.basename(raw0) in ("find",):
                if any(os.path.basename(t) in wcmds or t == "-delete" for t in seg):
                    for t in tokens:
                        if t in _SEPARATORS or t.startswith("-") or t in ("{}", ";", "+"):
                            continue
                        if _is_pathlike(t) or t == ".":
                            write_check(t, own_only=True, what=f"{word} with deletion")
                    # find without an explicit path searches the cwd
                    if word == "find" and not any(_is_pathlike(t) for t in seg[idx + 1:]):
                        write_check(".", own_only=True, what="find with deletion")


def _args_only(args: list[str]) -> list[str]:
    out: list[str] = []
    end_opts = False
    skip = False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in _REDIRECTS_WRITE or a in _REDIRECTS_READ:
            skip = True
            continue
        if a.isdigit() and i + 1 < len(args) and args[i + 1] in _REDIRECTS_WRITE + _REDIRECTS_READ:
            continue
        if not end_opts and a == "--":
            end_opts = True
            continue
        if not end_opts and a.startswith("-") and a != "-":
            continue
        out.append(a)
    return out


__all__ = [
    "PathPolicy", "CommandPolicy", "CommandDenied", "check_command", "split_heredocs", "strip_heredocs", "tokenize",
    "default_blocked_read", "default_system_read_roots", "DEFAULT_GROUPS", "HEREDOC_PKG_PATTERNS",
    "NETWORK_PATTERNS", "RUN_PREFIXES", "PROTECTED_FILES", "PROTECTED_DIRS", "SCRATCH_DIRS", "RUN_ROOT_WORK",
]
