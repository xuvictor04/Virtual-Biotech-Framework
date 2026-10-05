"""Local serving profiles (``configs/local_models.yaml``): load, validate, render.

A *serving profile* says how to start one inference server for the harness on
given hardware: checkpoint, engine version, docker image, the exact
``vllm serve`` arguments and what they buy (context, KV dtype, sequences, MTP).
This module is pure (no network, no GPU): it

* loads and validates the catalogue (:func:`load_local_profiles`,
  :func:`validate_local_models`) - the summary fields of every profile and every
  variant must agree with its arguments;
* resolves a profile plus variants / overrides into a :class:`ServeSpec`
  (:func:`resolve_serve`) and renders it as a bare-metal ``vllm serve`` command
  line or a ``docker run`` line (:func:`docker_run_argv`), choosing the image tag
  from the NVIDIA driver version (:func:`select_docker_tag`);
* parses ``nvidia-smi`` output (CSV query, the default table, or ``-L``) and
  picks a profile for the GPUs it lists (:func:`parse_nvidia_smi`,
  :func:`pick_profile`).

Only the standard library and PyYAML are imported, so
``deploy/local/serve_vllm.sh`` can render commands on a GPU box that has vLLM
but not the harness's dependencies.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..config import CONFIG_DIR, deep_merge

LOCAL_MODELS_FILE = CONFIG_DIR / "local_models.yaml"
SCHEMA_VERSION = 1

#: Model families the OpenAI-compatible adapter knows (vbt.providers.families).
KNOWN_FAMILIES = ("qwen3_8", "qwen3_6", "qwen3", "deepseek_v4", "generic")
VERIFICATION_STATUSES = ("recipe-verified", "supported, smoke-test")

#: Arguments every profile must pass: the harness relies on them.
REQUIRED_FLAGS = (
    "--served-model-name",            # the name the harness requests
    "--max-model-len",                # context window (read back from /v1/models)
    "--enable-prefix-caching",        # append-only agent histories
    "--enable-prompt-tokens-details",  # usage.prompt_tokens_details.cached_tokens
    "--enable-auto-tool-choice",      # tool calls parsed server-side
    "--tool-call-parser",
    "--reasoning-parser",             # reasoning returned separately (and thinking_token_budget)
)
#: Arguments the renderer owns (or that would contradict the positional model).
RENDERER_FLAGS = ("--host", "--port", "--model")

_PROFILE_KEYS = {
    "title", "digest_profile", "hardware", "hf_id", "tokenizer", "alternatives", "served_model_name", "family",
    "engine", "docker", "vllm_args", "context_tokens", "kv_cache_dtype", "max_num_seqs", "data_parallel_size",
    "mtp", "env", "capacity", "variants", "harness", "vision", "verification", "verification_detail", "caveats",
    "flag_min_versions", "host", "port", "container_port",
}
_REQUIRED_PROFILE_KEYS = ("title", "hardware", "hf_id", "served_model_name", "family", "vllm_args",
                          "context_tokens", "kv_cache_dtype", "max_num_seqs", "mtp", "harness", "verification",
                          "caveats")
#: Keys a variant may override besides remove / set.
_VARIANT_OVERRIDES = ("hf_id", "tokenizer", "served_model_name", "family", "env", "context_tokens",
                      "kv_cache_dtype", "max_num_seqs", "data_parallel_size", "mtp")
_VARIANT_KEYS = {"description", "remove", "set", *_VARIANT_OVERRIDES}

Arg = tuple  # (flag: str, value: Any | None) - None means a bare flag


class LocalProfileError(ValueError):
    """The serving-profile catalogue is invalid, or a profile/variant/option is unknown."""


# ---------------------------------------------------------------- small helpers

def version_tuple(value: Any) -> tuple[int, ...]:
    """``'v0.31.0'`` -> (0, 31, 0); ``'580.65.06'`` -> (580, 65, 6); ``'0.31.0rc1'`` -> (0, 31, 0).
    Unparseable values give ``()`` (compares lower than every version)."""
    if value is None:
        return ()
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(value))
    return tuple(int(p) for p in m.group(1).split(".")) if m else ()


def version_at_least(value: Any, minimum: Any) -> bool:
    return version_tuple(value) >= version_tuple(minimum)


def format_value(value: Any) -> str:
    """One argument value as vLLM expects it: mappings/lists as compact JSON."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def normalize_args(items: Iterable[Any] | None) -> list[Arg]:
    """``["--flag", {"--key": value}, ...]`` -> ``[("--flag", None), ("--key", value), ...]``."""
    out: list[Arg] = []
    for item in items or []:
        if isinstance(item, str):
            flag, value = item, None
        elif isinstance(item, Mapping) and len(item) == 1:
            flag, value = next(iter(item.items()))
        else:
            raise LocalProfileError(f"invalid vllm_args item {item!r}: use '--flag' or {{'--flag': value}}")
        if not isinstance(flag, str) or not flag.startswith("--"):
            raise LocalProfileError(f"invalid vllm_args flag {flag!r}: flags start with '--'")
        out.append((flag, copy.deepcopy(value)))
    return out


def args_to_argv(args: Sequence[Arg]) -> list[str]:
    argv: list[str] = []
    for flag, value in args:
        argv.append(flag)
        if value is not None:
            argv.append(format_value(value))
    return argv


def has_flag(args: Sequence[Arg], flag: str) -> bool:
    return any(f == flag for f, _ in args)


def arg_value(args: Sequence[Arg], flag: str, default: Any = None) -> Any:
    for f, v in args:
        if f == flag:
            return v
    return default


def set_arg(args: Sequence[Arg], flag: str, value: Any) -> list[Arg]:
    """Replace ``flag``'s value in place, or append it (``value=None``: bare flag)."""
    out = list(args)
    for i, (f, _) in enumerate(out):
        if f == flag:
            out[i] = (flag, copy.deepcopy(value))
            return out
    out.append((flag, copy.deepcopy(value)))
    return out


def remove_arg(args: Sequence[Arg], flag: str) -> list[Arg]:
    return [(f, v) for f, v in args if f != flag]


def shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in argv)


def shell_command(argv: Sequence[str], env: Mapping[str, str] | None = None, *, multiline: bool = True,
                  head: int = 3) -> str:
    """``argv`` as a copy-pasteable shell command: the first ``head`` words on the
    first line, then one ``--flag [value]`` per continuation line (a word that is
    neither a flag nor a flag's first value, e.g. a docker image, starts its own
    line and collects the following non-flag words)."""
    words = [f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items()]
    first = words + [shlex.quote(str(a)) for a in argv[:head]]
    if not multiline:
        return " ".join(first + [shlex.quote(str(a)) for a in argv[head:]])
    lines: list[list[str]] = [first]
    flag_line = False   # the current continuation line starts with a flag ...
    has_value = False   # ... that already has its value
    for a in argv[head:]:
        q = shlex.quote(str(a))
        if str(a).startswith("-"):
            lines.append([q])
            flag_line, has_value = True, False
        elif len(lines) > 1 and (not flag_line or not has_value):
            lines[-1].append(q)
            has_value = True
        else:
            lines.append([q])
            flag_line, has_value = False, True
    return " \\\n  ".join(" ".join(ln) for ln in lines)


# ---------------------------------------------------------------- loading and validation

def load_local_models(path: str | Path | None = None) -> dict[str, Any]:
    """The raw catalogue (validated). Raises :class:`LocalProfileError`."""
    p = Path(path) if path else LOCAL_MODELS_FILE
    if not p.is_file():
        raise LocalProfileError(f"serving-profile catalogue not found: {p}")
    try:
        doc = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise LocalProfileError(f"{p}: invalid YAML: {exc}") from None
    problems = validate_local_models(doc)
    if problems:
        raise LocalProfileError(f"{p}: invalid serving profiles:\n  - " + "\n  - ".join(problems))
    return doc


def _merged(name: str, raw: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    base = {
        "engine": copy.deepcopy(defaults.get("engine") or {}),
        "docker": copy.deepcopy(defaults.get("docker") or {}),
        "env": copy.deepcopy(defaults.get("env") or {}),
        "flag_min_versions": copy.deepcopy(defaults.get("flag_min_versions") or {}),
        "host": defaults.get("host", "127.0.0.1"),
        "port": defaults.get("port", 8000),
        "container_port": defaults.get("container_port", 8000),
        "vision": defaults.get("vision", False),
        "tokenizer": None,
        "alternatives": [],
        "variants": {},
        "data_parallel_size": 1,
        "capacity": {},
    }
    prof = deep_merge(base, dict(raw))
    prof["name"] = name
    return prof


def load_local_profiles(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """``{name: profile}`` with the catalogue defaults merged into every profile."""
    doc = load_local_models(path)
    defaults = doc.get("defaults") or {}
    return {str(n): _merged(str(n), p, defaults) for n, p in (doc.get("profiles") or {}).items()}


def _check_summary(where: str, prof: Mapping[str, Any], args: Sequence[Arg]) -> list[str]:
    """The summary fields of a (resolved) profile agree with its arguments."""
    problems: list[str] = []

    def agrees(got: Any, want: Any) -> bool:
        if isinstance(got, Mapping) and isinstance(want, Mapping):
            # summary mappings (mtp) name the essentials; the argument may carry more keys
            return all(got.get(k) == v for k, v in want.items())
        if isinstance(got, (dict, list)) or isinstance(want, (dict, list)):
            return got == want
        return str(got) == str(want)

    def same(field_name: str, flag: str, *, absent_value: Any = None) -> None:
        want = prof.get(field_name)
        if has_flag(args, flag):
            got = arg_value(args, flag)
            if not agrees(got, want):
                problems.append(f"{where}: {field_name}={want!r} but {flag} {format_value(got)}")
        elif want != absent_value:
            problems.append(f"{where}: {field_name}={want!r} but {flag} is not passed")

    same("served_model_name", "--served-model-name")
    same("context_tokens", "--max-model-len")
    same("kv_cache_dtype", "--kv-cache-dtype", absent_value="auto")
    same("max_num_seqs", "--max-num-seqs", absent_value=None)
    same("mtp", "--speculative-config", absent_value=None)
    same("data_parallel_size", "--data-parallel-size", absent_value=1)
    same("tokenizer", "--tokenizer", absent_value=None)
    for flag in REQUIRED_FLAGS:
        if not has_flag(args, flag):
            problems.append(f"{where}: missing required argument {flag}")
    for flag in RENDERER_FLAGS:
        if has_flag(args, flag):
            problems.append(f"{where}: {flag} must not be in vllm_args (the renderer adds --host/--port; the "
                            "model is positional)")
    fam = str(prof.get("family") or "")
    if fam.startswith("qwen") and not prof.get("vision") and not has_flag(args, "--language-model-only"):
        problems.append(f"{where}: vision is false but --language-model-only is not passed")
    if fam not in KNOWN_FAMILIES:
        problems.append(f"{where}: unknown family {fam!r} (known: {', '.join(KNOWN_FAMILIES)})")
    if not isinstance(prof.get("context_tokens"), int) or prof["context_tokens"] <= 0:
        problems.append(f"{where}: context_tokens must be a positive integer")
    return problems


def validate_local_models(doc: Any) -> list[str]:
    """Problems in a catalogue (an empty list means valid)."""
    problems: list[str] = []
    if not isinstance(doc, Mapping):
        return ["the catalogue must be a mapping"]
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    defaults = doc.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        return problems + ["defaults must be a mapping"]
    profiles = doc.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        return problems + ["profiles must be a non-empty mapping"]
    for name, raw in profiles.items():
        where = f"profiles.{name}"
        if not isinstance(raw, Mapping):
            problems.append(f"{where}: must be a mapping")
            continue
        unknown = set(raw) - _PROFILE_KEYS
        if unknown:
            problems.append(f"{where}: unknown keys {sorted(unknown)}")
        missing = [k for k in _REQUIRED_PROFILE_KEYS if k not in raw]
        if missing:
            problems.append(f"{where}: missing keys {missing}")
            continue
        prof = _merged(str(name), raw, defaults)
        try:
            args = normalize_args(prof["vllm_args"])
        except LocalProfileError as exc:
            problems.append(f"{where}: {exc}")
            continue
        flags = [f for f, _ in args]
        dups = sorted({f for f in flags if flags.count(f) > 1})
        if dups:
            problems.append(f"{where}: duplicate arguments {dups}")
        problems += _check_summary(where, prof, args)
        problems += _check_hardware(where, prof.get("hardware"))
        problems += _check_docker(where, prof)
        if prof.get("verification") not in VERIFICATION_STATUSES:
            problems.append(f"{where}: verification must be one of {VERIFICATION_STATUSES}")
        for i, alt in enumerate(prof.get("alternatives") or []):
            if not isinstance(alt, Mapping) or not alt.get("hf_id"):
                problems.append(f"{where}.alternatives[{i}]: needs hf_id")
            elif alt.get("verification") not in VERIFICATION_STATUSES:
                problems.append(f"{where}.alternatives[{i}]: verification must be one of {VERIFICATION_STATUSES}")
        if not isinstance(prof.get("caveats"), list):
            problems.append(f"{where}: caveats must be a list")
        h = prof.get("harness")
        if not isinstance(h, Mapping) or not h.get("profile"):
            problems.append(f"{where}: harness.profile is required")
        if not isinstance(prof.get("env"), Mapping):
            problems.append(f"{where}: env must be a mapping")
        for flag, ver in (prof.get("flag_min_versions") or {}).items():
            if not version_tuple(ver):
                problems.append(f"{where}: flag_min_versions[{flag}] is not a version: {ver!r}")
        variants = prof.get("variants") or {}
        if not isinstance(variants, Mapping):
            problems.append(f"{where}: variants must be a mapping")
            continue
        for vname, var in variants.items():
            vwhere = f"{where}.variants.{vname}"
            if not isinstance(var, Mapping):
                problems.append(f"{vwhere}: must be a mapping")
                continue
            unknown = set(var) - _VARIANT_KEYS
            if unknown:
                problems.append(f"{vwhere}: unknown keys {sorted(unknown)}")
                continue
            for flag in var.get("remove") or []:
                if not has_flag(args, flag):
                    problems.append(f"{vwhere}: removes {flag}, which the profile does not pass")
            try:
                vprof = apply_variant(prof, str(vname))
            except LocalProfileError as exc:
                problems.append(f"{vwhere}: {exc}")
                continue
            problems += _check_summary(vwhere, vprof, normalize_args(vprof["vllm_args"]))
    return problems


def _check_hardware(where: str, hw: Any) -> list[str]:
    if not isinstance(hw, Mapping):
        return [f"{where}: hardware must be a mapping"]
    problems: list[str] = []
    if not isinstance(hw.get("gpu_count"), int) or hw["gpu_count"] < 1:
        problems.append(f"{where}: hardware.gpu_count must be a positive integer")
    if not isinstance(hw.get("vram_gib_min"), (int, float)):
        problems.append(f"{where}: hardware.vram_gib_min must be a number")
    pats = hw.get("name_patterns")
    if not isinstance(pats, list) or not pats:
        problems.append(f"{where}: hardware.name_patterns must be a non-empty list")
    else:
        for pat in pats:
            try:
                re.compile(str(pat))
            except re.error as exc:
                problems.append(f"{where}: bad hardware.name_patterns entry {pat!r}: {exc}")
    if not version_tuple(hw.get("driver_min")):
        problems.append(f"{where}: hardware.driver_min must be a driver version")
    return problems


def _check_docker(where: str, prof: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    d = prof.get("docker") or {}
    if not d.get("image"):
        problems.append(f"{where}: docker.image is required")
    for key in ("tags", "fallback_tags"):
        tags = d.get(key)
        if tags is None and key == "fallback_tags":
            continue
        if not isinstance(tags, list) or not tags:
            problems.append(f"{where}: docker.{key} must be a non-empty list")
            continue
        for t in tags:
            if not isinstance(t, Mapping) or not t.get("tag") or not version_tuple(t.get("min_driver")):
                problems.append(f"{where}: docker.{key} entries need tag and min_driver: {t!r}")
    eng = prof.get("engine") or {}
    if not version_tuple(eng.get("min_version")):
        problems.append(f"{where}: engine.min_version is required")
    return problems


# ---------------------------------------------------------------- variants and resolution

def apply_variant(profile: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """A copy of ``profile`` with one variant applied (arguments and summary fields)."""
    variants = profile.get("variants") or {}
    if variant not in variants:
        avail = ", ".join(sorted(variants)) or "none"
        raise LocalProfileError(f"profile {profile.get('name')!r} has no variant {variant!r} (available: {avail})")
    var = variants[variant] or {}
    prof = copy.deepcopy(dict(profile))
    args = normalize_args(prof["vllm_args"])
    for flag in var.get("remove") or []:
        args = remove_arg(args, flag)
    for flag, value in (var.get("set") or {}).items():
        if not str(flag).startswith("--"):
            raise LocalProfileError(f"variant {variant!r}: invalid flag {flag!r}")
        args = set_arg(args, str(flag), value)
    for key in _VARIANT_OVERRIDES:
        if key in var:
            if key == "env":
                prof["env"] = {**(prof.get("env") or {}), **(var.get("env") or {})}
            else:
                prof[key] = copy.deepcopy(var[key])
    prof["vllm_args"] = [f if v is None else {f: v} for f, v in args]
    prof.setdefault("applied_variants", []).append(variant)
    return prof


@dataclass
class ServeSpec:
    """A resolved serving profile: everything needed to start the server."""

    profile: str
    variants: list[str]
    hf_id: str
    served_model_name: str
    family: str
    args: list[Arg]
    env: dict[str, str]
    engine_version: str
    host: str
    port: int
    container_port: int
    context_tokens: int
    kv_cache_dtype: str | None
    max_num_seqs: int | None
    data_parallel_size: int
    mtp: dict[str, Any] | None
    docker: dict[str, Any]
    engine: dict[str, Any]
    harness: dict[str, Any]
    verification: str
    notes: list[str] = field(default_factory=list)
    extra_argv: list[str] = field(default_factory=list)

    @property
    def server_argv(self) -> list[str]:
        """Everything after ``vllm serve``: the model and the arguments (no --host/--port)."""
        return [self.hf_id, *args_to_argv(self.args), *self.extra_argv]

    def vllm_argv(self, *, host: str | None = None, port: int | None = None) -> list[str]:
        """The bare-metal command: ``vllm serve <hf_id> ... --host H --port P``."""
        return ["vllm", "serve", *self.server_argv, "--host", str(host or self.host), "--port",
                str(port or self.port)]

    @property
    def base_url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::", "127.0.0.1") else self.host
        return f"http://{host}:{self.port}/v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile, "variants": list(self.variants), "hf_id": self.hf_id,
            "served_model_name": self.served_model_name, "family": self.family,
            "engine_version": self.engine_version, "host": self.host, "port": self.port,
            "context_tokens": self.context_tokens, "kv_cache_dtype": self.kv_cache_dtype,
            "max_num_seqs": self.max_num_seqs, "data_parallel_size": self.data_parallel_size, "mtp": self.mtp,
            "env": dict(self.env), "argv": self.vllm_argv(), "harness": dict(self.harness),
            "verification": self.verification, "notes": list(self.notes),
        }


def resolve_serve(profiles: Mapping[str, Mapping[str, Any]], name: str, *, variants: Iterable[str] = (),
                  bulk: bool = False, hf_id: str | None = None, engine_version: str | None = None,
                  data_parallel: int | None = None, host: str | None = None, port: int | None = None,
                  extra_args: Iterable[str] = ()) -> ServeSpec:
    """Resolve profile ``name`` with ``variants`` (``bulk`` adds ``bulk``) and overrides.

    * ``hf_id`` swaps the checkpoint (e.g. a listed alternative);
    * ``engine_version`` below a flag's minimum (``flag_min_versions``) drops that
      flag, e.g. ``--tool-strict-level`` for vLLM 0.30.0;
    * ``data_parallel`` sets ``--data-parallel-size`` (1 removes it);
    * ``extra_args`` are appended verbatim.
    """
    if name not in profiles:
        raise LocalProfileError(f"unknown serving profile {name!r}; available: {', '.join(profiles)}")
    prof: dict[str, Any] = copy.deepcopy(dict(profiles[name]))
    wanted = list(dict.fromkeys([*variants, *(["bulk"] if bulk else [])]))
    for v in wanted:
        prof = apply_variant(prof, v)
    notes: list[str] = []
    args = normalize_args(prof["vllm_args"])
    if hf_id and hf_id != prof["hf_id"]:
        listed = {a.get("hf_id"): a for a in prof.get("alternatives") or []}
        if hf_id in listed:
            alt = listed[hf_id]
            notes.append(f"checkpoint {hf_id} ({alt.get('verification')}): {alt.get('note', '').strip()}")
        else:
            notes.append(f"checkpoint {hf_id} is not listed for profile {name!r}; smoke-test it with "
                         "`vbt local check`")
        prof["hf_id"] = hf_id
    if data_parallel is not None:
        n = int(data_parallel)
        if n < 1:
            raise LocalProfileError("--data-parallel must be >= 1")
        if n == 1:
            args = remove_arg(args, "--data-parallel-size")
        else:
            args = set_arg(args, "--data-parallel-size", n)
        prof["data_parallel_size"] = n
    eng = dict(prof.get("engine") or {})
    version = str(engine_version or eng.get("min_version") or "")
    for flag, minimum in (prof.get("flag_min_versions") or {}).items():
        if has_flag(args, flag) and version_tuple(version) and not version_at_least(version, minimum):
            args = remove_arg(args, flag)
            notes.append(f"vLLM {version} < {minimum}: dropped {flag}")
    if version_tuple(version) and not version_at_least(version, eng.get("min_version")):
        fb = eng.get("fallback_version")
        if fb and version_at_least(version, fb):
            notes.append(f"vLLM {version} is the fallback engine (recommended: {eng.get('min_version')})")
        else:
            notes.append(f"vLLM {version} is older than the tested versions ({eng.get('min_version')}, fallback "
                         f"{fb}); expect differences")
    if prof.get("verification") != "recipe-verified":
        notes.append(f"verification: {prof.get('verification')} - run `vbt local check` against the server")
    return ServeSpec(
        profile=name, variants=wanted, hf_id=str(prof["hf_id"]), served_model_name=str(prof["served_model_name"]),
        family=str(prof["family"]), args=args, env={str(k): str(v) for k, v in (prof.get("env") or {}).items()},
        engine_version=version, host=str(host or prof.get("host") or "127.0.0.1"),
        port=int(port or prof.get("port") or 8000), container_port=int(prof.get("container_port") or 8000),
        context_tokens=int(prof["context_tokens"]), kv_cache_dtype=prof.get("kv_cache_dtype"),
        max_num_seqs=prof.get("max_num_seqs"), data_parallel_size=int(prof.get("data_parallel_size") or 1),
        mtp=copy.deepcopy(prof.get("mtp")), docker=copy.deepcopy(prof.get("docker") or {}), engine=eng,
        harness=dict(prof.get("harness") or {}), verification=str(prof.get("verification")), notes=notes,
        extra_argv=[str(a) for a in extra_args or []],
    )


# ---------------------------------------------------------------- docker

def _tags_for(docker: Mapping[str, Any], engine: Mapping[str, Any], version: str | None) -> list[dict[str, Any]]:
    tags = [dict(t) for t in docker.get("tags") or []]
    minimum = str(engine.get("min_version") or "")
    if not version or not version_tuple(version) or version_tuple(version) == version_tuple(minimum):
        return tags
    fallback = [dict(t) for t in docker.get("fallback_tags") or []]
    if fallback and all(version_tuple(t["tag"]) == version_tuple(version) for t in fallback):
        return fallback
    # another version: same tag scheme with the version substituted
    return [{**t, "tag": str(t["tag"]).replace(f"v{minimum}", f"v{version}")} for t in tags]


def select_docker_tag(docker: Mapping[str, Any], driver: str | None, *, engine: Mapping[str, Any] | None = None,
                      version: str | None = None) -> tuple[str, str]:
    """``(tag, note)`` for the NVIDIA ``driver`` version: the first tag whose
    ``min_driver`` the driver meets (tags are tried newest-driver first).
    Unknown driver: the first tag, with a note. Too old: :class:`LocalProfileError`."""
    tags = _tags_for(docker, engine or {}, version)
    if not tags:
        raise LocalProfileError("no docker tags configured")
    tags.sort(key=lambda t: version_tuple(t["min_driver"]), reverse=True)
    image = docker.get("image", "vllm/vllm-openai")
    if not driver or not version_tuple(driver):
        alt = ", ".join(f"{t['tag']} (driver >= {t['min_driver']})" for t in tags)
        return tags[0]["tag"], (f"NVIDIA driver unknown; using {image}:{tags[0]['tag']}. Choose by driver: {alt} "
                                "(pass --driver X.Y)")
    for t in tags:
        if version_at_least(driver, t["min_driver"]):
            cuda = f" (CUDA {t['cuda']})" if t.get("cuda") else ""
            return t["tag"], f"NVIDIA driver {driver}: {image}:{t['tag']}{cuda}"
    oldest = tags[-1]
    raise LocalProfileError(f"NVIDIA driver {driver} is too old for {image}: {oldest['tag']} needs >= "
                            f"{oldest['min_driver']}; upgrade the driver")


def docker_run_argv(spec: ServeSpec, *, driver: str | None = None, hf_cache: str | None = None,
                    name: str = "vbt-vllm", bind: str = "127.0.0.1", gpus: str = "all",
                    detach: bool = False) -> tuple[list[str], list[str]]:
    """``(argv, notes)`` of a ``docker run`` line serving ``spec``.

    The container listens on 0.0.0.0:<container_port>; the host publishes it on
    ``bind:port`` (loopback by default: vLLM has no authentication unless
    ``VLLM_API_KEY`` is set). ``hf_cache`` (default ``$HF_HOME`` or
    ``~/.cache/huggingface``) is mounted as the container's Hugging Face cache.
    """
    tag, note = select_docker_tag(spec.docker, driver, engine=spec.engine, version=spec.engine_version)
    image = f"{spec.docker.get('image', 'vllm/vllm-openai')}:{tag}"
    cache = os.path.expanduser(hf_cache or os.environ.get("HF_HOME") or "~/.cache/huggingface")
    argv = ["docker", "run", "--rm"]
    if detach:
        argv.append("-d")
    argv += ["--name", name, "--gpus", gpus, "--ipc=host", "-p", f"{bind}:{spec.port}:{spec.container_port}",
             "-v", f"{cache}:/root/.cache/huggingface"]
    for k, v in spec.env.items():
        argv += ["-e", f"{k}={v}"]
    argv += ["-e", "HF_TOKEN", "-e", "VLLM_API_KEY"]   # passed through only when set on the host
    argv += [image, *spec.server_argv, "--host", "0.0.0.0", "--port", str(spec.container_port)]
    return argv, [note]


# ---------------------------------------------------------------- nvidia-smi

@dataclass
class GPU:
    name: str
    memory_mib: int | None = None
    index: int | None = None

    @property
    def vram_gib(self) -> float | None:
        if self.memory_mib:
            return self.memory_mib / 1024.0
        m = re.search(r"(\d{2,3})\s*GB\b", self.name, re.I)   # e.g. "H100 80GB HBM3" (nvidia-smi -L)
        return float(m.group(1)) if m else None


@dataclass
class GPUInfo:
    gpus: list[GPU] = field(default_factory=list)
    driver_version: str | None = None
    cuda_version: str | None = None


NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"]


def detect_nvidia_smi(timeout_s: float = 15.0) -> str | None:
    """Output of :data:`NVIDIA_SMI_QUERY`, or None when nvidia-smi is missing or fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        res = subprocess.run([exe, *NVIDIA_SMI_QUERY[1:]], capture_output=True, text=True, timeout=timeout_s,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout if res.returncode == 0 and res.stdout.strip() else None


_DRIVER_RE = re.compile(r"Driver Version:\s*([\d.]+)")
_CUDA_RE = re.compile(r"CUDA Version:\s*([\d.]+)")
_TABLE_ROW_RE = re.compile(r"^\|\s*(\d+)\s+(.+?)\s+(On|Off)\s+\|")
_TABLE_MEM_RE = re.compile(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB")
_LIST_RE = re.compile(r"^GPU\s+(\d+):\s*(.+?)\s*(?:\(UUID:.*\))?\s*$")
_VERSION_FIELD_RE = re.compile(r"^\d{3,}\.\d+(?:\.\d+)*$")
_MEM_FIELD_RE = re.compile(r"^(\d+)\s*(?:MiB)?$", re.I)


def parse_nvidia_smi(text: str) -> GPUInfo:
    """GPUs, driver and CUDA version from nvidia-smi output: the CSV query
    (``--query-gpu=index,name,memory.total,driver_version --format=csv[,noheader][,nounits]``),
    the default table, or ``nvidia-smi -L``."""
    info = GPUInfo()
    text = text or ""
    m = _DRIVER_RE.search(text)
    if m:
        info.driver_version = m.group(1)
    m = _CUDA_RE.search(text)
    if m:
        info.cuda_version = m.group(1)
    lines = [ln.rstrip() for ln in text.splitlines()]
    if any(_TABLE_ROW_RE.match(ln) for ln in lines):            # default table output
        pending: GPU | None = None
        for ln in lines:
            row = _TABLE_ROW_RE.match(ln)
            if row:
                pending = GPU(name=row.group(2).strip(), index=int(row.group(1)))
                info.gpus.append(pending)
                continue
            mem = _TABLE_MEM_RE.search(ln)
            if mem and pending is not None and pending.memory_mib is None:
                pending.memory_mib = int(mem.group(2))
        return info
    if any(_LIST_RE.match(ln.strip()) for ln in lines):         # nvidia-smi -L
        for ln in lines:
            row = _LIST_RE.match(ln.strip())
            if row:
                info.gpus.append(GPU(name=row.group(2).strip(), index=int(row.group(1))))
        return info
    header: list[str] | None = None                              # CSV query
    for ln in lines:
        if not ln.strip():
            continue
        fields = [f.strip() for f in ln.split(",")]
        low = [f.lower() for f in fields]
        if "name" in low and any(f.startswith("memory.total") for f in low):
            header = [f.split(" ")[0] for f in low]
            continue
        gpu = GPU(name="")
        if header and len(header) == len(fields):
            rec = dict(zip(header, fields))
            gpu.name = rec.get("name", "")
            mm = _MEM_FIELD_RE.match(rec.get("memory.total", ""))
            gpu.memory_mib = int(mm.group(1)) if mm else None
            if rec.get("index", "").isdigit():
                gpu.index = int(rec["index"])
            if rec.get("driver_version") and not info.driver_version:
                info.driver_version = rec["driver_version"]
        else:
            for i, f in enumerate(fields):
                if i == 0 and f.isdigit() and len(f) <= 2 and len(fields) >= 3:
                    gpu.index = int(f)
                elif _VERSION_FIELD_RE.match(f):
                    info.driver_version = info.driver_version or f
                elif _MEM_FIELD_RE.match(f) and int(_MEM_FIELD_RE.match(f).group(1)) >= 1024:
                    gpu.memory_mib = int(_MEM_FIELD_RE.match(f).group(1))
                elif re.search(r"[A-Za-z]", f) and not gpu.name:
                    gpu.name = f
        if gpu.name:
            info.gpus.append(gpu)
    return info


@dataclass
class ProfilePick:
    """The serving profile recommended for the detected GPUs (``profile`` None: no fit)."""

    profile: str | None
    reason: str
    variants: list[str] = field(default_factory=list)
    data_parallel_size: int = 1
    gpu_name: str | None = None
    gpu_count: int = 0
    vram_gib: float | None = None
    driver_version: str | None = None
    docker_tag: str | None = None
    alternatives: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("profile", "variants", "data_parallel_size", "gpu_name", "gpu_count",
                                              "vram_gib", "driver_version", "docker_tag", "reason", "alternatives",
                                              "warnings")}

    def serve_args(self) -> list[str]:
        """The ``vbt local serve`` options reproducing this pick."""
        if not self.profile:
            return []
        out = ["--profile", self.profile]
        for v in self.variants:
            out += ["--variant", v]
        if self.data_parallel_size > 1:
            out += ["--data-parallel", str(self.data_parallel_size)]
        return out


def _gpu_class(gpu: GPU, singles: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, list[str]]:
    """``(profile, matched-by-name)`` for one GPU: name patterns, then the best VRAM fit."""
    named = [n for n, p in singles.items()
             if any(re.search(str(pat), gpu.name, re.I) for pat in p["hardware"].get("name_patterns") or [])]
    vram = gpu.vram_gib
    fits = [n for n in named if vram is None or vram + 0.5 >= float(singles[n]["hardware"]["vram_gib_min"])]
    if not fits:
        return None, named
    return max(fits, key=lambda n: float(singles[n]["hardware"]["vram_gib_min"])), named


def pick_profile(nvidia_smi: str | GPUInfo, profiles: Mapping[str, Mapping[str, Any]] | None = None) -> ProfilePick:
    """Pick a serving profile for the GPUs in ``nvidia-smi`` output.

    One GPU -> the single-GPU profile whose ``name_patterns`` match and whose
    ``vram_gib_min`` the card meets (the most demanding fit wins, e.g. a 144 GB
    GH200 -> h200). Several GPUs of one class -> ``dp`` (data-parallel replicas;
    a matching variant for H100 / B200 / RTX PRO 6000) with
    ``data_parallel_size`` = GPU count; 4+ H200/B200 also list ``deepseek-v4``.
    Unrecognised GPUs -> ``profile=None`` with a suggestion.
    """
    profiles = profiles if profiles is not None else load_local_profiles()
    info = parse_nvidia_smi(nvidia_smi) if isinstance(nvidia_smi, str) else nvidia_smi
    if not info.gpus:
        return ProfilePick(None, "no NVIDIA GPU found in the nvidia-smi output", driver_version=info.driver_version)
    singles = {n: p for n, p in profiles.items() if int(p["hardware"].get("gpu_count", 1)) == 1}
    classes = [_gpu_class(g, singles) for g in info.gpus]
    first = info.gpus[0]
    pick = ProfilePick(None, "", gpu_name=first.name, gpu_count=len(info.gpus), vram_gib=first.vram_gib,
                       driver_version=info.driver_version)
    if pick.vram_gib is not None:
        pick.vram_gib = round(pick.vram_gib, 1)
    known = [c for c, _ in classes if c]
    if not known:
        named = classes[0][1]
        vram = first.vram_gib
        if named:
            need = min(float(singles[n]["hardware"]["vram_gib_min"]) for n in named)
            pick.reason = (f"{first.name} has {vram:.1f} GiB; the matching profile(s) {', '.join(named)} need >= "
                           f"{need:g} GiB")
        else:
            pick.reason = f"unrecognised GPU {first.name!r}"
        if vram is not None and vram + 0.5 >= 30 and "5090" in profiles:
            pick.alternatives.append("5090")
            pick.warnings.append("Try --profile 5090 (Qwen3.8-27B INT4/W4A16 runs on any NVIDIA GPU with >= 32 GB "
                                 "via Marlin; add --variant ampere on Ampere cards) and run `vbt local check`.")
        elif vram is not None:
            pick.warnings.append("Qwen3.8-27B needs a GPU with >= 32 GB of memory (INT4) or 80+ GB (FP8/NVFP4).")
        return pick
    counts: dict[str, int] = {}
    for c in known:
        counts[c] = counts.get(c, 0) + 1
    cls = max(counts, key=lambda c: (counts[c], -known.index(c)))
    if len(counts) > 1 or len(known) < len(info.gpus):
        pick.warnings.append(f"mixed GPUs ({', '.join(g.name for g in info.gpus)}); sized for the {counts[cls]} "
                             f"x {cls} GPU(s)")
    n = counts[cls]
    pick.gpu_count = n
    pick.gpu_name = next(g.name for g, (c, _) in zip(info.gpus, classes) if c == cls)
    if n == 1 or "dp" not in profiles or cls not in ("h100", "h200", "b200", "rtxpro6000"):
        pick.profile = cls
        pick.data_parallel_size = n
        pick.reason = f"{n} x {pick.gpu_name}: profile {cls}"
        if n > 1:
            pick.reason += f" with --data-parallel {n} (one replica per GPU)"
    else:
        pick.profile = "dp"
        pick.data_parallel_size = n
        dp_variants = (profiles["dp"].get("variants") or {})
        if cls in dp_variants and cls != "h200":
            pick.variants = [cls]
        pick.reason = f"{n} x {pick.gpu_name}: data-parallel replicas (profile dp" + \
            (f", variant {cls}" if pick.variants else "") + f", --data-parallel {n})"
        pick.alternatives.append(f"{cls} (a single replica on one GPU)")
        if cls in ("h200", "b200") and n >= 4 and "deepseek-v4" in profiles:
            pick.alternatives.append("deepseek-v4 (max quality: DeepSeek-V4-Flash, DP4+EP)")
    if cls == "rtxpro6000" and n >= 8 and "deepseek-v4" in profiles:
        pick.alternatives.append("deepseek-v4 --variant rtxpro6000x8")
    prof = profiles[pick.profile]
    try:
        pick.docker_tag, _ = select_docker_tag(prof.get("docker") or {}, info.driver_version,
                                               engine=prof.get("engine") or {})
    except LocalProfileError as exc:
        pick.warnings.append(str(exc))
    if info.driver_version and not version_at_least(info.driver_version, prof["hardware"].get("driver_min")):
        pick.warnings.append(f"driver {info.driver_version} < {prof['hardware'].get('driver_min')} required by vLLM "
                             f"{(prof.get('engine') or {}).get('min_version')}")
    return pick


__all__ = [
    "GPU", "GPUInfo", "LOCAL_MODELS_FILE", "LocalProfileError", "NVIDIA_SMI_QUERY", "ProfilePick", "ServeSpec",
    "apply_variant", "arg_value", "args_to_argv", "detect_nvidia_smi", "docker_run_argv", "format_value", "has_flag",
    "load_local_models", "load_local_profiles", "normalize_args", "parse_nvidia_smi", "pick_profile",
    "remove_arg", "resolve_serve", "select_docker_tag", "set_arg", "shell_command", "shell_join",
    "validate_local_models", "version_at_least", "version_tuple",
]
