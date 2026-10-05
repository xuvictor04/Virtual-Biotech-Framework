"""``vbt local check`` and ``vbt local bench``: probe a running OpenAI-compatible server.

The check answers "will the harness work on this server?" before a long run. It
talks to the server twice over: raw HTTP for the server facts (``/health``,
``/v1/models``, ``/version``) and the harness's own adapter
(:class:`vbt.providers.openai_compat.OpenAICompatProvider`) for every model
probe, so request encoding and response decoding are exercised exactly as in a
session. Checks (each pass / warn / fail / skip with its threshold):

========================  =====================================================================
``health``                ``GET /health`` answers 200
``models``                the served model name is listed; ``max_model_len`` >= the configured
                          context window
``version``               engine version >= 0.31.0 (``--tool-strict-level``); warn otherwise
``prepare``               the adapter's ``prepare()`` (health + model discovery)
``tool_call``             one tool call with the right arguments
``parallel_tool_calls``   three independent lookups -> >= 2 calls in one turn
``strict_schema``         ``submit_result`` with the Case 1 ``TrialAnnotation`` schema (13 anyOf,
                          7 $ref, 2 regex patterns), ``strict: true``; arguments validated with
                          pydantic (a forced ``tool_choice`` retry if the model did not call it)
``forced_tool_choice``    ``tool_choice`` naming a tool the prompt does not ask for
``reasoning_effort``      xhigh and medium return reasoning, thinking off returns none
``thinking_budget``       ``thinking_token_budget`` caps reasoning (no runaway to max_tokens)
``needle_<N>k``           retrieve a code hidden in an N-token document (default 32K; 200K with
                          ``--long-context``)
``prefix_cache``          a repeated long prefix reports ``cached_tokens > 0``
``throughput``            aggregate decode tokens/s at concurrency N
``malformed_calls``       share of K tool-call trials that are not schema-valid calls
========================  =====================================================================

``vbt local bench`` sweeps throughput at concurrency 1 / 8 / 32. Both write a
JSON and a Markdown report (default ``<runs_dir>/local/<check|bench>-<UTC time>/``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..providers.base import (
    Message,
    ModelResponse,
    ModelSettings,
    ProviderError,
    StopReason,
    ThinkingBlock,
    ToolCall,
    ToolSpec,
)

LOCAL_PROVIDERS = ("vllm", "sglang", "openai_compat", "llamacpp")
DEFAULT_BASE_URL = "http://localhost:8000/v1"
PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"

MODEL_CHECKS = ("tool_call", "parallel_tool_calls", "strict_schema", "forced_tool_choice", "reasoning_effort",
                "thinking_budget", "needle", "prefix_cache", "throughput", "malformed_calls")
CHECK_NAMES = ("health", "models", "version", "prepare", *MODEL_CHECKS)

ProviderFactory = Callable[["CheckOptions", "dict[str, Any] | None"], Any]


# ---------------------------------------------------------------- options and results

@dataclass
class CheckOptions:
    """Where to probe and the pass thresholds."""

    base_url: str = DEFAULT_BASE_URL
    model: str | None = None              # served model name (None: the only/first one served)
    family: str = "auto"
    provider_name: str = "vllm"
    api_key: str | None = None
    timeout_s: float = 30.0               # connect / plain HTTP
    read_timeout_s: float = 600.0         # silence between streamed chunks
    check_timeout_s: float = 900.0        # wall-clock cap per check
    only: tuple[str, ...] = ()
    skip: tuple[str, ...] = ()
    min_context_tokens: int | None = None  # models: max_model_len must reach this
    min_engine_version: str = "0.31.0"
    needle_tokens: tuple[int, ...] = (32768,)
    needle_depth: float = 0.5
    prefix_tokens: int = 12000
    thinking_budget: int = 256
    max_thinking_overrun: float = 1.5     # reasoning tokens <= budget * this + 64
    min_parallel_calls: int = 2
    concurrency: int = 8
    output_tokens: int = 256
    min_tokens_per_s: float = 20.0
    ignore_eos: bool = True               # throughput: generate exactly output_tokens
    trials: int = 10
    max_malformed_rate: float = 0.1
    chars_per_token: float = 4.0          # haystack sizing (actual prompt tokens are reported)
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("api_key", None)
        return d


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    threshold: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckReport:
    base_url: str
    model: str | None
    family: str | None
    started_at: str
    finished_at: str = ""
    server: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    @property
    def ok(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    def result(self, name: str) -> CheckResult | None:
        return next((c for c in self.checks if c.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        return {"tool": "vbt local check", "ok": self.ok, "counts": self.counts(), "base_url": self.base_url,
                "model": self.model, "family": self.family, "started_at": self.started_at,
                "finished_at": self.finished_at, "server": self.server, "options": self.options,
                "checks": [c.as_dict() for c in self.checks]}

    def markdown(self) -> str:
        c = self.counts()
        verdict = "PASS" if self.ok else "FAIL"
        srv = self.server
        lines = [f"# Local server check: {self.model or '?'} at {self.base_url}", "",
                 f"- Started: {self.started_at} (finished {self.finished_at})",
                 f"- Server: {srv.get('engine') or 'engine ?'} {srv.get('version') or ''}; "
                 f"max_model_len {srv.get('max_model_len') or '?'}; root {srv.get('root') or '?'}; "
                 f"family {self.family or '?'}",
                 f"- Result: **{verdict}** ({c[PASS]} passed, {c[FAIL]} failed, {c[WARN]} warnings, "
                 f"{c[SKIP]} skipped)", "",
                 "| Check | Status | Detail | Threshold | Time (s) |", "|---|---|---|---|---|"]
        for r in self.checks:
            detail = r.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r.name} | {r.status.upper() if r.status == FAIL else r.status} | {detail} | "
                         f"{r.threshold.replace('|', '/')} | {r.elapsed_s:.1f} |")
        lines += ["", "## Metrics", ""]
        for r in self.checks:
            if r.metrics:
                lines += [f"### {r.name}", "", "```json", json.dumps(r.metrics, indent=1, default=str), "```", ""]
        return "\n".join(lines)


# ---------------------------------------------------------------- provider plumbing

def default_provider_factory(opts: CheckOptions, extra_body: dict[str, Any] | None = None) -> Any:
    """The harness adapter for ``opts`` (``extra_body`` is merged into every request)."""
    try:
        from ..providers.openai_compat import OpenAICompatProvider
    except ImportError as exc:  # pragma: no cover - present once the local provider is installed
        raise ProviderError(f"vbt.providers.openai_compat is not available ({exc})") from None
    kwargs: dict[str, Any] = {"base_url": opts.base_url, "model": opts.model, "api_key": opts.api_key,
                              "family": opts.family or "auto", "name": opts.provider_name,
                              "timeout_s": opts.timeout_s, "read_timeout_s": opts.read_timeout_s,
                              "served_model_name": opts.model}
    if extra_body:
        kwargs["extra_body"] = dict(extra_body)
    return OpenAICompatProvider(**kwargs)


def family_traits(model: str | None, family: str | None) -> tuple[str, bool]:
    """``(family name, supports thinking_token_budget)`` from vbt.providers.families
    when available; otherwise inferred from the model name."""
    explicit = None if (family or "auto").lower() in ("", "auto") else family
    try:
        from ..providers.families import resolve_family
        fam = resolve_family(model, explicit)
        return str(fam.name), bool(getattr(fam, "supports_thinking_budget", True))
    except Exception:  # noqa: BLE001 - families module absent or unknown family: infer
        name = explicit or ("deepseek_v4" if "deepseek" in (model or "").lower() or "dsv4" in (model or "").lower()
                            else "qwen3_8" if "qwen" in (model or "").lower() else "generic")
        return name, name.startswith("qwen")


def split_base_url(url: str) -> tuple[str, str]:
    """``(api, root)``: ``http://h:8000/v1`` -> (``http://h:8000/v1``, ``http://h:8000``)."""
    u = (url or DEFAULT_BASE_URL).strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
    if u.endswith("/v1"):
        return u, u[:-3]
    return u + "/v1", u


def _is_loopback(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in ("localhost", "::1") or host.startswith("127.")


def _tool_spec(name: str, description: str, schema: dict[str, Any], *, strict: bool = False) -> ToolSpec:
    """A ToolSpec, marked strict when ToolSpec has the ``strict`` field."""
    try:
        return ToolSpec(name, description, schema, strict=strict)  # type: ignore[call-arg]
    except TypeError:  # older ToolSpec without the field
        spec = ToolSpec(name, description, schema)
        if strict:
            object.__setattr__(spec, "strict", True)
        return spec


def _reasoning(resp: ModelResponse) -> str:
    return "".join(b.text for b in resp.message.content if isinstance(b, ThinkingBlock) and b.text)


def _invalid(call: ToolCall) -> str | None:
    native = call.native or {}
    if native.get("invalid_arguments") is not None:
        return str(native.get("error") or "invalid JSON arguments")
    return None


def _prompt_tokens(resp: ModelResponse) -> int:
    u = resp.usage
    return int(u.input_tokens + u.cache_read_tokens + u.cache_write_tokens)


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] if lo == hi else vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


# ---------------------------------------------------------------- probe content

GENE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string", "description": "HGNC gene symbol, e.g. TP53"},
        "organism": {"type": "string", "enum": ["human", "mouse"], "description": "Organism (default human)"},
    },
    "required": ["symbol"],
    "additionalProperties": False,
}
CHECK_SYSTEM = ("You are a biomedical research assistant working inside an automated harness. Use the tools to "
                "look things up instead of answering from memory, and keep answers short.")

TRIAL_PROMPT = """Annotate this clinical trial and record the annotation with the submit_result tool.

Registry record (ClinicalTrials.gov):
- NCT ID: NCT01234567
- Overall status: Terminated
- Why stopped: the sponsor terminated the study after a planned interim analysis showed futility.
- Primary endpoint: progression-free survival; hazard ratio 0.97 (95% CI 0.81-1.16), p = 0.74.
- Secondary endpoint: overall survival; no significant difference (p = 0.52).
- Serious adverse events (experimental arm): 31.2% of participants in total; infections 6.1%,
  gastrointestinal 4.0%, cardiac 1.2%; other organ classes not reported.
- Results are posted on ClinicalTrials.gov; no publication was found.

Use only these facts: results and adverse events come from ClinicalTrials.gov, the only source
you consulted. Your confidence is high."""

GENES = ("TP53", "BRCA1", "EGFR", "KRAS", "PCSK9", "IL6", "TNF", "APOE", "CFTR", "HBB", "MYC", "VEGFA", "ESR1",
         "PTEN", "BRAF", "JAK2", "ALK", "ERBB2", "CD274", "LDLR")


class _Finding(BaseModel):
    claim: str
    evidence_level: str = Field(pattern="^(strong|moderate|weak)$")
    pmids: list[str] = Field(default_factory=list)


class _Findings(BaseModel):
    """Arguments of the malformed-call probe's record_findings tool."""

    gene: str
    findings: list[_Finding] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "gene": {"type": "string", "description": "HGNC symbol"},
        "findings": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "evidence_level": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                    "pmids": {"type": "array", "items": {"type": "string", "pattern": "^[0-9]+$"}},
                },
                "required": ["claim", "evidence_level"],
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["gene", "findings", "confidence"],
}

_SUBJECTS = ("the committee", "a field team", "the archivist", "the survey", "a ferry crew", "the botanist",
             "the old mill", "a weather station", "the night shift", "the harbour office", "a cartographer",
             "the orchard")
_VERBS = ("recorded", "described", "measured", "noted", "catalogued", "reported", "observed", "sketched",
          "summarised", "revisited")
_OBJECTS = ("the tides along the northern shore", "a quiet change in the river level", "the colour of the evening "
            "sky", "three new paths through the forest", "the migration of grey geese", "an unusual frost in "
            "spring", "the price of salt in the market", "the slow growth of the lichen", "a long dispute about "
            "fences", "the timetable of the mountain railway", "the sound of bells across the valley",
            "the number of lanterns on the bridge")
_TAILS = ("before the winter came.", "during a long and uneventful week.", "while the rain kept falling.",
          "and nobody thought it important.", "in a small notebook with a red cover.", "for the third year "
          "in a row.", "as the light faded.", "with more care than usual.")
_CODE_WORDS = ("MAGENTA", "COBALT", "SAFFRON", "UMBER", "VIRIDIAN", "OCHRE", "CERULEAN", "SIENNA")


def make_haystack(n_tokens: int, rng: random.Random, *, needle: str | None = None, depth: float = 0.5,
                  chars_per_token: float = 4.0) -> str:
    """Deterministic filler prose of about ``n_tokens`` tokens; ``needle`` is
    inserted at relative ``depth``."""
    budget = max(200, int(n_tokens * chars_per_token))
    parts: list[str] = []
    size = 0
    while size < budget:
        s = (f"In year {rng.randint(1700, 1999)}, {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} "
             f"{rng.choice(_OBJECTS)} {rng.choice(_TAILS)}")
        parts.append(s)
        size += len(s) + 1
    if needle:
        at = min(len(parts), max(0, int(len(parts) * depth)))
        parts.insert(at, needle)
    paras = [" ".join(parts[i:i + 8]) for i in range(0, len(parts), 8)]
    return "\n\n".join(paras)


def _est_tokens(text: str, chars_per_token: float) -> int:
    return int(math.ceil(len(text or "") / max(1.0, chars_per_token)))


# ---------------------------------------------------------------- the checker

class LocalChecker:
    """Runs the checks of :data:`CHECK_NAMES` against one server."""

    def __init__(self, opts: CheckOptions, provider_factory: ProviderFactory | None = None,
                 *, http_transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.opts = opts
        self.factory = provider_factory or default_provider_factory
        self.api, self.root = split_base_url(opts.base_url)
        self.rng = random.Random(opts.seed)
        self.nonce = "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(10))
        self.server: dict[str, Any] = {}
        self.model: str | None = opts.model
        self.family: str | None = None
        self.supports_budget = True
        self.provider: Any = None
        self._providers: list[Any] = []
        self._transport = http_transport
        self._unreachable: str | None = None
        self._no_model: str | None = None

    # ---- infrastructure

    def _selected(self, name: str) -> bool:
        def hit(pats: Sequence[str]) -> bool:
            return any(name == p or name.startswith(p + "_") or (p == "needle" and name.startswith("needle"))
                       for p in pats)
        if self.opts.only and not hit(self.opts.only):
            return False
        return not hit(self.opts.skip)

    def _http(self) -> httpx.AsyncClient:
        headers = {}
        key = self.opts.api_key or os.environ.get("VBT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        kwargs: dict[str, Any] = {"timeout": httpx.Timeout(self.opts.timeout_s), "headers": headers,
                                  "trust_env": not _is_loopback(self.root)}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    def _new_provider(self, extra_body: dict[str, Any] | None = None) -> Any:
        opts = self.opts
        if self.model != opts.model:
            opts = CheckOptions(**{f.name: getattr(self.opts, f.name) for f in fields(CheckOptions)})
            opts.model = self.model
        p = self.factory(opts, extra_body)
        self._providers.append(p)
        return p

    async def aclose(self) -> None:
        for p in self._providers:
            close = getattr(p, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001 - best effort
                    pass
        self._providers.clear()

    async def _complete(self, user: str, *, tools: Sequence[ToolSpec] = (), effort: str | None = "medium",
                        thinking: bool = True, max_tokens: int = 4096, provider: Any = None,
                        system: str = CHECK_SYSTEM, on_text: Any = None, on_thinking: Any = None,
                        **extra: Any) -> tuple[ModelResponse, float]:
        settings = ModelSettings(provider=self.opts.provider_name, model=self.model or "", max_tokens=max_tokens,
                                 effort=effort, thinking=thinking,
                                 extra={"agent_name": "vbt-local-check", **{k: v for k, v in extra.items()
                                                                           if v is not None}})
        t0 = time.perf_counter()
        resp = await (provider or self.provider).complete(settings=settings, system=system,
                                                          messages=[Message.user(user)], tools=list(tools),
                                                          on_text=on_text, on_thinking=on_thinking)
        return resp, time.perf_counter() - t0

    async def _run(self, name: str, fn: Callable[[], Awaitable[CheckResult]]) -> CheckResult:
        t0 = time.perf_counter()
        if name != "health" and self._unreachable:
            return CheckResult(name, SKIP, self._unreachable)
        if name == "prepare" and self._no_model:
            return CheckResult(name, SKIP, self._no_model)
        if _is_model_check(name):
            if self._no_model:
                return CheckResult(name, SKIP, self._no_model)
            if self.provider is None:
                return CheckResult(name, SKIP, "the adapter could not be created (see prepare)")
        try:
            res = await asyncio.wait_for(fn(), timeout=self.opts.check_timeout_s)
        except asyncio.TimeoutError:
            res = CheckResult(name, FAIL, f"timed out after {self.opts.check_timeout_s:g} s")
        except ProviderError as exc:
            res = CheckResult(name, FAIL, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a probe bug or an unexpected server answer
            res = CheckResult(name, FAIL, f"{type(exc).__name__}: {exc}")
        res.name = name
        res.elapsed_s = round(time.perf_counter() - t0, 3)
        return res

    # ---- server checks (raw HTTP)

    async def check_health(self) -> CheckResult:
        async with self._http() as http:
            try:
                r = await http.get(self.root + "/health")
            except httpx.TransportError as exc:
                self._unreachable = f"server unreachable ({self.root})"
                return CheckResult("health", FAIL, f"cannot connect to {self.root}/health ({type(exc).__name__}: "
                                   f"{exc}); start the server: `vbt local serve` (or docker compose -f "
                                   "deploy/local/docker-compose.yml --profile h100 up -d)", "HTTP 200")
        if r.status_code == 200:
            return CheckResult("health", PASS, "GET /health 200", "HTTP 200")
        if r.status_code in (404, 405):
            return CheckResult("health", WARN, f"GET /health returned {r.status_code} (server without /health)",
                               "HTTP 200")
        return CheckResult("health", FAIL, f"GET /health returned HTTP {r.status_code} (still loading?)", "HTTP 200")

    async def check_models(self) -> CheckResult:
        async with self._http() as http:
            try:
                r = await http.get(self.api + "/models")
            except httpx.TransportError as exc:
                self._unreachable = f"server unreachable ({self.api})"
                return CheckResult("models", FAIL, f"cannot connect to {self.api}/models ({exc})")
        if r.status_code >= 400:
            self._no_model = f"GET /v1/models failed (HTTP {r.status_code})"
            return CheckResult("models", FAIL, f"GET /v1/models returned HTTP {r.status_code}: {r.text[:200]}")
        try:
            payload = r.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            self._no_model = "GET /v1/models did not return a model list"
            return CheckResult("models", FAIL, f"GET /v1/models returned no JSON model list: {r.text[:200]!r} "
                               "(is the base URL an OpenAI-compatible /v1 endpoint?)")
        data = [d for d in (payload.get("data") or []) if isinstance(d, dict) and d.get("id")]
        ids = [str(d["id"]) for d in data]
        self.server["models"] = ids
        threshold = "served model listed" + (f"; max_model_len >= {self.opts.min_context_tokens}"
                                             if self.opts.min_context_tokens else "")
        if not ids:
            self._no_model = "the server lists no models"
            return CheckResult("models", FAIL, "GET /v1/models lists no models", threshold)
        notes: list[str] = []
        if not self.model:
            self.model = ids[0]
            notes.append(f"no model configured; using {self.model!r}" + (f" (served: {ids})" if len(ids) > 1 else ""))
        entry = next((d for d in data if str(d["id"]) == self.model), None)
        if entry is None:
            self._no_model = f"model {self.model!r} is not served"
            return CheckResult("models", FAIL, f"model {self.model!r} is not served; served: {ids}. Start vLLM with "
                               f"--served-model-name {self.model} or set provider.options.served_model_name",
                               threshold, {"served": ids})
        mlen = entry.get("max_model_len") or entry.get("context_length") or entry.get("max_context_length")
        self.server.update({"max_model_len": int(mlen) if mlen else None, "root": entry.get("root"),
                            "owned_by": entry.get("owned_by")})
        metrics = {"served": ids, "model": self.model, "max_model_len": self.server["max_model_len"],
                   "root": entry.get("root")}
        if not mlen:
            return CheckResult("models", WARN, "; ".join(notes + ["max_model_len not reported (not vLLM?)"]),
                               threshold, metrics)
        if self.opts.min_context_tokens and int(mlen) < int(self.opts.min_context_tokens):
            return CheckResult("models", FAIL, f"max_model_len {mlen} < configured context window "
                               f"{self.opts.min_context_tokens}: lower models.<tier>.context_window_tokens or "
                               "raise --max-model-len", threshold, metrics)
        detail = f"{self.model} served (root {entry.get('root')}), max_model_len {mlen}"
        return CheckResult("models", PASS, "; ".join([detail, *notes]), threshold, metrics)

    async def check_version(self) -> CheckResult:
        from .profiles import version_at_least  # local import: profiles imports the config module
        threshold = f">= {self.opts.min_engine_version}"
        async with self._http() as http:
            try:
                r = await http.get(self.root + "/version")
            except httpx.TransportError as exc:
                return CheckResult("version", WARN, f"GET /version failed ({exc})", threshold)
        if r.status_code >= 400:
            return CheckResult("version", WARN, f"GET /version returned HTTP {r.status_code} (not vLLM?): "
                               "--tool-strict-level and the vLLM-specific checks may not apply", threshold)
        try:
            payload = r.json()
        except ValueError:
            payload = r.text
        version = payload.get("version") if isinstance(payload, dict) else str(payload).strip()
        self.server.update({"engine": "vLLM", "version": version})
        if version and version_at_least(version, self.opts.min_engine_version):
            return CheckResult("version", PASS, f"vLLM {version}", threshold, {"version": version})
        return CheckResult("version", WARN, f"vLLM {version or '?'} < {self.opts.min_engine_version}: "
                           "--tool-strict-level is unavailable (per-tool strict and forced tool_choice still work "
                           "on 0.30) and hybrid prefix-cache fixes are missing", threshold, {"version": version})

    async def check_prepare(self) -> CheckResult:
        self.family, self.supports_budget = family_traits(self.model, self.opts.family)
        self.provider = self._new_provider()
        prepare = getattr(self.provider, "prepare", None)
        if prepare is None:
            return CheckResult("prepare", SKIP, "the provider has no prepare()")
        await prepare()
        info: dict[str, Any] = {}
        server_info = getattr(self.provider, "server_info", None)
        if server_info is not None:
            try:
                info = await server_info() or {}
            except Exception as exc:  # noqa: BLE001 - informational
                info = {"error": f"{type(exc).__name__}: {exc}"}
        window = None
        cw = getattr(self.provider, "context_window", None)
        if cw is not None and self.model:
            try:
                window = cw(self.model)
            except Exception:  # noqa: BLE001
                window = None
        metrics = {"family": self.family, "context_window": window,
                   "server_info": {k: v for k, v in info.items() if k in ("version", "max_model_len", "family",
                                                                         "served_model_name", "error")}}
        return CheckResult("prepare", PASS, f"adapter ready (family {self.family}, context window {window})",
                           "prepare() succeeds", metrics)

    # ---- model checks

    def _gene_tool(self) -> ToolSpec:
        return _tool_spec("get_gene_info", "Look up a human gene by its HGNC symbol: full name, chromosome and a "
                          "short summary. One gene per call.", GENE_TOOL_SCHEMA)

    async def check_tool_call(self) -> CheckResult:
        resp, dt = await self._complete("Look up the human gene TP53 with the get_gene_info tool.",
                                        tools=[self._gene_tool()], thinking_budget=1024, max_tokens=4096,
                                        session_key="check-tool")
        calls = resp.message.tool_calls
        metrics = {"calls": [{"name": c.name, "input": c.input} for c in calls], "stop_reason": resp.stop_reason.value,
                   "latency_s": round(dt, 2), "output_tokens": resp.usage.output_tokens}
        threshold = "1 get_gene_info call with symbol TP53"
        bad = [_invalid(c) for c in calls if _invalid(c)]
        if bad:
            return CheckResult("tool_call", FAIL, f"invalid tool-call arguments: {bad[0]}", threshold, metrics)
        hit = [c for c in calls if c.name == "get_gene_info" and str(c.input.get("symbol", "")).upper() == "TP53"]
        if not hit:
            return CheckResult("tool_call", FAIL, f"no get_gene_info(symbol=TP53) call (calls: "
                               f"{[c.name for c in calls]}, text: {resp.message.text[:120]!r})", threshold, metrics)
        if resp.stop_reason != StopReason.TOOL_USE:
            return CheckResult("tool_call", WARN, f"call made but stop reason is {resp.stop_reason.value}",
                               threshold, metrics)
        return CheckResult("tool_call", PASS, f"get_gene_info(TP53) in {dt:.1f} s", threshold, metrics)

    async def check_parallel_tool_calls(self) -> CheckResult:
        prompt = ("Look up these three human genes with get_gene_info: TP53, BRCA1 and EGFR. The lookups are "
                  "independent, so issue all three get_gene_info calls together in this one turn (parallel tool "
                  "calls) before you answer.")
        resp, dt = await self._complete(prompt, tools=[self._gene_tool()], thinking_budget=1024, max_tokens=4096,
                                        session_key="check-parallel")
        calls = [c for c in resp.message.tool_calls if c.name == "get_gene_info" and not _invalid(c)]
        symbols = [str(c.input.get("symbol", "")).upper() for c in calls]
        threshold = f">= {self.opts.min_parallel_calls} calls in one turn"
        metrics = {"calls": len(resp.message.tool_calls), "valid_calls": len(calls), "symbols": symbols,
                   "latency_s": round(dt, 2)}
        if len(calls) >= self.opts.min_parallel_calls:
            return CheckResult("parallel_tool_calls", PASS, f"{len(calls)} calls in one turn ({', '.join(symbols)})",
                               threshold, metrics)
        return CheckResult("parallel_tool_calls", FAIL, f"only {len(calls)} valid call(s) in one turn; the CSO's "
                           "Task fan-out would serialise", threshold, metrics)

    async def check_strict_schema(self) -> CheckResult:
        from ..case_studies.trial_outcomes.schema import TrialAnnotation
        from ..tools.base import inline_refs
        raw = TrialAnnotation.model_json_schema()
        raw_text = json.dumps(raw)
        stats = {"anyOf": raw_text.count('"anyOf"'), "$ref": raw_text.count('"$ref"'),
                 "pattern": raw_text.count('"pattern"')}
        submit = _tool_spec("submit_result", "Submit your final structured result (validated against the required "
                            "schema). Call exactly once, when all fields are filled; this ends your task.",
                            inline_refs(raw), strict=True)
        threshold = "submit_result arguments validate as TrialAnnotation"
        resp, dt = await self._complete(TRIAL_PROMPT, tools=[submit], thinking_budget=1024, max_tokens=8192,
                                        session_key="check-strict")
        forced = False
        calls = [c for c in resp.message.tool_calls if c.name == "submit_result"]
        if not calls:
            forced = True
            resp, dt2 = await self._complete(TRIAL_PROMPT, tools=[submit], thinking_budget=1024, max_tokens=8192,
                                             session_key="check-strict", tool_choice={"name": "submit_result"})
            dt += dt2
            calls = [c for c in resp.message.tool_calls if c.name == "submit_result"]
        metrics: dict[str, Any] = {"schema": stats, "forced_retry": forced, "latency_s": round(dt, 2)}
        if not calls:
            return CheckResult("strict_schema", FAIL, "submit_result was not called, even with a forced tool_choice",
                               threshold, metrics)
        call = calls[0]
        if _invalid(call):
            return CheckResult("strict_schema", FAIL, f"invalid JSON arguments: {_invalid(call)}", threshold, metrics)
        try:
            ann = TrialAnnotation.model_validate(call.input)
        except ValidationError as exc:
            errs = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6]]
            metrics["errors"] = errs
            return CheckResult("strict_schema", FAIL, f"arguments fail the schema ({len(exc.errors())} error(s): "
                               f"{'; '.join(errs[:3])})", threshold, metrics)
        metrics["primary_endpoint_result"] = ann.primary_endpoint_result.value
        detail = (f"valid TrialAnnotation ({stats['anyOf']} anyOf, {stats['$ref']} $ref, {stats['pattern']} "
                  f"patterns)" + ("; needed a forced tool_choice" if forced else ""))
        return CheckResult("strict_schema", WARN if forced else PASS, detail, threshold, metrics)

    async def check_forced_tool_choice(self) -> CheckResult:
        answer = _tool_spec("submit_answer", "Submit the final answer.", {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "confidence": {"type": "string",
                                                                       "enum": ["high", "medium", "low"]}},
            "required": ["answer", "confidence"], "additionalProperties": False}, strict=True)
        resp, dt = await self._complete("Say hello and describe in one sentence what you can do.",
                                        tools=[self._gene_tool(), answer], thinking_budget=512, max_tokens=4096,
                                        session_key="check-forced", tool_choice={"name": "submit_answer"})
        calls = resp.message.tool_calls
        names = [c.name for c in calls]
        threshold = "exactly the forced tool (submit_answer), valid arguments"
        metrics = {"calls": names, "latency_s": round(dt, 2), "stop_reason": resp.stop_reason.value}
        if not calls:
            return CheckResult("forced_tool_choice", FAIL, f"no tool call (text: {resp.message.text[:120]!r}); "
                               "tool_choice is not enforced", threshold, metrics)
        if names[0] != "submit_answer" or any(n != "submit_answer" for n in names):
            return CheckResult("forced_tool_choice", FAIL, f"called {names} instead of submit_answer", threshold,
                               metrics)
        args = calls[0].input
        if _invalid(calls[0]) or not isinstance(args.get("answer"), str) or \
                args.get("confidence") not in ("high", "medium", "low"):
            return CheckResult("forced_tool_choice", FAIL, f"invalid submit_answer arguments {args!r}", threshold,
                               metrics)
        return CheckResult("forced_tool_choice", PASS, f"submit_answer forced ({dt:.1f} s)", threshold, metrics)

    async def check_reasoning_effort(self) -> CheckResult:
        tiers = (("xhigh", "xhigh", True), ("medium", "medium", True), ("none", None, False))
        per: dict[str, Any] = {}
        problems: list[str] = []
        for label, effort, thinking in tiers:
            resp, dt = await self._complete("What is 17 * 23? Reply with just the number.", effort=effort,
                                            thinking=thinking, max_tokens=4096,
                                            thinking_budget=1024 if thinking else None,
                                            session_key=f"check-effort-{label}")
            reasoning = _reasoning(resp)
            per[label] = {"reasoning_chars": len(reasoning), "output_tokens": resp.usage.output_tokens,
                          "correct": "391" in resp.message.text, "latency_s": round(dt, 2)}
            if thinking and not reasoning:
                problems.append(f"{label}: no reasoning returned (is --reasoning-parser set?)")
            if not thinking and reasoning:
                problems.append(f"{label}: reasoning returned with thinking off")
        threshold = "xhigh/medium return reasoning; none returns none"
        if problems:
            return CheckResult("reasoning_effort", FAIL, "; ".join(problems), threshold, per)
        detail = ", ".join(f"{k} {v['reasoning_chars']} chars" for k, v in per.items())
        return CheckResult("reasoning_effort", PASS, detail, threshold, per)

    async def check_thinking_budget(self) -> CheckResult:
        if not self.supports_budget:
            return CheckResult("thinking_budget", SKIP, f"family {self.family} has no thinking_token_budget")
        budget = int(self.opts.thinking_budget)
        limit = int(budget * self.opts.max_thinking_overrun) + 64
        threshold = f"reasoning <= {limit} tokens (budget {budget}), no max_tokens stop"
        resp, dt = await self._complete(
            "List every prime number below 400, double-checking each one, then count them. Final answer: only the "
            "count.", effort="medium", thinking=True, max_tokens=budget + 1536, thinking_budget=budget,
            session_key="check-budget")
        reasoning = _reasoning(resp)
        cpt = self.opts.chars_per_token
        out = resp.usage.output_tokens
        est = max(0, out - _est_tokens(resp.message.text, cpt)) if out else _est_tokens(reasoning, cpt)
        metrics = {"budget": budget, "reasoning_tokens_est": est, "reasoning_chars": len(reasoning),
                   "output_tokens": out, "stop_reason": resp.stop_reason.value, "latency_s": round(dt, 2)}
        if not reasoning:
            return CheckResult("thinking_budget", FAIL, "no reasoning returned", threshold, metrics)
        if resp.stop_reason in (StopReason.MAX_TOKENS, StopReason.CONTEXT_EXCEEDED) or est > limit:
            return CheckResult("thinking_budget", FAIL, f"reasoning ran to ~{est} tokens (stop "
                               f"{resp.stop_reason.value}): thinking_token_budget is not enforced", threshold, metrics)
        return CheckResult("thinking_budget", PASS, f"~{est} reasoning tokens for a {budget}-token budget", threshold,
                           metrics)

    async def check_needle(self, n_tokens: int) -> CheckResult:
        name = f"needle_{n_tokens // 1024}k"
        mlen = self.server.get("max_model_len")
        if mlen and n_tokens + 1024 > int(mlen):
            return CheckResult(name, SKIP, f"{n_tokens} tokens do not fit max_model_len {mlen}")
        rng = random.Random(f"{self.nonce}-needle-{n_tokens}")
        code = f"{rng.choice(_CODE_WORDS)}-{rng.randint(1000, 9999)}"
        doc = make_haystack(n_tokens, rng, needle=f"The access code for the archive vault is {code}.",
                            depth=self.opts.needle_depth, chars_per_token=self.opts.chars_per_token)
        prompt = (f"{doc}\n\nQuestion: what is the access code for the archive vault? Reply with the code only.")
        resp, dt = await self._complete(prompt, effort=None, thinking=False, max_tokens=64,
                                        system="You answer questions about the document the user provides.",
                                        session_key=f"check-needle-{n_tokens}")
        found = code.lower() in resp.message.text.lower()
        metrics = {"target_tokens": n_tokens, "prompt_tokens": _prompt_tokens(resp), "depth": self.opts.needle_depth,
                   "expected": code, "answer": resp.message.text[:80], "latency_s": round(dt, 2)}
        threshold = "the hidden code is returned"
        if found:
            return CheckResult(name, PASS, f"found at depth {self.opts.needle_depth:g} in a "
                               f"{_prompt_tokens(resp) or n_tokens}-token prompt ({dt:.1f} s)", threshold, metrics)
        return CheckResult(name, FAIL, f"expected {code}, got {resp.message.text[:60]!r}", threshold, metrics)

    async def check_prefix_cache(self) -> CheckResult:
        rng = random.Random(f"{self.nonce}-prefix")
        doc = f"[document {self.nonce}]\n\n" + make_haystack(self.opts.prefix_tokens, rng,
                                                               chars_per_token=self.opts.chars_per_token)
        system = "You are a careful reader. Answer in one word."
        r1, dt1 = await self._complete(f"{doc}\n\nQuestion: which season is mentioned most often?", effort=None,
                                       thinking=False, max_tokens=16, system=system, session_key="check-prefix")
        r2, dt2 = await self._complete(f"{doc}\n\nQuestion: is a river mentioned? Answer yes or no.", effort=None,
                                       thinking=False, max_tokens=16, system=system, session_key="check-prefix")
        cached1, cached2 = r1.usage.cache_read_tokens, r2.usage.cache_read_tokens
        prompt2 = _prompt_tokens(r2)
        ratio = round(cached2 / prompt2, 3) if prompt2 else 0.0
        metrics = {"prefix_tokens_target": self.opts.prefix_tokens, "first": {"prompt_tokens": _prompt_tokens(r1),
                   "cached_tokens": cached1, "latency_s": round(dt1, 2)},
                   "second": {"prompt_tokens": prompt2, "cached_tokens": cached2, "latency_s": round(dt2, 2)},
                   "hit_ratio": ratio}
        threshold = "cached_tokens > 0 on the repeated prefix"
        if cached2 <= 0:
            return CheckResult("prefix_cache", FAIL, "no cached tokens on the second request: start vLLM with "
                               "--enable-prefix-caching and --enable-prompt-tokens-details (without it usage has no "
                               "cached_tokens)", threshold, metrics)
        status = PASS if ratio >= 0.5 else WARN
        return CheckResult("prefix_cache", status, f"{cached2}/{prompt2} prompt tokens cached ({ratio:.0%})",
                           threshold, metrics)

    async def _load(self, n: int, output_tokens: int, provider: Any, *, prompt_tokens: int = 0,
                    tag: str = "tp") -> dict[str, Any]:
        """``n`` concurrent plain completions; per-request latency, TTFT and tokens."""
        rows: list[dict[str, Any]] = []

        async def one(i: int) -> None:
            first: list[float] = []
            t0 = time.perf_counter()

            def mark(_text: str) -> None:
                if not first:
                    first.append(time.perf_counter() - t0)
            rng = random.Random(f"{self.nonce}-{tag}-{i}")
            context = make_haystack(prompt_tokens, rng, chars_per_token=self.opts.chars_per_token) + "\n\n" \
                if prompt_tokens else ""
            topic = GENES[i % len(GENES)]
            prompt = (f"[{tag} {self.nonce} {i}] {context}Write a long, detailed essay about the biology of the "
                      f"{topic} gene and its role in disease.")
            try:
                resp, dt = await self._complete(prompt, effort=None, thinking=False, max_tokens=output_tokens,
                                                provider=provider, session_key=f"{tag}-{i}", on_text=mark,
                                                on_thinking=mark)
                rows.append({"latency_s": dt, "output_tokens": resp.usage.output_tokens,
                             "prompt_tokens": _prompt_tokens(resp), "ttft_s": first[0] if first else None})
            except Exception as exc:  # noqa: BLE001 - counted as an error
                rows.append({"error": f"{type(exc).__name__}: {exc}"})

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0
        ok = [r for r in rows if "error" not in r]
        out_tokens = sum(r["output_tokens"] for r in ok)
        lat = [r["latency_s"] for r in ok]
        ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
        per_req = [r["output_tokens"] / r["latency_s"] for r in ok if r["latency_s"] > 0]
        return {"concurrency": n, "requests": n, "errors": len(rows) - len(ok),
                "error_samples": [r["error"] for r in rows if "error" in r][:3],
                "output_tokens": out_tokens, "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
                "wall_s": round(wall, 3), "tokens_per_s": round(out_tokens / wall, 1) if wall > 0 else None,
                "requests_per_s": round(len(ok) / wall, 3) if wall > 0 else None,
                "mean_request_tokens_per_s": round(statistics.mean(per_req), 1) if per_req else None,
                "p50_latency_s": round(_pct(lat, 0.5), 2) if lat else None,
                "p90_latency_s": round(_pct(lat, 0.9), 2) if lat else None,
                "p50_ttft_s": round(_pct(ttft, 0.5), 3) if ttft else None}

    def _load_provider(self) -> Any:
        return self._new_provider({"ignore_eos": True}) if self.opts.ignore_eos else self.provider

    async def check_throughput(self) -> CheckResult:
        n = max(1, int(self.opts.concurrency))
        row = await self._load(n, int(self.opts.output_tokens), self._load_provider())
        threshold = f">= {self.opts.min_tokens_per_s:g} output tokens/s at concurrency {n}, no errors"
        if row["errors"]:
            return CheckResult("throughput", FAIL, f"{row['errors']}/{n} requests failed: {row['error_samples'][:1]}",
                               threshold, row)
        tps = row["tokens_per_s"] or 0.0
        detail = (f"{tps:g} tok/s aggregate at concurrency {n} ({row['mean_request_tokens_per_s']} tok/s per "
                  f"request, p50 latency {row['p50_latency_s']} s)")
        return CheckResult("throughput", PASS if tps >= self.opts.min_tokens_per_s else FAIL, detail, threshold, row)

    async def check_malformed_calls(self) -> CheckResult:
        k = max(1, int(self.opts.trials))
        tool = _tool_spec("record_findings", "Record literature findings about one gene: each finding is a claim "
                          "with an evidence level and optional PubMed IDs.", FINDINGS_SCHEMA)
        sem = asyncio.Semaphore(max(1, min(k, int(self.opts.concurrency))))
        outcomes: list[str] = []
        samples: list[str] = []

        async def trial(i: int) -> None:
            gene = GENES[i % len(GENES)]
            prompt = (f"Record two findings about the human gene {gene} with the record_findings tool: one with "
                      "strong and one with moderate evidence, overall confidence 0.8. Include PubMed IDs only if "
                      "you are sure of them.")
            async with sem:
                try:
                    resp, _ = await self._complete(prompt, tools=[tool], thinking_budget=1024, max_tokens=4096,
                                                   session_key=f"check-malformed-{i}")
                except Exception as exc:  # noqa: BLE001
                    outcomes.append("error")
                    samples.append(f"error: {type(exc).__name__}: {exc}"[:200])
                    return
            calls = resp.message.tool_calls
            if not calls:
                outcomes.append("no_call")
                samples.append(f"no call: {resp.message.text[:100]!r}")
                return
            for c in calls:
                if c.name != "record_findings":
                    outcomes.append("wrong_tool")
                    samples.append(f"wrong tool {c.name!r}")
                    return
                if _invalid(c):
                    outcomes.append("invalid_json")
                    samples.append(f"invalid JSON: {(c.native or {}).get('invalid_arguments')!r}"[:200])
                    return
                try:
                    _Findings.model_validate(c.input)
                except ValidationError as exc:
                    outcomes.append("schema")
                    samples.append(f"schema: {exc.errors()[0]['msg']}")
                    return
            outcomes.append("ok")

        await asyncio.gather(*(trial(i) for i in range(k)))
        errors = outcomes.count("error")
        done = k - errors
        bad = done - outcomes.count("ok")
        rate = round(bad / done, 3) if done else 1.0
        breakdown = {o: outcomes.count(o) for o in sorted(set(outcomes))}
        metrics = {"trials": k, "completed": done, "malformed": bad, "rate": rate, "breakdown": breakdown,
                   "samples": samples[:5]}
        threshold = f"malformed rate <= {self.opts.max_malformed_rate:g} over {k} trials"
        if not done:
            return CheckResult("malformed_calls", FAIL, f"all {k} trials failed: {samples[:1]}", threshold, metrics)
        status = PASS if rate <= self.opts.max_malformed_rate and not errors else FAIL
        detail = f"{bad}/{done} malformed ({rate:.0%})" + (f"; {errors} request error(s)" if errors else "")
        return CheckResult("malformed_calls", status, detail, threshold, metrics)

    # ---- driver

    async def run(self) -> CheckReport:
        report = CheckReport(base_url=self.opts.base_url, model=self.model, family=None,
                             started_at=_now(), options=self.opts.as_dict())
        plan: list[tuple[str, Callable[[], Awaitable[CheckResult]]]] = [
            ("health", self.check_health), ("models", self.check_models), ("version", self.check_version),
            ("prepare", self.check_prepare), ("tool_call", self.check_tool_call),
            ("parallel_tool_calls", self.check_parallel_tool_calls), ("strict_schema", self.check_strict_schema),
            ("forced_tool_choice", self.check_forced_tool_choice),
            ("reasoning_effort", self.check_reasoning_effort), ("thinking_budget", self.check_thinking_budget)]
        for n in self.opts.needle_tokens:
            plan.append((f"needle_{int(n) // 1024}k", lambda n=int(n): self.check_needle(n)))
        plan += [("prefix_cache", self.check_prefix_cache), ("throughput", self.check_throughput),
                 ("malformed_calls", self.check_malformed_calls)]
        # The server facts (and, for model checks, the adapter) are prerequisites.
        required = {"health", "models"}
        if any(self._selected(name) for name, _ in plan if _is_model_check(name)):
            required.add("prepare")
        try:
            for name, fn in plan:
                if not self._selected(name) and name not in required:
                    continue
                res = await self._run(name, fn)
                if name == "prepare" and res.status == FAIL:
                    self._no_model = self._no_model or "the adapter's prepare() failed"
                if self._selected(name) or res.status == FAIL:
                    report.checks.append(res)
        finally:
            await self.aclose()
        report.model, report.family, report.server = self.model, self.family, dict(self.server)
        report.finished_at = _now()
        return report


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_model_check(name: str) -> bool:
    return name in MODEL_CHECKS or name.startswith("needle_")


async def run_checks(opts: CheckOptions, provider_factory: ProviderFactory | None = None, *,
                     http_transport: httpx.AsyncBaseTransport | None = None) -> CheckReport:
    """Run the selected checks against ``opts.base_url``."""
    return await LocalChecker(opts, provider_factory, http_transport=http_transport).run()


async def run_bench(opts: CheckOptions, levels: Sequence[int] = (1, 8, 32), *, output_tokens: int = 256,
                    prompt_tokens: int = 512, provider_factory: ProviderFactory | None = None) -> dict[str, Any]:
    """Throughput sweep: ``level`` concurrent requests per level (``output_tokens``
    each with ignore_eos, prompts of ~``prompt_tokens``)."""
    checker = LocalChecker(opts, provider_factory)
    started = _now()
    rows: list[dict[str, Any]] = []
    try:
        models = await checker.check_models()
        if models.status == FAIL:
            return {"tool": "vbt local bench", "ok": False, "error": models.detail, "base_url": opts.base_url,
                    "started_at": started, "levels": []}
        checker.family, _ = family_traits(checker.model, opts.family)
        checker.provider = checker._new_provider()
        provider = checker._load_provider()
        for level in levels:
            rows.append(await checker._load(int(level), int(output_tokens), provider, prompt_tokens=prompt_tokens,
                                            tag=f"bench{level}"))
    finally:
        await checker.aclose()
    return {"tool": "vbt local bench", "ok": all(r["errors"] == 0 for r in rows), "base_url": opts.base_url,
            "model": checker.model, "family": checker.family, "server": checker.server, "started_at": started,
            "finished_at": _now(), "output_tokens": output_tokens, "prompt_tokens": prompt_tokens,
            "ignore_eos": opts.ignore_eos, "levels": rows}


def bench_markdown(result: Mapping[str, Any]) -> str:
    lines = [f"# Throughput sweep: {result.get('model') or '?'} at {result.get('base_url')}", "",
             f"- Started: {result.get('started_at')}; {result.get('output_tokens')} output tokens per request "
             f"(ignore_eos {result.get('ignore_eos')}), prompts of ~{result.get('prompt_tokens')} tokens", "",
             "| Concurrency | Errors | Output tokens | Wall (s) | Aggregate tok/s | Per-request tok/s | "
             "p50 latency (s) | p90 latency (s) | p50 TTFT (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in result.get("levels") or []:
        lines.append(f"| {r['concurrency']} | {r['errors']} | {r['output_tokens']} | {r['wall_s']} | "
                     f"{r['tokens_per_s']} | {r['mean_request_tokens_per_s']} | {r['p50_latency_s']} | "
                     f"{r['p90_latency_s']} | {r['p50_ttft_s']} |")
    if result.get("error"):
        lines += ["", f"Error: {result['error']}"]
    return "\n".join(lines) + "\n"


def write_report(out_dir: str | Path, data: Mapping[str, Any], markdown: str, stem: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp, mp = out / f"{stem}.json", out / f"{stem}.md"
    jp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    mp.write_text(markdown if markdown.endswith("\n") else markdown + "\n")
    return jp, mp


# ---------------------------------------------------------------- CLI

def _csv(value: Any) -> list[str]:
    out: list[str] = []
    for v in value or []:
        out += [p.strip() for p in str(v).split(",") if p.strip()]
    return out


def options_from_config(config: Mapping[str, Any] | None, args: argparse.Namespace | None = None) -> CheckOptions:
    """Probe options from the harness config (when its provider is a local one)
    and the command-line flags (which win)."""
    config = config or {}
    prov = config.get("provider") or {}
    local = prov.get("name") in LOCAL_PROVIDERS
    popts = dict(prov.get("options") or {}) if local else {}
    get = (lambda k, d=None: getattr(args, k, d) if args is not None else d)
    base_urls = popts.get("base_urls")
    if isinstance(base_urls, str):
        base_urls = [u for u in base_urls.replace(",", " ").split() if u]
    base_url = (get("base_url") or popts.get("base_url") or (base_urls[0] if base_urls else None)
                or os.environ.get("VBT_LLM_BASE_URL") or DEFAULT_BASE_URL)
    models = config.get("models") or {}
    tier_model = (models.get("scientist") or {}).get("model") if local and isinstance(models, Mapping) else None
    model = get("served_model") or popts.get("served_model_name") or popts.get("model") or tier_model
    windows = [int(t["context_window_tokens"]) for t in models.values()
               if local and isinstance(t, Mapping) and t.get("context_window_tokens")] if isinstance(models, Mapping) \
        else []
    opts = CheckOptions(base_url=str(base_url), model=model or None,
                        family=str(get("family") or popts.get("family") or "auto"),
                        provider_name=str(prov.get("name")) if local else "vllm",
                        api_key=popts.get("api_key") or None,
                        read_timeout_s=float(popts.get("read_timeout_s") or 600.0))
    opts.min_context_tokens = get("min_context") or (max(windows) if windows else None)
    if args is None:
        return opts
    for flag, attr, conv in (("timeout", "check_timeout_s", float), ("prefix_tokens", "prefix_tokens", int),
                             ("thinking_budget", "thinking_budget", int), ("concurrency", "concurrency", int),
                             ("output_tokens", "output_tokens", int), ("trials", "trials", int),
                             ("min_tps", "min_tokens_per_s", float), ("max_malformed_rate", "max_malformed_rate", float),
                             ("seed", "seed", int), ("min_version", "min_engine_version", str)):
        v = getattr(args, flag, None)
        if v is not None:
            setattr(opts, attr, conv(v))
    needles = [int(x) for x in _csv(getattr(args, "needle_tokens", None))] or list(opts.needle_tokens)
    if getattr(args, "long_context", None):
        needles.append(int(args.long_context))
    opts.needle_tokens = tuple(dict.fromkeys(needles))
    opts.only = tuple(_csv(getattr(args, "only", None)))
    opts.skip = tuple(_csv(getattr(args, "skip", None)))
    if getattr(args, "no_ignore_eos", False):
        opts.ignore_eos = False
    unknown = [n for n in (*opts.only, *opts.skip) if n not in CHECK_NAMES and not n.startswith("needle")]
    if unknown:
        raise ValueError(f"unknown check(s) {unknown}; known: {', '.join(CHECK_NAMES)}")
    return opts


def _out_dir(args: argparse.Namespace, config: Mapping[str, Any] | None, kind: str) -> Path:
    if getattr(args, "out", None):
        return Path(args.out)
    from ..config import resolve_path
    runs = ((config or {}).get("paths") or {}).get("runs_dir") or "runs"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return resolve_path(runs) / "local" / f"{kind}-{stamp}"


def cmd_check(args: argparse.Namespace, config: Mapping[str, Any] | None = None) -> int:
    try:
        opts = options_from_config(config, args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = asyncio.run(run_checks(opts))
    data = report.as_dict()
    jp, mp = write_report(_out_dir(args, config, "check"), data, report.markdown(), "check")
    if getattr(args, "json", False):
        print(json.dumps(data, indent=1, default=str))
    else:
        c = report.counts()
        print(f"vbt local check: {report.model or '?'} at {report.base_url}")
        width = max(len(r.name) for r in report.checks) if report.checks else 10
        for r in report.checks:
            print(f"  {r.status.upper():4}  {r.name.ljust(width)}  {r.detail}")
        print(f"{'PASS' if report.ok else 'FAIL'}: {c[PASS]} passed, {c[FAIL]} failed, {c[WARN]} warnings, "
              f"{c[SKIP]} skipped")
        print(f"report: {mp}\n        {jp}")
    return 0 if report.ok else 1


def cmd_bench(args: argparse.Namespace, config: Mapping[str, Any] | None = None) -> int:
    try:
        opts = options_from_config(config, args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    levels = [int(x) for x in _csv(getattr(args, "levels", None))] or [1, 8, 32]
    result = asyncio.run(run_bench(opts, levels, output_tokens=int(getattr(args, "output_tokens", None) or 256),
                                   prompt_tokens=int(getattr(args, "prompt_tokens", None) or 512)))
    md = bench_markdown(result)
    jp, mp = write_report(_out_dir(args, config, "bench"), result, md, "bench")
    if getattr(args, "json", False):
        print(json.dumps(result, indent=1, default=str))
    else:
        print(md)
        print(f"report: {mp}\n        {jp}")
    return 0 if result.get("ok") else 1


def _add_target_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", help="server API base (default: provider.options.base_url, $VBT_LLM_BASE_URL or "
                                      f"{DEFAULT_BASE_URL})")
    p.add_argument("--served-model", metavar="NAME", help="served model name (default: provider.options."
                                                          "served_model_name, else the first served model)")
    p.add_argument("--family", help="model family (default: provider.options.family or auto)")
    p.add_argument("--out", metavar="DIR", help="report directory (default <runs_dir>/local/<kind>-<UTC time>)")
    p.add_argument("--json", action="store_true", help="print the JSON report")
    p.add_argument("--no-ignore-eos", action="store_true", help="throughput: stop at EOS instead of generating "
                                                                "exactly --output-tokens")


def add_check_parser(lsub: Any) -> argparse.ArgumentParser:
    p = lsub.add_parser("check", help="probe a running server: tools, strict schema, reasoning, long context, "
                                      "prefix cache, throughput")
    _add_target_arguments(p)
    p.add_argument("--only", action="append", metavar="CHECKS", help=f"comma list of checks ({', '.join(CHECK_NAMES)})")
    p.add_argument("--skip", action="append", metavar="CHECKS", help="comma list of checks to skip")
    p.add_argument("--needle-tokens", action="append", metavar="N[,N]", help="needle haystack sizes (default 32768)")
    p.add_argument("--long-context", type=int, nargs="?", const=200000, metavar="N",
                   help="also run a long needle (default 200000 tokens)")
    p.add_argument("--prefix-tokens", type=int, help="prefix-cache probe size (default 12000)")
    p.add_argument("--thinking-budget", type=int, help="thinking_token_budget probe (default 256)")
    p.add_argument("--concurrency", type=int, help="throughput / malformed-call concurrency (default 8)")
    p.add_argument("--output-tokens", type=int, help="tokens per throughput request (default 256)")
    p.add_argument("--trials", type=int, help="malformed-call trials (default 10)")
    p.add_argument("--min-tps", type=float, help="throughput threshold, aggregate tokens/s (default 20)")
    p.add_argument("--max-malformed-rate", type=float, help="malformed-call threshold (default 0.1)")
    p.add_argument("--min-context", type=int, help="required max_model_len (default: the configured context window)")
    p.add_argument("--min-version", help="required engine version (default 0.31.0; lower is a warning)")
    p.add_argument("--timeout", type=float, help="wall-clock cap per check in seconds (default 900)")
    p.add_argument("--seed", type=int, help="seed for the generated documents")
    p.set_defaults(handler=cmd_check)
    return p


def add_bench_parser(lsub: Any) -> argparse.ArgumentParser:
    p = lsub.add_parser("bench", help="throughput sweep at concurrency 1/8/32")
    _add_target_arguments(p)
    p.add_argument("--levels", action="append", metavar="N[,N]", help="concurrency levels (default 1,8,32)")
    p.add_argument("--output-tokens", type=int, help="tokens per request (default 256)")
    p.add_argument("--prompt-tokens", type=int, help="prompt size per request (default 512)")
    p.set_defaults(handler=cmd_bench)
    return p


__all__ = [
    "CHECK_NAMES", "CheckOptions", "CheckReport", "CheckResult", "LocalChecker", "add_bench_parser",
    "add_check_parser", "bench_markdown", "cmd_bench", "cmd_check", "default_provider_factory", "family_traits",
    "make_haystack", "options_from_config", "run_bench", "run_checks", "split_base_url", "write_report",
]
