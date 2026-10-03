"""``vbt replay``: re-run a recorded session's turns and compare the two runs.

LLM sampling makes a replay a *comparison*, not a reproduction: the same
pinned models, prompts and turns are sent again, and ``replay_diff.json``
records what matches (agents dispatched, artifact names and byte-identical
hashes, claims, per-turn status and cost, verify status). For an exact
re-execution of the code the agents wrote, use ``vbt verify --rerun``.

* ``parse_turns(source)`` -- the user turns of a run: ``inputs/query.txt``
  ``--- turn N ---`` blocks, falling back to ``session_report.json`` prompts.
* ``replay_run(run_dir, model=None, quiet=False)`` -- loads the pinned
  ``inputs/config.json`` (profiles, models, web, orchestration), warns on
  harness/submodule commit drift and prompt-hash drift, replays the turns into
  a new run and writes ``replay_diff.json`` there.
* ``compare_runs(a, b)`` / ``format_diff(diff)``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = ["parse_turns", "load_pinned", "drift_warnings", "replay_run", "compare_runs", "format_diff",
           "CAVEAT", "DIFF_FILE"]

DIFF_FILE = "replay_diff.json"
CAVEAT = ("LLM sampling makes this a comparison, not a reproduction: the same turns were sent to the same "
          "pinned models and prompts, but outputs differ between samples. For an exact re-execution of the "
          "code the agents wrote, run 'vbt verify --rerun <run>'.")

_TURN_RE = re.compile(r"^--- turn (\d+) ---[ \t]*$", re.MULTILINE)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _parse_blocks(text: str) -> list[str]:
    marks = list(_TURN_RE.finditer(text or ""))
    if not marks:
        return []
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append(text[m.end():end].strip("\n").strip())
    return out


def _report_prompts(run_dir: Path) -> list[str]:
    rep = _read_json(run_dir / "session_report.json", {})
    turns = rep.get("turns") if isinstance(rep, Mapping) else None
    out = []
    for t in sorted((t for t in turns or [] if isinstance(t, Mapping)), key=lambda t: t.get("turn") or 0):
        if t.get("prompt"):
            out.append(str(t["prompt"]))
    return out


def parse_turns(source: str | Path) -> list[str]:
    """User turns from query.txt text, a query.txt path, or a run directory.

    ``--- turn N ---`` blocks are split in order. When no block is found, the
    prompts of ``session_report.json`` are used (paths only); plain legacy text
    without markers is a single turn.
    """
    if isinstance(source, str) and not source.strip():
        return []
    run_dir: Path | None = None
    text = ""
    p = Path(str(source)) if not isinstance(source, str) or "\n" not in source else None
    if p is not None and p.exists():
        if p.is_dir():
            run_dir = p
            p = p / "inputs" / "query.txt"
        else:
            run_dir = p.parent.parent if p.parent.name == "inputs" else p.parent
        try:
            text = p.read_text(encoding="utf-8") if p.is_file() else ""
        except OSError:
            text = ""
    elif isinstance(source, str):
        text = source
    turns = [t for t in _parse_blocks(text) if t]
    if turns:
        return turns
    if run_dir is not None:
        turns = _report_prompts(run_dir)
        if turns:
            return turns
    return [text.strip()] if text.strip() else []


def load_pinned(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = _read_json(run_dir / "inputs" / "config.json", None)
    if not isinstance(cfg, dict):
        man = _read_json(run_dir / "MANIFEST.json", {})
        cfg = man.get("config") if isinstance(man, Mapping) and isinstance(man.get("config"), dict) else {}
    return cfg


def drift_warnings(pinned: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Differences between a run's pinned harness/prompts and the current checkout."""
    from .pinning import git_info, prompt_hashes

    out: list[str] = []
    then = pinned.get("harness") or {}
    now = git_info((config.get("vars") or {}).get("upstream"))
    if then.get("commit") and now.get("commit") and then["commit"] != now["commit"]:
        out.append(f"harness commit drift: run used {str(then['commit'])[:12]}, current checkout is "
                   f"{str(now['commit'])[:12]}")
    if then.get("dirty"):
        out.append("the run was made from a harness checkout with uncommitted changes")
    if then.get("upstream_commit") and now.get("upstream_commit") and then["upstream_commit"] != now["upstream_commit"]:
        out.append(f"upstream submodule drift: run used {str(then['upstream_commit'])[:12]}, current is "
                   f"{str(now['upstream_commit'])[:12]}")
    old = pinned.get("prompt_hashes") or {}
    if old:
        cur = prompt_hashes(config)
        for group in ("upstream", "local"):
            a, b = old.get(group) or {}, cur.get(group) or {}
            changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
            gone = sorted(set(a) - set(b))
            new = sorted(set(b) - set(a))
            if changed:
                out.append(f"{group} prompt drift (changed): {', '.join(changed[:10])}"
                           + (f" (+{len(changed) - 10} more)" if len(changed) > 10 else ""))
            if gone:
                out.append(f"{group} prompts missing now: {', '.join(gone[:10])}")
            if new:
                out.append(f"{group} prompts added since the run: {', '.join(new[:10])}")
    return out


def _replay_config(pinned: Mapping[str, Any], *, model: str | None, runs_dir: str | Path | None,
                   profiles: list[str] | None = None,
                   extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The pinned run's configuration; ``profiles`` replaces the pinned profile
    list and ``extra`` (CLI flag overrides such as ``preflight`` or ``web``) is
    merged on top of the pinned sections."""
    from .config import deep_merge, load_config

    profs = list(profiles if profiles is not None else (pinned.get("profiles") or []))
    overrides: dict[str, Any] = {}
    for key in ("models", "web", "orchestration", "limits", "agent_overrides", "model_aliases"):
        if isinstance(pinned.get(key), dict) and pinned[key]:
            overrides[key] = pinned[key]
    prov = pinned.get("provider")
    pname = prov.get("name") if isinstance(prov, Mapping) else prov if isinstance(prov, str) else None
    if pname:
        overrides["provider"] = {"name": pname}  # options were redacted when pinned; profiles restore them
    if runs_dir is not None:
        overrides["paths"] = {"runs_dir": str(runs_dir)}
    if extra:
        overrides = deep_merge(overrides, dict(extra))
    config = load_config(profs, overrides)
    if model:
        from .cli import resolve_model
        m = resolve_model(config, model)
        for tier in ("orchestrator", "scientist", "bulk"):
            if tier in config["models"]:
                config["models"][tier] = {**config["models"][tier], "model": m}
    config["profiles"] = profs
    return config


async def replay_run(run_dir: str | Path, *, model: str | None = None, quiet: bool = False, provider=None,
                     on_event: Callable[..., Any] | None = None, runs_dir: str | Path | None = None,
                     start_mcp: bool = True, profiles: list[str] | None = None,
                     config: dict[str, Any] | None = None, echo: Callable[[str], None] | None = None,
                     overrides: Mapping[str, Any] | None = None
                     ) -> tuple[Path, dict[str, Any]]:
    """Replay ``run_dir``'s turns into a new run (next to it unless ``runs_dir``).

    ``profiles`` replaces the pinned profile list; ``overrides`` (e.g. the CLI's
    ``preflight`` / ``web`` flags) is merged over the pinned settings.

    Returns ``(new_run_dir, diff)``; the diff is also written to
    ``<new run>/replay_diff.json``. ``quiet`` suppresses progress lines.
    """
    from .orchestrator import open_session

    src = Path(run_dir).expanduser().resolve()
    say = echo or (lambda s: None if quiet else print(s))
    pinned = load_pinned(src)
    turns = parse_turns(src)
    if not turns:
        raise ValueError(f"no turns recorded in {src} (inputs/query.txt and session_report.json are empty)")
    if config is None:
        config = _replay_config(pinned, model=model, runs_dir=runs_dir or src.parent, profiles=profiles,
                                extra=overrides)
    warnings = drift_warnings(pinned, config)
    for w in warnings:
        say(f"warning: {w}")
    session = await open_session(config, provider=provider, on_event=on_event, start_mcp=start_mcp,
                                 interface="replay", profiles=tuple(config.get("profiles") or ()))
    new_dir = session.run.dir
    session.run.trace("replay_of", run_id=src.name, run_dir=str(src), n_turns=len(turns), drift=warnings)
    errors: list[dict[str, Any]] = []
    try:
        for i, q in enumerate(turns, 1):
            say(f"[replay] turn {i}/{len(turns)}")
            try:
                await session.ask(q)
            except Exception as exc:  # noqa: BLE001 - keep replaying; the turn is recorded as failed
                errors.append({"turn": i, "error": f"{type(exc).__name__}: {exc}"})
                say(f"[ERROR] Turn {i} did not complete: {type(exc).__name__}: {exc}")
    finally:
        await session.close()
    diff = compare_runs(src, new_dir, write=False)
    diff["drift_warnings"] = warnings
    diff["replay_errors"] = errors
    diff["model_override"] = model
    _write_diff(new_dir, diff)
    return new_dir, diff


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _agents(run_dir: Path) -> list[str]:
    man = _read_json(run_dir / "MANIFEST.json", {}) or {}
    names = [a for a in man.get("agents") or [] if isinstance(a, str)]
    rep = _read_json(run_dir / "session_report.json", {}) or {}
    for t in rep.get("turns") or []:
        if isinstance(t, Mapping):
            names += [a for a in t.get("agents") or [] if isinstance(a, str)]
    return sorted(set(a for a in names if a and a != "cso" and not a.startswith("_")))


def _artifacts(run_dir: Path) -> dict[str, str | None]:
    """{path under work/: sha256} from the MANIFEST (rescanned when absent)."""
    man = _read_json(run_dir / "MANIFEST.json", {}) or {}
    arts = man.get("artifacts")
    out: dict[str, str | None] = {}
    if isinstance(arts, Mapping) and arts:
        for rel, e in arts.items():
            out[str(rel)] = (e or {}).get("sha256") if isinstance(e, Mapping) else None
        return out
    if isinstance(arts, list):
        for e in arts:
            if isinstance(e, Mapping) and e.get("path"):
                out[str(e["path"])] = e.get("sha256")
        if out:
            return out
    from .audit.storage import sha256_file
    work = run_dir / "work"
    if work.is_dir():
        for p in sorted(work.rglob("*")):
            if p.is_file():
                try:
                    out[p.relative_to(run_dir).as_posix()] = sha256_file(p)
                except OSError:
                    pass
    return out


def _claims(run_dir: Path) -> list[dict[str, Any]]:
    try:
        from .audit.claims import load_claims
        return [c for c in load_claims(run_dir) if isinstance(c, Mapping)]
    except Exception:  # noqa: BLE001
        data = _read_json(run_dir / "evidence" / "claims.json", {})
        if isinstance(data, Mapping):
            data = data.get("claims")
        return [c for c in data or [] if isinstance(c, Mapping)]


def _turns(run_dir: Path) -> list[dict[str, Any]]:
    rep = _read_json(run_dir / "session_report.json", {}) or {}
    return [t for t in rep.get("turns") or [] if isinstance(t, Mapping)]


def _verify_status(run_dir: Path) -> str | None:
    try:
        from .verify import verify_run
        return str(verify_run(run_dir).get("status"))
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}: {exc}"


def _total_cost(run_dir: Path) -> float | None:
    rep = _read_json(run_dir / "logs" / "cost_report.json", {}) or {}
    try:
        return round(float(rep.get("total_usd")), 6) if rep.get("total_usd") is not None else None
    except (TypeError, ValueError):
        return None


def _write_diff(run_dir: Path, diff: Mapping[str, Any]) -> Path:
    from .audit.storage import write_text_atomic
    path = Path(run_dir) / DIFF_FILE
    write_text_atomic(path, json.dumps(diff, indent=2, default=str, ensure_ascii=False))
    return path


def compare_runs(a: str | Path, b: str | Path, *, write: bool = True, verify: bool = True) -> dict[str, Any]:
    """Compare run ``a`` (original) with run ``b`` (replay); writes ``b/replay_diff.json``."""
    a, b = Path(a).resolve(), Path(b).resolve()
    ag_a, ag_b = set(_agents(a)), set(_agents(b))
    ar_a, ar_b = _artifacts(a), _artifacts(b)
    same = sorted(set(ar_a) & set(ar_b))
    identical = [p for p in same if ar_a[p] and ar_a[p] == ar_b[p]]
    cl_a, cl_b = _claims(a), _claims(b)
    texts_b = {_norm(c.get("text")): c for c in cl_b if c.get("text")}
    matched = [{"text": c.get("text"), "id_a": c.get("id"), "id_b": texts_b[_norm(c.get("text"))].get("id")}
               for c in cl_a if c.get("text") and _norm(c.get("text")) in texts_b]
    ta, tb = _turns(a), _turns(b)
    by_a = {t.get("turn"): t for t in ta}
    by_b = {t.get("turn"): t for t in tb}
    turns = []
    for n in sorted({k for k in list(by_a) + list(by_b) if isinstance(k, int)}):
        x, y = by_a.get(n) or {}, by_b.get(n) or {}
        turns.append({"turn": n, "prompt": (x.get("prompt") or y.get("prompt") or "")[:300],
                      "status_a": x.get("status"), "status_b": y.get("status"),
                      "cost_a": x.get("cost_usd"), "cost_b": y.get("cost_usd"),
                      "agents_a": x.get("agents") or [], "agents_b": y.get("agents") or []})
    diff: dict[str, Any] = {
        "original": {"run_id": a.name, "run_dir": str(a)},
        "replay": {"run_id": b.name, "run_dir": str(b)},
        "caveat": CAVEAT,
        "agents": {"both": sorted(ag_a & ag_b), "only_original": sorted(ag_a - ag_b),
                   "only_replay": sorted(ag_b - ag_a)},
        "artifacts": {"n_original": len(ar_a), "n_replay": len(ar_b), "same_name": same,
                      "byte_identical": identical,
                      "differing": [p for p in same if p not in identical],
                      "only_original": sorted(set(ar_a) - set(ar_b)), "only_replay": sorted(set(ar_b) - set(ar_a))},
        "claims": {"n_original": len(cl_a), "n_replay": len(cl_b), "n_matched_text": len(matched),
                   "matched": matched},
        "turns": turns,
        "cost_usd": {"original": _total_cost(a), "replay": _total_cost(b)},
    }
    if verify:
        diff["verify"] = {"original": _verify_status(a), "replay": _verify_status(b)}
    if write:
        _write_diff(b, diff)
    return diff


def format_diff(diff: Mapping[str, Any]) -> str:
    """Short human summary of a replay diff."""
    ag = diff.get("agents") or {}
    ar = diff.get("artifacts") or {}
    cl = diff.get("claims") or {}
    cost = diff.get("cost_usd") or {}
    ver = diff.get("verify") or {}
    lines = [f"Replay of {diff.get('original', {}).get('run_id')} -> {diff.get('replay', {}).get('run_id')}",
             f"  agents: {len(ag.get('both') or [])} in both"
             + (f"; only original: {', '.join(ag['only_original'])}" if ag.get("only_original") else "")
             + (f"; only replay: {', '.join(ag['only_replay'])}" if ag.get("only_replay") else ""),
             f"  artifacts: {ar.get('n_original', 0)} vs {ar.get('n_replay', 0)}; "
             f"{len(ar.get('same_name') or [])} same name, {len(ar.get('byte_identical') or [])} byte-identical",
             f"  claims: {cl.get('n_original', 0)} vs {cl.get('n_replay', 0)}; "
             f"{cl.get('n_matched_text', 0)} with matching text"]
    for t in diff.get("turns") or []:
        lines.append(f"  turn {t.get('turn')}: {t.get('status_a')} -> {t.get('status_b')}"
                     + (f" (${float(t['cost_a'] or 0):.2f} -> ${float(t['cost_b'] or 0):.2f})"
                        if t.get("cost_a") is not None or t.get("cost_b") is not None else ""))
    if cost:
        lines.append(f"  cost: ${float(cost.get('original') or 0):.2f} -> ${float(cost.get('replay') or 0):.2f}")
    if ver:
        lines.append(f"  verify: {ver.get('original')} -> {ver.get('replay')}")
    for w in diff.get("drift_warnings") or []:
        lines.append(f"  warning: {w}")
    for e in diff.get("replay_errors") or []:
        lines.append(f"  [ERROR] turn {e.get('turn')}: {e.get('error')}")
    lines += ["", CAVEAT]
    return "\n".join(lines)
