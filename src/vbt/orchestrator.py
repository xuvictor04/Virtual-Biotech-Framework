"""CSO session: the multi-turn Virtual Biotech workflow (Fig. 1C).

Turn 1 (strategic orientation, if enabled)
    The Chief of Staff prepares a briefing *in parallel* with the CSO's
    clarification interview (``Runtime.delegate``, so the brief is a recorded
    delegation). If the CSO has questions, the turn ends with the briefing and
    the questions; otherwise it proceeds straight to work. An introduction or
    meta question (``META_QUERY``) is answered directly and orientation waits
    for the first scientific query. A failed brief degrades to a notice.
Every research turn
    The CSO decomposes the query and delegates with ``Task`` (parallel where
    independent). Before the CSO's synthesis is accepted, the harness checks
    the review policy (``orchestration.review_policy``): a specialist counts as
    reviewed only if a ``scientific-reviewer`` delegation *started after* it
    ended. Unreviewed work re-opens the loop (bounded by ``max_review_rounds``);
    when rounds run out the turn is recorded ``reviewed: false``. With
    ``orchestration.enforce_plan`` a multi-specialist turn without a recorded
    plan gets one plan nudge.
Follow-ups
    The CSO conversation is persistent (append-only, saved to
    ``logs/cso_state.json`` after every turn so ``--resume`` can continue it),
    while each delegation starts a fresh specialist.

Failure semantics: an interrupted (cancelled), budget-stopped or crashed turn
is still recorded (``status`` interrupted / budget_exceeded / failed: ...)
with the partial CSO text and a notice, the history is repaired so the next
``ask`` is valid, and the exception propagates. The end-of-turn data warning
names only *unresolved* data-source failures (``vbt.failures``).

Readiness (``vbt.preflight``): ``open_session`` refuses to create a run when
credentials, reference data or MCP commands are missing (unless
``preflight.skip``; skipped for the mock provider and when
``orchestration.require_reference_data`` is false). With
``preflight.allow_missing_data`` the run proceeds degraded (MANIFEST.degraded)
and every agent is told which servers lack data. Each turn re-checks
credentials and data before any model call; a failing check records the turn
as ``not_sent`` ("This turn has not been sent to the model").

Config (in-code defaults): ``orchestration.review_policy`` (``research``),
``orchestration.enforce_plan`` (false), ``orchestration.max_review_rounds`` (2),
``orchestration.strategic_orientation`` (true), ``limits.max_turn_cost_usd``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import failures as _failures
from .budget import BudgetExceeded
from .providers.base import Message, message_from_dict, message_to_dict
from .runtime import Runtime

log = logging.getLogger(__name__)

__all__ = ["CSOSession", "SessionBusy", "open_session", "NO_CLARIFY", "META_QUERY", "AWAITING_USER",
           "ORIENTATION_INSTRUCTION", "REVIEW_NUDGE", "PLAN_NUDGE", "STATE_FILE"]

NO_CLARIFY = "NO_CLARIFICATION_NEEDED"
META_QUERY = "META_QUERY"
AWAITING_USER = "[AWAITING_USER]"
STATE_FILE = "logs/cso_state.json"

ORIENTATION_INSTRUCTION = f"""[Harness — strategic orientation]
This is the first scientific query of the session. Your Chief of Staff is preparing an
intelligence briefing in parallel (field overview, data landscape, recent developments),
which you will receive before any analysis starts, so do not request one yourself.

Right now, conduct your clarification interview: if the query is ambiguous in scope,
priorities, depth or constraints, ask 2-4 focused questions (with copy-paste options) and
stop. If it is already fully specified, reply with exactly: {NO_CLARIFY}
If the message is not a scientific query (an introduction, a greeting, or a question about
you, the team or how this system works), reply with exactly: {META_QUERY}

User query:
"""

META_ANSWER = ("[Harness] The user's message above is an introduction or a meta question, not a research "
               "query. Answer it directly and briefly; no orientation, delegation or analysis is needed.")

BRIEF_PROMPT = ("Perform rapid due diligence for the user query below. Provide a structured intelligence "
                "brief covering: 1) Field Overview, 2) Data Landscape & Feasibility (inventory the tools "
                "and data available in this run with ListTools), 3) Recent Context (last 6-12 months), "
                "4) Key Considerations.\n\nUser query: {query}")

REVIEW_NUDGE = """[Harness — review policy]
You delegated analyses this turn ({agents}) but the scientific-reviewer has not evaluated
their outputs since they finished. Before your final synthesis, dispatch `scientific-reviewer`
with the user's question and each specialist's findings (numbers, files, limitations). If it
flags gaps or unsupported claims, re-delegate to the relevant specialists, then synthesize."""

PLAN_NUDGE = """[Harness — plan]
You dispatched several specialists this turn ({agents}) without a recorded plan. Before your
final synthesis, record the plan with mcp__provenance__write_plan (goal, steps, agents, real data
dependencies) as it was actually executed; deviations are recorded, not forbidden."""

SUPPORT_AGENTS = frozenset({"chief-of-staff", "scientific-reviewer"})
REVIEWER = "scientific-reviewer"
WRITE_PLAN_TOOL = "mcp__provenance__write_plan"
THINKING_EXCERPT = 2000


class SessionBusy(RuntimeError):
    """A turn is already running in this session."""

    def __init__(self, msg: str = "previous query still processing") -> None:
        super().__init__(msg)


def _iso(epoch: float | None) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat(timespec="seconds")


def _strip_sentinels(text: str) -> str:
    """Remove the orientation sentinels from model text."""
    out = (text or "").replace(NO_CLARIFY, "").replace(META_QUERY, "")
    return out.strip()


def _last_assistant_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "assistant" and m.text:
            return m.text
    return ""


class CSOSession:
    def __init__(self, runtime: Runtime, *, interface: str = "chat"):
        self.rt = runtime
        self.interface = interface
        self.history: list[Message] = []
        self.turn = 0
        self.oriented = not self.orchestration.get("strategic_orientation", True)
        self.pending_brief: str | None = None
        self._lock = asyncio.Lock()
        self._task: asyncio.Future | None = None
        self._cancel_requested = False
        self._ts: dict[str, Any] = {}       # per-turn state (review / plan / unstreamed)
        self.last_turn: dict[str, Any] | None = None

    # ------------------------------------------------------------------ properties

    @property
    def run(self):
        return self.rt.run

    @property
    def config(self) -> dict[str, Any]:
        return self.rt.config

    @property
    def orchestration(self) -> dict[str, Any]:
        return self.rt.config.get("orchestration") or {}

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    # ------------------------------------------------------------------ public API

    async def ask(self, user_input: str) -> str:
        """Process one user turn and return the CSO's reply.

        Raises :class:`SessionBusy` while another turn runs. The turn runs as a
        stored task so :meth:`cancel` can interrupt it; an interrupted turn is
        recorded (status ``interrupted``) and ``CancelledError`` propagates.
        """
        if self._lock.locked():
            raise SessionBusy()
        async with self._lock:
            self._cancel_requested = False
            task = asyncio.ensure_future(self._turn(user_input))
            self._task = task
            try:
                return await task
            finally:
                self._task = None

    def cancel(self) -> bool:
        """Cancel the running turn (it is recorded as interrupted). False when idle."""
        task = self._task
        if task is None or task.done():
            return False
        self._cancel_requested = True
        task.cancel()
        return True

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    async def close(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            await self.rt.cancel_outstanding()
        finally:
            try:
                self.run.close()
            finally:
                await self.rt.aclose()

    # ------------------------------------------------------------------ turn

    def _preflight_turn(self) -> None:
        """Per-turn readiness gate (raises DataReadinessError before any model call)."""
        pre = self.config.get("preflight") or {}
        if pre.get("skip"):
            return
        from .preflight import require_ready
        require_ready(self.config, per_turn=True, provider=self.rt.provider,
                      allow_missing_data=bool(pre.get("allow_missing_data")))

    async def _turn(self, user_input: str) -> str:
        rt, run = self.rt, self.run
        self.turn += 1
        n = self.turn
        reply = ""
        status = "in_progress"
        t0 = time.time()
        cost0 = run.cost.total_usd
        try:
            self._preflight_turn()
        except Exception as exc:  # noqa: BLE001 - DataReadinessError (or a broken check): do not send
            from .preflight import TURN_NOT_SENT, DataReadinessError
            msg = str(exc) if isinstance(exc, DataReadinessError) else \
                f"Readiness check failed: {type(exc).__name__}: {exc}. {TURN_NOT_SENT}"
            if TURN_NOT_SENT not in msg:
                msg = f"{msg} {TURN_NOT_SENT}"
            ev0 = len(run.events())
            self._ts = {"t0": t0, "unstreamed": [msg], "review_rounds": 0, "plan_nudged": False,
                        "review_skipped": False}
            run.trace("turn_start", turn=n, prompt=user_input, interface=self.interface)
            rt.emit("turn_start", turn=n, prompt=user_input)
            run.trace("turn_not_sent", turn=n, error=msg[:2000])
            self._finish(n, user_input, msg, "not_sent", t0, cost0, len(rt.delegation_log), ev0)
            return msg
        try:
            rt.repair_history(self.history, "the previous turn was interrupted before it finished")
        except Exception:  # noqa: BLE001 - a repair problem must not block the turn
            log.exception("history repair failed")
        hist0 = len(self.history)
        deleg0 = len(rt.delegation_log)
        ev0 = len(run.events())
        self._ts = {"t0": t0, "unstreamed": [], "review_rounds": 0, "plan_nudged": False, "review_skipped": False}
        run.trace("turn_start", turn=n, prompt=user_input, interface=self.interface)
        rt.emit("turn_start", turn=n, prompt=user_input)
        limits = self.config.get("limits") or {}
        turn_limit = limits.get("max_turn_cost_usd")
        turn_limit = float(turn_limit) if turn_limit not in (None, "", 0) else None
        try:
            with rt.cost_scope("turn", turn_limit):
                reply, cso_status = await self._dispatch(user_input)
            status = "completed" if cso_status == "completed" else f"incomplete: {cso_status}"
            if status != "completed":
                note = (f"[Turn incomplete: the CSO stopped ({cso_status}). "
                        "Evidence coverage may be incomplete.]")
                reply = (reply + "\n\n" if reply else "") + note
                self._ts["unstreamed"].append(note)
        except BaseException as exc:
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                status, detail = "interrupted", "cancelled by the user" if self._cancel_requested else \
                    f"{type(exc).__name__}"
            elif isinstance(exc, BudgetExceeded):
                status, detail = "budget_exceeded", f"budget exhausted ({exc})"
            else:
                status, detail = f"failed: {type(exc).__name__}: {exc}", f"{type(exc).__name__}: {exc}"
            partial = self._partial_text(hist0)
            notice = f"Turn interrupted: {detail}. Evidence coverage is incomplete."
            reply = (partial + "\n\n" if partial else "") + notice
            self._ts["unstreamed"].append(notice)
            try:
                rt.repair_history(self.history, f"the turn was interrupted ({detail})")
            except Exception:  # noqa: BLE001
                log.exception("history repair failed")
            self._finish(n, user_input, reply, status, t0, cost0, deleg0, ev0)
            raise
        self._finish(n, user_input, reply, status, t0, cost0, deleg0, ev0)
        return self.last_turn.get("response", reply) if self.last_turn else reply

    def _partial_text(self, hist0: int) -> str:
        parts = [_strip_sentinels(m.text) for m in self.history[hist0:] if m.role == "assistant" and m.text]
        return "\n\n".join(p for p in parts if p)

    async def _dispatch(self, user_input: str) -> tuple[str, str]:
        if not self.oriented:
            return await self._orientation(user_input)
        return await self._work(self._with_brief(user_input))

    # ------------------------------------------------------------------ records

    def _finish(self, n: int, prompt: str, reply: str, status: str, t0: float, cost0: float, deleg0: int,
                ev0: int) -> None:
        """Build and save the turn record, emit turn_end, save the session state. Never raises."""
        rt, run = self.rt, self.run
        rec: dict[str, Any] = {"turn": n, "prompt": prompt, "response": reply, "status": status}
        try:
            events = run.events()[ev0:]
        except Exception:  # noqa: BLE001
            events = []
        try:
            rec.update(self._turn_record(events, deleg0, t0, cost0))
        except Exception as exc:  # noqa: BLE001 - the record must be written anyway
            log.exception("building the turn record failed")
            rec["record_error"] = f"{type(exc).__name__}: {exc}"
        data_failures = rec.get("data_source_failures") or []
        notice = _failures.data_failure_notice(data_failures)
        if notice:
            reply = f"{reply}\n\n> {notice}" if reply else f"> {notice}"
            rec["response"] = reply
            self._ts["unstreamed"].append(notice)
        rec["ended"] = _iso(time.time())
        rec["duration_s"] = round(time.time() - t0, 1)
        self.last_turn = rec
        try:
            run.finish_turn(rec)
        except Exception:  # noqa: BLE001 - finish_turn guards itself; belt and braces
            log.exception("finish_turn failed")
        try:
            run.trace("turn_end", turn=n, status=status, cost_usd=rec.get("cost_usd"))
        except Exception:  # noqa: BLE001
            pass
        rt.emit("turn_end", turn=n, status=status, reply=reply, cost_usd=rec.get("cost_usd"),
                cumulative_usd=rec.get("cumulative_cost_usd"), unstreamed=list(self._ts.get("unstreamed") or []),
                run_dir=str(run.dir), agents=rec.get("agents") or [],
                data_source_failures=[f.get("tool") for f in data_failures])
        try:
            self.save_state()
        except Exception as exc:  # noqa: BLE001
            note = getattr(run, "note_audit_error", None)
            if callable(note):
                note(f"save_state: {type(exc).__name__}: {exc}")

    def _turn_record(self, events: list[dict[str, Any]], deleg0: int, t0: float, cost0: float) -> dict[str, Any]:
        rt, run = self.rt, self.run
        delegs = list(rt.delegation_log[deleg0:])
        records = _failures.records_from_trace(events, run_dir=run.dir)
        summary = _failures.summarize(records)
        data_failures = summary["unresolved_data"]
        thinking: list[dict[str, Any]] = []
        mcp_used: list[str] = []
        compactions = 0
        for ev in events:
            kind = ev.get("type")
            if kind == "model_call" and ev.get("agent") == "cso":
                for t in ev.get("thinking") or []:
                    if t:
                        thinking.append({"text": str(t)[:THINKING_EXCERPT], "chars": len(str(t))})
            elif kind == "tool_start":
                tool = str(ev.get("tool") or "")
                if tool.startswith("mcp__") and tool not in mcp_used:
                    mcp_used.append(tool)
            elif kind == "compaction":
                compactions += 1
        cost = run.cost.total_usd
        plan_written = self._plan_written(events)
        specialists = self._specialists(delegs)
        reviewed = bool(specialists) and not self._ts.get("review_skipped") and \
            not self._unreviewed(delegs, "always")
        return {
            "started": _iso(t0),
            "agents": list(dict.fromkeys(d["agent"] for d in delegs)),
            "delegations": [{"agent": d.get("agent"), "status": d.get("status"), "stop_reason": d.get("stop_reason"),
                             "cost_usd": d.get("cost_usd"), "duration_s": d.get("duration_s"),
                             "tool_use_id": d.get("tool_use_id"), "description": d.get("description")}
                            for d in delegs],
            "cost_usd": round(cost - cost0, 6),
            "cumulative_cost_usd": round(cost, 6),
            "tool_failures": summary["all"],
            "data_source_failures": data_failures,
            "recovered_errors": summary["recovered_count"],
            "other_errors": summary["other_error_count"],
            "reviewed": bool(reviewed),
            "review_rounds": int(self._ts.get("review_rounds") or 0),
            "review_skipped": bool(self._ts.get("review_skipped")),
            "plan_written": plan_written,
            "plan_missing": bool(len(set(specialists)) >= 2 and not plan_written),
            "plan_nudged": bool(self._ts.get("plan_nudged")),
            "compactions": compactions,
            "thinking_traces": thinking,
            "subagent_traces": [{"agent": d.get("agent"), "description": d.get("description"),
                                 "start": _iso(d.get("start_ts")), "end": _iso(d.get("end_ts")),
                                 "status": d.get("status"), "duration_s": d.get("duration_s"),
                                 "model_calls": d.get("model_calls"), "tool_calls": d.get("tool_calls"),
                                 "cost_usd": d.get("cost_usd"), "transcript_path": d.get("transcript_path")}
                                for d in delegs],
            "mcp_tools_used": mcp_used,
            "interface": self.interface,
        }

    def save_state(self) -> Path:
        """Persist the CSO conversation and session state to ``logs/cso_state.json``."""
        from .audit.storage import write_text_atomic

        path = self.run.dir / STATE_FILE
        state = {
            "version": 1, "saved": _iso(time.time()), "turn": self.turn, "oriented": self.oriented,
            "pending_brief": self.pending_brief, "interface": self.interface,
            "history": [message_to_dict(m) for m in self.history],
            "delegation_log": self.rt.delegation_log,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(state, default=str, ensure_ascii=False))
        return path

    def restore_state(self) -> bool:
        """Load ``logs/cso_state.json`` (if present) into this session; repair the history."""
        path = self.run.dir / STATE_FILE
        recorded = [t.get("turn") for t in getattr(self.run, "turns", []) or [] if isinstance(t, Mapping)]
        recorded_max = max([int(t) for t in recorded if isinstance(t, int)] or [0])
        state: dict[str, Any] = {}
        if path.is_file():
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("cannot read %s: %s", path, exc)
                state = {}
        history: list[Message] = []
        for d in state.get("history") or []:
            try:
                history.append(message_from_dict(d))
            except Exception as exc:  # noqa: BLE001 - skip a damaged message, repair fixes pairing
                log.warning("skipping unreadable history message: %s", exc)
        self.history = history
        self.turn = max(int(state.get("turn") or 0), recorded_max)
        if "oriented" in state:
            self.oriented = bool(state["oriented"]) or self.oriented
        elif recorded_max:
            self.oriented = True
        self.pending_brief = state.get("pending_brief")
        if isinstance(state.get("delegation_log"), list):
            self.rt.delegation_log[:] = [d for d in state["delegation_log"] if isinstance(d, dict)]
        fixes = self.rt.repair_history(self.history, "the session was resumed")
        self.run.trace("session_resumed", turn=self.turn, n_messages=len(self.history), state_found=bool(state),
                       history_fixes=fixes, oriented=self.oriented, interface=self.interface)
        return bool(state)

    # ------------------------------------------------------------------ phases

    async def _orientation(self, query: str) -> tuple[str, str]:
        rt = self.rt
        brief_task: asyncio.Future | None = None
        if "chief-of-staff" in rt.agents:
            brief_task = asyncio.ensure_future(rt.delegate(
                "chief-of-staff", BRIEF_PROMPT.format(query=query), description="strategic briefing",
                parent_agent="cso", depth=1))
            await asyncio.sleep(0)  # let the brief start, so it is on record even if cancelled at once
        settled = False
        try:
            # The clarification call runs without tools; it is part of the CSO's
            # persistent conversation so the next turn sees its own questions.
            clar = await rt.run_agent(rt.cso, ORIENTATION_INSTRUCTION + query, history=self.history,
                                      allow_tools=False, stream_text=False, description="clarification")
            text = clar.full_text or clar.text or ""
            if META_QUERY in text and NO_CLARIFY not in text:
                await self._cancel_brief(brief_task)
                settled = True
                rt.run.trace("orientation_deferred", reason="meta query")
                return await self._work(META_ANSWER)
            brief = await self._await_brief(brief_task)
            settled = True
        finally:
            if not settled:
                await self._cancel_brief(brief_task)
        self.oriented = True
        if brief:
            rt.run.trace("briefing", text=brief[:20000], chars=len(brief))
            rt.emit("briefing", text=brief)
            self._write_brief(brief)
        if NO_CLARIFY in text:
            note = f"[Chief of Staff briefing]\n{brief}\n\n" if brief else ""
            return await self._work(note + "Proceed with the analysis of the user query above.")
        self.pending_brief = brief or None
        questions = _strip_sentinels(text)
        if questions:
            self._ts["unstreamed"].append(questions)
        out = []
        if brief:
            out.append("### Chief of Staff briefing\n\n" + brief)
        out.append(questions)
        status = clar.status if clar.status else "completed"
        return "\n\n---\n\n".join(o for o in out if o), status

    @staticmethod
    async def _cancel_brief(task: asyncio.Future | None) -> None:
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _await_brief(self, task: asyncio.Future | None) -> str:
        if task is None:
            return ""
        try:
            res = await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the turn
            partial = getattr(exc, "agent_result", None)
            body = ((partial.text or partial.full_text) if partial is not None else "") or ""
            msg = f"[Chief of Staff briefing unavailable: {type(exc).__name__}: {exc}]"
            return f"{msg}\n\n{body}".strip() if body else msg
        text = (res.text or res.full_text or "").strip()
        if res.status != "completed":
            head = f"[Chief of Staff briefing incomplete: {res.status}" + (f" ({res.error})" if res.error else "") + "]"
            return f"{head}\n\n{text}".strip()
        return text or "[Chief of Staff briefing unavailable: no brief was written]"

    def _write_brief(self, brief: str) -> None:
        run = self.run
        writer = getattr(run, "_write_harness_text", None)
        try:
            if callable(writer):
                writer("report/chief_of_staff_brief.md", brief)
            else:
                (run.dir / "report" / "chief_of_staff_brief.md").write_text(brief)
        except Exception as exc:  # noqa: BLE001
            note = getattr(run, "note_audit_error", None)
            if callable(note):
                note(f"chief_of_staff_brief.md: {type(exc).__name__}: {exc}")

    def _with_brief(self, user_input: str) -> str:
        if self.pending_brief:
            text = (f"[Chief of Staff briefing, prepared during orientation]\n{self.pending_brief}\n\n"
                    f"[User reply]\n{user_input}")
            self.pending_brief = None
            return text
        return user_input

    # ------------------------------------------------------------------ review / plan policy

    def _is_support(self, agent: str) -> bool:
        if agent in SUPPORT_AGENTS:
            return True
        d = self.rt.agents.get(agent)
        return getattr(d, "tier", "") == "support"

    def _specialists(self, delegs: list[dict[str, Any]]) -> list[str]:
        return [d["agent"] for d in delegs if not self._is_support(d["agent"])]

    def _wrote_files(self, agent: str) -> bool:
        """The specialist wrote files under its own ``work/<agent>/`` during this turn."""
        try:
            from .audit.storage import work_owner
        except ImportError:  # pragma: no cover
            return False
        n = self.turn
        arts = (getattr(self.run, "manifest", None) or {}).get("artifacts") or {}
        for rel, e in list(arts.items()):
            if isinstance(e, Mapping) and work_owner(rel) == agent and n in (e.get("turn"), e.get("modified_turn")):
                return True
        # Fallback: a file newer than the turn start (capture may be disabled).
        root = self.run.dir / "work" / agent
        t0 = float(self._ts.get("t0") or 0) - 1.0
        try:
            for p in root.rglob("*"):
                if p.is_file() and p.stat().st_mtime >= t0:
                    return True
        except OSError:
            pass
        return False

    def _unreviewed(self, delegs: list[dict[str, Any]], policy: str | None = None) -> list[str]:
        """Specialists of this turn the policy requires reviewed but no reviewer saw."""
        from .agents import review_policy

        policy = policy or review_policy(self.config)
        if policy == "never":
            return []
        specs = [d for d in delegs if not self._is_support(d["agent"])]
        if not specs:
            return []
        distinct = set(d["agent"] for d in specs)
        if policy == "multi_specialist" and len(distinct) < 2:
            return []
        if policy == "research" and len(distinct) < 2 and not any(self._wrote_files(a) for a in distinct):
            return []
        # A reviewer dispatched in the same CSO message as the specialist cannot have
        # seen its output, whatever the clock says (instant mocks, queued delegations).
        batch = {c.id: i for i, m in enumerate(self.history) if m.role == "assistant" for c in m.tool_calls}
        reviews = [(float(d["start_ts"]), batch.get(d.get("tool_use_id"))) for d in delegs
                   if d["agent"] == REVIEWER and d.get("start_ts") is not None]
        out = []
        for d in specs:
            end = d.get("end_ts")
            mine = batch.get(d.get("tool_use_id"))
            if end is not None and any(s >= float(end) and (b is None or mine is None or b != mine)
                                       for s, b in reviews):
                continue
            out.append(d["agent"])
        return list(dict.fromkeys(out))

    def _plan_written(self, events: list[dict[str, Any]] | None = None) -> bool:
        if events is None:
            events = self.run.events()
            t0 = float(self._ts.get("t0") or 0)
            events = [e for e in events if float(e.get("t") or 0) >= t0]
        for ev in events:
            if ev.get("type") != "tool_end" or ev.get("tool") != WRITE_PLAN_TOOL or ev.get("is_error"):
                continue
            out = ev.get("output")
            try:
                data = json.loads(out) if isinstance(out, str) else out
            except ValueError:
                data = None
            if isinstance(data, Mapping) and data.get("ok") is False:
                continue
            return True
        return False

    async def _work(self, message: str) -> tuple[str, str]:
        from .agents import review_policy

        rt = self.rt
        cfg = self.orchestration
        policy = review_policy(self.config)
        enforce = policy != "never" and REVIEWER in rt.agents
        enforce_plan = bool(cfg.get("enforce_plan", False))
        max_rounds = int(cfg.get("max_review_rounds", 2) or 0)
        start = len(rt.delegation_log)
        ts = self._ts

        async def check(messages: list[Message]) -> str | None:
            final = _last_assistant_text(messages).rstrip()
            if final.endswith("?") or AWAITING_USER in final:
                return None
            log_ = rt.delegation_log[start:]
            specs = list(dict.fromkeys(self._specialists(log_)))
            if enforce_plan and not ts.get("plan_nudged") and len(specs) >= 2 and not self._plan_written():
                ts["plan_nudged"] = True
                rt.run.trace("plan_nudge", agents=specs)
                rt.emit("warning", message="[harness] no plan recorded for a multi-specialist turn: asking the CSO "
                                           "to record it")
                return PLAN_NUDGE.format(agents=", ".join(specs))
            if not enforce:
                return None
            unreviewed = self._unreviewed(log_, policy)
            if not unreviewed:
                return None
            if ts["review_rounds"] >= max_rounds:
                if not ts.get("review_skipped"):
                    ts["review_skipped"] = True
                    rt.run.trace("review_skipped", agents=unreviewed, rounds=ts["review_rounds"])
                    rt.emit("warning", message="[harness] review rounds exhausted; unreviewed: "
                                               + ", ".join(unreviewed))
                return None
            ts["review_rounds"] += 1
            rt.run.trace("review_enforced", agents=unreviewed, round=ts["review_rounds"])
            rt.emit("review_enforced", agents=unreviewed, round=ts["review_rounds"])
            return REVIEW_NUDGE.format(agents=", ".join(unreviewed))

        result = await rt.run_agent(rt.cso, message, history=self.history, stream_text=True, after_end_turn=check)
        text = _strip_sentinels(result.full_text or result.text or "")
        return text, result.status or "completed"


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


def _copy_environment_spec(run) -> None:
    from .config import PROJECT_ROOT

    src = PROJECT_ROOT / "environment.yml"
    if not src.is_file():
        return
    writer = getattr(run, "_write_harness_text", None)
    text = src.read_text(encoding="utf-8")
    if callable(writer):
        writer("inputs/environment.yml", text)
    else:
        (run.dir / "inputs" / "environment.yml").write_text(text)


async def open_session(config: dict[str, Any], *, provider=None, on_event=None, run_id: str | None = None,
                       start_mcp: bool = True, interface: str = "chat", profiles: tuple[str, ...] | list[str] = (),
                       resume: str | Path | None = None) -> CSOSession:
    """Create (or, with ``resume=<run dir>``, reopen) a run and its CSO session.

    The pinned configuration (``vbt.pinning``) is built after the MCP servers
    start and written to MANIFEST.config and inputs/config.json; the installed
    distribution list goes to inputs/environment.txt.
    """
    from .config import resolve_path
    from .pinning import build_pinned_config, installed_distributions
    from .session import Run

    profiles = tuple(profiles or config.get("profiles") or ())
    # Readiness gate before any run directory exists or any model call is made.
    pre = config.get("preflight") or {}
    allow_missing = bool(pre.get("allow_missing_data"))
    checks: list[Any] = []
    if not pre.get("skip"):
        from .preflight import require_ready
        checks = require_ready(config, provider=provider, allow_missing_data=allow_missing)
    if resume:
        run = Run.open_existing(resume)
    else:
        run = Run(resolve_path(config["paths"]["runs_dir"]), run_id=run_id, config={
            "provider": config["provider"]["name"], "models": config["models"],
            "web": config.get("web", {}), "orchestration": config.get("orchestration", {}),
            "audit": config.get("audit") or {},
        })
    rt = Runtime(config, run, provider=provider, on_event=on_event)
    try:
        degraded: dict[str, str] = {}
        if checks:
            from .preflight import degraded_servers
            degraded = degraded_servers(config, checks)
            failed = [c for c in checks if c.required and not c.ok and c.kind == "data"]
            if failed and not degraded:  # missing data not tied to one server (allow_missing_data)
                degraded = {"(reference data)": "; ".join(f"{c.label}: {c.detail}" for c in failed)[:500]}
        if degraded:
            rt.set_degraded(degraded)
            run.mark_degraded(degraded)
            rt.emit("warning", message="Running without reference data for: " + ", ".join(sorted(degraded)))
        if start_mcp:
            failures = await rt.start_mcp()
            if failures:
                rt.emit("warning", message=f"MCP servers unavailable: {', '.join(sorted(failures))}")
        try:
            pinned = build_pinned_config(config, rt, interface=interface, profiles=profiles)
        except Exception as exc:  # noqa: BLE001 - pinning must never block a session
            log.exception("building the pinned config failed")
            pinned = {"provider": {"name": config["provider"]["name"]}, "models": config.get("models"),
                      "pinning_error": f"{type(exc).__name__}: {exc}"}
        pinned["preflight"] = {"skipped": bool(pre.get("skip")) or not checks,
                               "allow_missing_data": allow_missing,
                               "checks": [{"label": c.label, "ok": c.ok, "kind": c.kind} for c in checks],
                               "degraded_servers": degraded}
        if resume:
            prev = dict(run.config or {}) if isinstance(run.config, Mapping) else {}
            resumes = list(prev.get("resumes") or [])
            resumes.append({"at": _iso(time.time()), "interface": interface,
                            "harness": pinned.get("harness"), "models": pinned.get("models"),
                            "provider": pinned.get("provider")})
            pinned = {**prev, "resumes": resumes} if prev else {**pinned, "resumes": resumes}
        run.set_config(pinned)
        if not resume:
            writer = getattr(run, "_write_harness_text", None)
            try:
                text = "\n".join(installed_distributions()) + "\n"
                if callable(writer):
                    writer("inputs/environment.txt", text)
                else:
                    (run.dir / "inputs" / "environment.txt").write_text(text)
                _copy_environment_spec(run)
            except Exception as exc:  # noqa: BLE001
                run.note_audit_error(f"environment record: {type(exc).__name__}: {exc}")
        session = CSOSession(rt, interface=interface)
        if resume:
            session.restore_state()
    except BaseException:
        await rt.aclose()
        raise
    return session
