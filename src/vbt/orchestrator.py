"""CSO session: the multi-turn Virtual Biotech workflow (Fig. 1C).

Turn 1 (strategic orientation, if enabled)
    The Chief of Staff prepares a briefing *in parallel* with the CSO's
    clarification interview. If the CSO has questions, the turn ends with the
    briefing summary and the questions; otherwise it proceeds straight to work.
Every research turn
    The CSO decomposes the query and delegates with ``Task`` (parallel where
    independent). Before the CSO's synthesis is accepted, the harness checks
    that the Scientific Reviewer evaluated this turn's specialist outputs and
    re-opens the loop if not (bounded by ``max_review_rounds``). The reviewer's
    critique flows back to the CSO, which may re-delegate for refinement.
Follow-ups
    The CSO conversation is persistent (append-only), so follow-up questions
    build on earlier turns, while each delegation starts a fresh specialist.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .providers.base import Message
from .runtime import Runtime

ORIENTATION_INSTRUCTION = """[Harness — strategic orientation]
This is the first scientific query of the session. Your Chief of Staff is preparing an
intelligence briefing in parallel (field overview, data landscape, recent developments),
which you will receive before any analysis starts, so do not request one yourself.

Right now, conduct your clarification interview: if the query is ambiguous in scope,
priorities, depth or constraints, ask 2-4 focused questions (with copy-paste options) and
stop. If it is already fully specified, reply with exactly: NO_CLARIFICATION_NEEDED

User query:
"""

NO_CLARIFY = "NO_CLARIFICATION_NEEDED"

REVIEW_NUDGE = """[Harness — review policy]
You delegated analyses this turn ({agents}) but the scientific-reviewer has not evaluated
their outputs since. Before your final synthesis, dispatch `scientific-reviewer` with the
user's question and each specialist's findings (numbers, files, limitations). If it flags
gaps or unsupported claims, re-delegate to the relevant specialists, then synthesize."""


class CSOSession:
    def __init__(self, runtime: Runtime):
        self.rt = runtime
        self.history: list[Message] = []
        self.turn = 0
        self.oriented = not runtime.config.get("orchestration", {}).get("strategic_orientation", True)
        self.pending_brief: str | None = None

    @property
    def run(self):
        return self.rt.run

    async def ask(self, user_input: str) -> str:
        """Process one user turn and return the CSO's reply."""
        self.turn += 1
        self.rt.begin_turn()
        t0, cost0 = time.time(), self.run.cost.total_usd
        n_deleg0 = len(self.rt.delegation_log)
        if self.turn == 1:
            (self.run.dir / "inputs" / "query.txt").write_text(user_input)
        self.run.trace("turn_start", turn=self.turn, prompt=user_input)
        status = "completed"
        try:
            if not self.oriented:
                self.oriented = True
                reply = await self._orientation(user_input)
            else:
                reply = await self._work(self._with_brief(user_input))
        except Exception as exc:
            status = f"failed: {type(exc).__name__}: {exc}"
            reply = f"[Turn failed: {type(exc).__name__}: {exc}]"
            raise
        finally:
            delegs = self.rt.delegation_log[n_deleg0:]
            failures = [e for d in delegs for e in d["tool_errors"]]
            if failures and status == "completed":
                names = ", ".join(dict.fromkeys(f["tool"] for f in failures))
                reply += (f"\n\n> Data/evidence warning: these tools failed during this turn: {names}. "
                          "Their results cannot support this answer.")
            self.run.finish_turn({
                "turn": self.turn, "prompt": user_input, "response": reply, "status": status,
                "agents": list(dict.fromkeys(d["agent"] for d in delegs)),
                "cost_usd": round(self.run.cost.total_usd - cost0, 6),
                "cumulative_cost_usd": round(self.run.cost.total_usd, 6),
                "duration_s": round(time.time() - t0, 1), "tool_failures": failures,
            })
            self.run.trace("turn_end", turn=self.turn, status=status)
        return reply

    # ------------------------------------------------------------------ phases

    async def _orientation(self, query: str) -> str:
        cos = self.rt.agents.get("chief-of-staff")
        brief_task = None
        if cos is not None:
            brief_task = asyncio.create_task(self.rt.run_agent(
                cos,
                "Perform rapid due diligence for the user query below. Provide a structured intelligence "
                "brief covering: 1) Field Overview, 2) Data Landscape & Feasibility (inventory the tools "
                "and data available in this run with ListTools), 3) Recent Context (last 6-12 months), "
                f"4) Key Considerations.\n\nUser query: {query}",
                depth=1))
        # The clarification call runs without tools; it is part of the CSO's
        # persistent conversation so the next turn sees its own questions.
        clar = await self.rt.run_agent(self.rt.cso, ORIENTATION_INSTRUCTION + query,
                                       history=self.history, allow_tools=False, stream_text=False)
        brief = (await brief_task).text if brief_task else ""
        if brief:
            self.run.trace("briefing", text=brief)
            (self.run.dir / "report" / "chief_of_staff_brief.md").write_text(brief)

        if NO_CLARIFY in clar.text:
            note = f"[Chief of Staff briefing]\n{brief}\n\n" if brief else ""
            return await self._work(note + "Proceed with the analysis of the user query above.")
        self.pending_brief = brief
        out = []
        if brief:
            out.append("### Chief of Staff briefing\n\n" + brief)
        out.append(clar.text)
        return "\n\n---\n\n".join(out)

    def _with_brief(self, user_input: str) -> str:
        if self.pending_brief:
            text = (f"[Chief of Staff briefing, prepared during orientation]\n{self.pending_brief}\n\n"
                    f"[User reply]\n{user_input}")
            self.pending_brief = None
            return text
        return user_input

    async def _work(self, message: str) -> str:
        cfg = self.rt.config.get("orchestration", {})
        enforce = cfg.get("enforce_review", True) and "scientific-reviewer" in self.rt.agents
        max_rounds = int(cfg.get("max_review_rounds", 2))
        start = len(self.rt.delegation_log)
        nudges = 0

        async def check_review(_messages: list[Message]) -> str | None:
            nonlocal nudges
            if not enforce or nudges >= max_rounds:
                return None
            log = self.rt.delegation_log[start:]
            last_review = max((i for i, d in enumerate(log) if d["agent"] == "scientific-reviewer"), default=-1)
            unreviewed = [d["agent"] for d in log[last_review + 1:]
                          if d["agent"] not in ("scientific-reviewer", "chief-of-staff")]
            if not unreviewed:
                return None
            nudges += 1
            self.run.trace("review_enforced", agents=unreviewed, round=nudges)
            return REVIEW_NUDGE.format(agents=", ".join(dict.fromkeys(unreviewed)))

        result = await self.rt.run_agent(self.rt.cso, message, history=self.history, stream_text=True,
                                         after_end_turn=check_review)
        return result.text

    async def close(self) -> None:
        self.run.close()
        await self.rt.aclose()


async def open_session(config: dict[str, Any], *, provider=None, on_event=None, run_id: str | None = None,
                       start_mcp: bool = True) -> CSOSession:
    from .config import resolve_path
    from .session import Run

    run = Run(resolve_path(config["paths"]["runs_dir"]), run_id=run_id, config={
        "provider": config["provider"]["name"], "models": config["models"],
        "web": config.get("web", {}), "orchestration": config.get("orchestration", {}),
    })
    rt = Runtime(config, run, provider=provider, on_event=on_event)
    if start_mcp:
        failures = await rt.start_mcp()
        if failures:
            rt.emit("warning", message=f"MCP servers unavailable: {', '.join(sorted(failures))}")
    return CSOSession(rt)
