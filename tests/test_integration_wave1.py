"""Cross-package contracts after merging the provider (P2), audit (P4) and
roster/MCP (P6) packages into the current runtime."""

from vbt.orchestrator import open_session
from vbt.providers.base import ImagePart, ProviderCapabilities, TextBlock, Usage
from vbt.providers.mock import ScriptedProvider, call, reply
from vbt.session import CostLedger
from vbt.tools.base import Tool, schema


def _plot_tool():
    # Shape returned by MCPBridge._convert for an MCP result with an image.
    async def handler(ctx, a):
        return [TextBlock("UMAP of 1,200 cells"), ImagePart("image/png", "iVBORw0KGgo=", "mcp__single_cell__umap")]
    return Tool("mcp__single_cell__umap", "plot", schema({}), handler, source="mcp:single_cell")


async def _run_plot(config, provider):
    seen = []

    def after(messages):
        seen.append(messages[-1].content[0])
        return reply("done")

    provider.script = ScriptedProvider.from_rules(
        {"single-cell-analyst": [reply(call("mcp__single_cell__umap")), after]}).script
    config["orchestration"]["enforce_review"] = False
    session = await open_session(config, provider=provider, start_mcp=False)
    agent = session.rt.agents["single-cell-analyst"]
    await session.rt.run_agent(agent, "plot it", depth=1, extra_tools=[_plot_tool()])
    ends = [e for e in session.run.events() if e["type"] == "tool_end" and e["tool"] == "mcp__single_cell__umap"]
    await session.close()
    return seen[0], ends[0]


async def test_mcp_image_parts_reach_a_vision_model(config):
    result, end = await _run_plot(config, ScriptedProvider(lambda *a: reply("x")))
    assert isinstance(result.content, list)
    assert [type(p) for p in result.content] == [TextBlock, ImagePart]
    assert "iVBORw0KGgo" not in end["output"] and "[image: mcp__single_cell__umap]" in end["output"]


async def test_mcp_image_parts_flatten_without_vision(config):
    provider = ScriptedProvider(lambda *a: reply("x"), capabilities=ProviderCapabilities())
    result, _ = await _run_plot(config, provider)
    assert isinstance(result.content, str)
    assert "[image: mcp__single_cell__umap]" in result.content and "iVBORw0KGgo" not in result.content


def test_cost_ledger_round_trips_new_usage_fields():
    ledger = CostLedger()
    ledger.add("cso", Usage(10, 5, 3, 4, cache_write_1h_tokens=2, server_tool_requests={"web_search": 2}), 0.5)
    back = CostLedger.from_report(ledger.report())
    assert back.by_agent["cso"] == ledger.by_agent["cso"]
    assert back.report() == ledger.report()
