"""Tool abstraction shared by built-in tools, MCP-bridged tools and delegation."""

from __future__ import annotations

import fnmatch
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union

from ..providers.base import ToolSpec

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import Runtime
    from ..session import Run


class ToolFailure(Exception):
    """Raise inside a handler to return an error result to the model."""


@dataclass
class ToolContext:
    """Everything a tool may need about the calling agent and the run."""

    agent: str
    run: "Run"
    runtime: "Runtime"
    tool_call_id: str = ""
    depth: int = 0

    @property
    def workspace(self) -> Path:
        return self.run.agent_dir(self.agent)


Handler = Callable[[ToolContext, dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    source: str = "builtin"  # builtin | mcp:<server> | provenance | delegation
    tags: set[str] = field(default_factory=set)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    async def __call__(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        r = self.handler(ctx, args)
        if inspect.isawaitable(r):
            r = await r
        return r


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs``/``$ref`` so the schema works with every provider's tool API."""
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].split("/")[-1] in defs:
                target = walk(defs[node["$ref"].split("/")[-1]])
                return {**target, **{k: walk(v) for k, v in node.items() if k != "$ref"}}
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=1, default=str)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def extend(self, tools: list[Tool]) -> None:
        for t in tools:
            self.add(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def select(self, patterns: list[str]) -> list[Tool]:
        """Resolve an allowlist of exact names or glob patterns (``mcp__genetics__*``)."""
        chosen: dict[str, Tool] = {}
        for pat in patterns:
            for name in sorted(fnmatch.filter(self._tools, pat)):
                chosen[name] = self._tools[name]
        return list(chosen.values())

    def missing(self, patterns: list[str]) -> list[str]:
        return [p for p in patterns if not fnmatch.filter(self._tools, p)]
