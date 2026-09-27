# Swapping the model provider

Only `src/vbt/providers/` knows about a specific LLM API. Agents, tools, MCP
servers, orchestration, bulk runs and case studies use the neutral types in
`providers/base.py`:

- `Message(role, content=[TextBlock | ThinkingBlock | ToolCall | ToolResult | OpaqueBlock])`
- `ToolSpec(name, description, input_schema)` — JSON Schema, passed through from MCP
- `ModelSettings(provider, model, max_tokens, effort, thinking, temperature, extra)`
- `ModelResponse(message, stop_reason, usage, model, cost_usd)`

## Add a provider

1. Subclass `LLMProvider` and implement `complete(settings, system, messages, tools, on_text)`:
   - encode neutral messages to the vendor format (tool calls ↔ function calls,
     tool results ↔ tool/function messages);
   - map the vendor's stop reason to `StopReason` (`tool_use` whenever the reply
     contains tool calls);
   - return `Usage` and a USD cost from your price table.
   - Keep vendor-only items (reasoning signatures, server-tool blocks) in
     `native` / `OpaqueBlock` so they replay unchanged to the same provider.
2. Optionally implement `web_search(query)` (used by the `WebSearch` tool) and
   return `supports_web_search() -> True`.
3. Register it in `providers/__init__.py`: `register_provider("myvendor", factory)`.
4. Create a profile, e.g. `configs/profiles/myvendor.yaml`:

```yaml
provider:
  name: myvendor
  options: {}
models:
  orchestrator: {model: <id>, effort: null, max_tokens: 32000, thinking: true}
  scientist:    {model: <id>, effort: null, max_tokens: 32000, thinking: true}
  support:      {model: <cheaper id>, effort: null, max_tokens: 16000, thinking: false}
  bulk:         {model: <id>, effort: null, max_tokens: 16000, thinking: false}
```

5. Run `vbt --profile myvendor doctor --smoke` and the test suite; the
   scripted `mock` provider in `providers/mock.py` shows the minimal contract.

Requirements on the model: reliable parallel tool calling and long contexts
(the CSO accumulates a multi-turn conversation; scientists read large tool
outputs, which the harness truncates to `limits.tool_output_max_chars`).
