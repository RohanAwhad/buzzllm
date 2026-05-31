# Phase 2: Provider System

## Goal

Implement all 4 LLM providers as pure functions: request builders, SSE line parsers, and tool-response message assemblers. No networking yet — this phase is fully unit-testable with captured SSE samples.

## Source reference

- `src/buzzllm/llm.py:204-236` — OpenAI Chat request builder
- `src/buzzllm/llm.py:239-309` — OpenAI Chat SSE parser
- `src/buzzllm/llm.py:312-335` — OpenAI Chat tool message assembler
- `src/buzzllm/llm.py:343-369` — Anthropic request builder
- `src/buzzllm/llm.py:372-438` — Anthropic SSE parser
- `src/buzzllm/llm.py:441-468` — Anthropic tool message assembler
- `src/buzzllm/llm.py:476-502` — OpenAI Responses request builder
- `src/buzzllm/llm.py:505-538` — OpenAI Responses SSE parser
- `src/buzzllm/llm.py:554-576` — VertexAI Anthropic request builder

## Deliverables

### File structure

```
rust/src/providers/
  mod.rs                  Provider enum + dispatch
  openai_chat.rs
  openai_responses.rs
  anthropic.rs
  vertexai_anthropic.rs
```

### Provider trait (or function triple)

Each provider must supply 3 functions. Use an enum with methods rather than a trait (simpler, no dyn dispatch needed):

```rust
pub enum Provider {
    OpenaiChat,
    OpenaiResponses,
    Anthropic,
    VertexaiAnthropic,
}

impl Provider {
    pub fn make_request_args(&self, opts: &LlmOptions, prompt: &str, system_prompt: &str) -> RequestArgs;
    pub fn parse_sse_line(&self, line: &str, message_started: bool) -> Vec<StreamResponse>;
    pub fn assemble_tool_messages(&self, messages: &mut Vec<Value>, tool_calls: &HashMap<String, ToolCall>);
}
```

### Provider-specific details to preserve

#### OpenAI Chat
- Reasoning models list: `gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini, o4-mini, o3, o3-pro, gpt-5-pro` + any model starting with `gpt-5`
- For reasoning models: system role becomes `developer`, add `response_format: {type: "text"}`, add `reasoning_effort` field
- `gpt-5.1` without `--think`: `reasoning_effort: "none"`, all others: `"high"`
- Non-reasoning models get `temperature` and `max_tokens`
- Auth header: `Authorization: Bearer <key>`
- SSE format: `data: {json}\n` lines, terminated by `data: [DONE]`
- Tool calls arrive incrementally: first chunk has `id` + `function.name`, subsequent chunks append to `function.arguments`
- Tracks `current_tool_call_id` across chunks

#### Anthropic
- System prompt goes in top-level `system` field (not in messages)
- Think mode: `max_tokens: 32000`, `thinking: {type: "enabled", budget_tokens: 24000}`
- Auth headers: `x-api-key` + `anthropic-version: 2023-06-01`
- SSE format: `event:` lines (ignored) + `data:` lines with `type` field
- Tool calls: `content_block_start` with `type: "tool_use"` gives id+name, `content_block_delta` with `partial_json` appends arguments
- Tool results: assistant message has `content: [{type: "tool_use", ...}]`, user message has `content: [{type: "tool_result", ...}]`

#### OpenAI Responses
- Uses `input` instead of `messages`, `instructions` instead of system message
- Always sends `reasoning: {effort: "high", summary: "detailed"}` and `store: false`
- Tools: `NotImplementedError` in Python — port this as returning an error
- SSE events: `response.created`, `response.output_text.delta`, `response.reasoning_summary_text.delta`, `response.completed`

#### VertexAI Anthropic
- Same SSE parser and message assembler as Anthropic
- Different request builder: adds `anthropic_version: "vertex-2023-10-16"`
- Auth: calls `gcloud auth print-access-token` via subprocess
- Uses `Authorization: Bearer <token>` (not `x-api-key`)

### Global state migration

Python uses module-level `TOOL_CALLS: dict` and `current_tool_call_id: str`. In Rust:
- Pass `&mut HashMap<String, ToolCall>` and `&mut String` (current_tool_call_id) into the SSE parser
- No global mutable state

## Verification

For each provider, create unit tests with real captured SSE lines:

1. **Request builder test**: call `make_request_args()`, assert JSON body structure, headers, and conditional fields (think mode, reasoning models, tools)
2. **SSE parser test**: feed captured lines from a real API response, assert correct sequence of `StreamResponse` types and deltas
3. **Tool call accumulation test**: feed multi-chunk tool call SSE lines, assert `ToolCall` is assembled correctly with complete name + arguments
4. **Message assembler test**: given a populated `tool_calls` map, assert output messages match expected provider format

Capture real SSE samples by running the Python version with `--sse` and saving output to `rust/tests/fixtures/`.
