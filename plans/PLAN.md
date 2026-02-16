# Plan: Subagent Tool Runner (SSE subprocess)

1. Feature name: Subagent Tool Runner (SSE subprocess)
2. Objective: Allow main agent to spawn subagents with tool subsets using the same model/provider, returning only concatenated output_text.
3. Context/problem: current system is single-agent; tool wiring is tied to system prompts; globals make in-process nesting unsafe.
4. Non-goals: openai-responses tool support, streaming subagent output to user, persistent subagent state, new UI/CLI flags.
5. Success criteria: call_subagent tool available when tools are enabled; tool_subset allowlist enforced; output is output_text only; tests pass; no regression in existing flows.
6. Constraints: no new deps; no new try/except unless explicitly requested; ASCII only; minimal diff; subprocess uses SSE.

Links
1. ARCH.md
2. FLOWS.md
3. API.md
4. DATA.md
5. TESTPLAN.md
6. VERIFIER.md
7. IMPLEMENTER.md
8. DEVLOG_CONTRACT.md
9. RISKS.md
10. CHECKLIST.md
11. FEEDBACK.md

Milestones
1. M1: Tool catalog + schema builder for named tool subsets (files: src/buzzllm/tools/catalog.py, src/buzzllm/tools/utils.py).
2. M2: Subagent runner reads JSON from stdin, runs invoke_llm SSE (files: src/buzzllm/subagent_runner.py).
3. M3: call_subagent tool + wiring in main tool set (files: src/buzzllm/subagent.py, src/buzzllm/main.py).
4. M4: Tests + fixtures (files: tests/unit/test_subagent_*.py, tests/integration/test_subagent_runner.py).

Acceptance
1. call_subagent returns concatenated output_text string.
2. Subagent tool_subset is enforced; unknown tool names return an error string.
3. SSE parser ignores tool_call/tool_result/reasoning_content events.
4. uv run pytest passes.

# Plan: OpenAI Responses Tooling + Streaming Parity

1. Feature name: OpenAI Responses Tooling + Streaming Parity
2. Objective: Enable tools and multi-turn tool loop for openai-responses with streaming parity to openai-chat.
3. Context/problem: openai-responses request args ignore tools, tool loop is messages-only, and stream handler does not parse tool calls.
4. Non-goals: new CLI flags, new providers, refactoring beyond input/messages routing, new dependencies.
5. Success criteria: tools execute end-to-end via openai-responses; tool calls parsed and emitted; tool results appended via previous_response_id; unit tests pass; integration test passes when API key is present.
6. Constraints: no new deps; minimal diff; no new try/except; ASCII only; keep openai-chat/anthropic behavior unchanged; keep requests streaming.

Links
1. ARCH.md
2. FLOWS.md
3. API.md
4. DATA.md
5. TESTPLAN.md
6. VERIFIER.md
7. IMPLEMENTER.md
8. DEVLOG_CONTRACT.md
9. RISKS.md
10. CHECKLIST.md
11. FEEDBACK.md

Milestones
1. M1: Request args + input contract updated for openai-responses (files: src/buzzllm/llm.py, tests/unit/test_llm_request_args.py).
2. M2: Stream handler parses tool-call events + reasoning summary (files: src/buzzllm/llm.py, tests/unit/test_llm_stream_handlers.py).
3. M3: Tool response wiring + input/messages routing in invoke_llm (files: src/buzzllm/llm.py, tests/unit/test_llm_tool_messages.py).
4. M4: Integration test (skipped without API key) + full pytest (files: tests/integration/test_openai_responses_integration.py).

Acceptance
1. openai-responses can call tools and continue via previous_response_id.
2. Streaming emits output_text and tool_call events for responses API.
3. Request args honor think/temperature/max_tokens for responses.
4. uv run pytest passes; integration test skipped without OPENAI_API_KEY.

# Plan: Structured Outputs (OpenAI response_format + Anthropic output_config)

1. Feature name: Structured Outputs (OpenAI response_format + Anthropic output_config)
2. Objective: Add JSON-structured output via provider-native API parameters with no system prompt changes.
3. Context/problem: output is free-form text; no schema validation or JSON-only response mode.
4. Non-goals: response repair loop, new CLI flags, changes to openai-responses flow, prompt edits.
5. Success criteria: openai-chat uses response_format; anthropic/vertex uses output_config.format; output_structured event emitted; JSON printed once in non-SSE; tests pass.
6. Constraints: no new deps; no new try/except; ASCII only; keep streaming logic intact.

Links
1. ARCH.md
2. FLOWS.md
3. API.md
4. DATA.md
5. TESTPLAN.md
6. VERIFIER.md
7. IMPLEMENTER.md
8. DEVLOG_CONTRACT.md
9. RISKS.md
10. CHECKLIST.md
11. FEEDBACK.md

Milestones
1. M1: Add structured output config to LLMOptions and request args (file: src/buzzllm/llm.py; tests: tests/unit/test_llm_request_args.py).
2. M2: Collect output_text into buffer and emit output_structured at end (file: src/buzzllm/llm.py; tests: tests/unit/test_llm_stream_handlers.py).
3. M3: Add structured-output unit tests (files: tests/unit/test_llm_structured_output.py).

Acceptance
1. Structured output uses response_format (OpenAI) and output_config.format (Anthropic/Vertex).
2. No system prompt changes when structured output is enabled.
3. SSE includes output_structured; non-SSE prints JSON once at end.
4. uv run pytest passes.
