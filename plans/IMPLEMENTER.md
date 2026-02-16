# Implementer Guide

Allowed tools/commands
1. rg, uv run pytest, python -m buzzllm.subagent_runner (local), git status, git diff.

Repo conventions
1. Keep changes minimal and localized.
2. Keep tool names in a single catalog module.
3. Do not add new dependencies.

Do not list
1. Do not refactor unrelated code.
2. Do not rename widely used symbols.
3. Do not add try/except unless explicitly requested.
4. Do not change CLI behavior beyond adding call_subagent tool availability.

Milestones
1. M1: Add tool catalog + helper builder; tests for catalog pass.
2. M2: Add subagent runner + SSE parser; unit tests pass.
3. M3: Add call_subagent tool + main wiring; integration tests pass.
4. M4: Full test run passes; update devlogs.md.

Devlog + checkpoints
1. Update devlogs.md every iteration and after each verifier run.
2. Commit checkpoint after each milestone with message: ralph(iter N): <milestone>.
3. If stuck for 3 iterations, propose/perform rollback and log it.

## OpenAI Responses Implementer Milestones

Allowed tools/commands
1. rg, uv run pytest, python -m buzzllm.main (local), git status, git diff.

Do not list
1. Do not change openai-chat or anthropic behavior.
2. Do not weaken or delete tests; update expectations instead.
3. Do not add new dependencies.

Milestones
1. M1: Update make_openai_responses_request_args to build input list + tools (file: src/buzzllm/llm.py; tests: tests/unit/test_llm_request_args.py).
2. M2: Parse tool call events in handle_openai_responses_stream_response (file: src/buzzllm/llm.py; tests: tests/unit/test_llm_stream_handlers.py).
3. M3: Implement tool_call_response_to_openai_responses_messages and invoke_llm input routing (file: src/buzzllm/llm.py; tests: tests/unit/test_llm_tool_messages.py).
4. M4: Add integration test for responses API and run full pytest (file: tests/integration/test_openai_responses_integration.py).

Devlog + checkpoints
1. Update devlogs.md every iteration and after each verifier run.
2. Commit checkpoint after each milestone with message: ralph(iter N): <milestone>.
3. If stuck for 3 iterations, propose/perform rollback and log it.

## Structured Outputs Implementer Notes

1. Do not modify system prompts for structured output.
2. When structured output is enabled, buffer output_text and emit output_structured once.
3. Non-SSE should print JSON once, then DONE marker.
