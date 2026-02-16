# BuzzLLM Development Log

## 2026-01-01: Test Suite Implementation

### Goal
Add comprehensive test suite: unit tests, integration tests (OpenAI + Anthropic), and e2e CLI tests.

### Setup
- Added test dependencies to pyproject.toml: pytest, pytest-asyncio, pytest-cov, pytest-mock, respx, responses
- Configured pytest markers: unit, integration, e2e, docker

### Decisions
- Websearch tests: Mock only (no real API hits)
- Docker tests: Skip if Docker unavailable
- Integration tests: Skip if API keys not set
- Global state (TOOL_CALLS, AVAILABLE_TOOLS): Reset via autouse fixture

### Progress
- [x] pyproject.toml updated
- [x] tests/conftest.py
- [x] tests/unit/ (9 files)
- [x] tests/integration/ (2 files)
- [x] tests/e2e/test_cli.py

### Final Results
```
167 passed, 3 skipped, 2 warnings in 30.27s
```
- 148 unit tests pass (3 Docker tests skipped when Docker unavailable)
- 7 integration tests pass
- 12 e2e tests pass

### Issues Fixed During Implementation
1. **CWD patching in codesearch tests**: CWD captured at import time, monkeypatch didn't work. Fixed by using actual project directory instead of temp paths.
2. **RetryError wrapping in websearch**: `_search_brave` has @retry decorator, changed to `pytest.raises(RetryError)`.
3. **Docker skip detection**: subprocess.run(["docker", "info"]) unreliable. Fixed using Docker SDK's `client.ping()`.
4. **Anthropic model names**: `claude-haiku-4-5-20251101` doesn't exist. Changed to `claude-3-5-haiku-20241022`.
5. **Thinking mode support**: Haiku doesn't support extended thinking. Changed thinking tests to use `claude-sonnet-4-20250514`.
6. **OpenAI reasoning model**: `gpt-5.1-nano` not in OPENAI_REASONING_MODELS. Changed to `gpt-5.1`.

### Models Used
- OpenAI: `gpt-4.1-mini` (standard), `gpt-5.1` (reasoning)
- Anthropic: `claude-3-5-haiku-20241022` (standard), `claude-sonnet-4-20250514` (thinking)

## 2026-01-01: Brief Mode Flag

### Goal
Add `--brief/-b` flag to hide intermediate tool calls/results, showing only final LLM output.

### Changes
- `main.py`: Added `--brief/-b` argument, passed through `chat()` to `invoke_llm()`
- `llm.py`: Updated `invoke_llm()` and `print_to_stdout()` to skip `tool_call` and `tool_result` types when brief=True

### Use Case
Useful when using buzzllm as a Claude Code skill - prevents flooding context with intermediate steps.

## 2026-02-15: Iter 1 M1 Tool Catalog

Timestamp: 2026-02-15
Goal: Add tool catalog + helper for tool subset schemas.
Changes: Added tool catalog mapping and schema builder; added unit tests for tool subset validation.
Commands+Results: (none)
Decisions: Keep tool catalog as a single mapping with per-tool descriptions.
Next: Implement subagent runner + SSE parser.
Checkpoint: 82170a5

## 2026-02-15: Iter 1 M2 Subagent Runner

Timestamp: 2026-02-15
Goal: Add subagent runner + SSE output parser.
Changes: Added subagent runner entrypoint and SSE output_text parser; added unit test for mixed SSE events.
Commands+Results: (none)
Decisions: Keep SSE parsing minimal and output_text-only for subagent output.
Next: Add call_subagent tool and main wiring.
Checkpoint: a2db6f5

## 2026-02-15: Iter 1 M3 call_subagent Tool

Timestamp: 2026-02-15
Goal: Add call_subagent tool and wire into main tool list.
Changes: Added subagent context/config + payload builder, call_subagent execution, and integration/unit tests.
Commands+Results: (none)
Decisions: Validate tool_subset against catalog before spawning subprocess.
Next: Implement openai-responses tooling updates.
Checkpoint: 6587e1f
