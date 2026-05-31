# Phase 9: Testing Strategy

## Goal

Port the Python test suite (~90 tests) to Rust and establish the test infrastructure. Mirror the existing 3-tier structure: unit tests (fast, no external deps), integration tests (need API keys), and e2e tests (run the compiled binary).

## Python test inventory

The Python suite has:
- **38 unit tests** across 9 files — parsers, schema gen, dataclass behavior, CLI args, pagination, path validation
- **7 integration tests** across 2 files — real OpenAI/Anthropic API calls with streaming
- **12 e2e tests** across 1 file — subprocess CLI invocations
- **Shared fixtures** in `conftest.py` — mock HTML/JSON, fake API keys, temp CWD, global state reset

## Rust test structure

```
rust/
  src/
    ...                         # inline #[cfg(test)] mod tests for unit tests
  tests/
    fixtures/
      openai_chat_sse.txt       # captured SSE lines from real OpenAI response
      openai_chat_tool_sse.txt  # SSE with tool calls
      anthropic_sse.txt         # captured SSE from Anthropic
      anthropic_tool_sse.txt    # SSE with tool use blocks
      openai_responses_sse.txt  # captured SSE from Responses API
      duckduckgo_lite.html      # saved DDG HTML for parsing tests
      brave_search.json         # saved Brave API response
    integration/
      mod.rs
      openai_test.rs
      anthropic_test.rs
    e2e/
      mod.rs
      cli_test.rs
```

### Fixture capture

Before writing tests, capture real SSE samples from the Python version:

```bash
# OpenAI chat — plain text
buzzllm "gpt-4o-mini" "https://api.openai.com/v1/chat/completions" \
    "say hello" --provider openai-chat --api-key-name OPENAI_API_KEY -S \
    2>/dev/null > rust/tests/fixtures/openai_chat_sse.txt

# OpenAI chat — with tool call
buzzllm "gpt-4o-mini" "https://api.openai.com/v1/chat/completions" \
    "search the web for rust programming" --provider openai-chat \
    --api-key-name OPENAI_API_KEY --system-prompt websearch -S \
    2>/dev/null > rust/tests/fixtures/openai_chat_tool_sse.txt

# Anthropic — plain text
buzzllm "claude-3-5-haiku-20241022" "https://api.anthropic.com/v1/messages" \
    "say hello" --provider anthropic --api-key-name ANTHROPIC_API_KEY -S \
    2>/dev/null > rust/tests/fixtures/anthropic_sse.txt

# Anthropic — with tool call
buzzllm "claude-3-5-haiku-20241022" "https://api.anthropic.com/v1/messages" \
    "search the web for rust" --provider anthropic \
    --api-key-name ANTHROPIC_API_KEY --system-prompt websearch -S \
    2>/dev/null > rust/tests/fixtures/anthropic_tool_sse.txt
```

Also save DDG HTML and Brave JSON from manual requests for websearch parsing tests.

---

## Unit tests

Inline `#[cfg(test)]` modules in each source file. No network, no Docker, no API keys.

### `types.rs` tests (port of `test_llm_dataclasses.py`)

| Test | What it verifies |
|------|-----------------|
| `test_llm_options_defaults` | Default values: max_tokens=8192, temperature=0.8, think=false |
| `test_llm_options_custom` | Custom values stored correctly |
| `test_request_args_structure` | data/headers accessible |
| `test_stream_response_types` | Each StreamResponseType variant constructible |
| `test_stream_response_to_json` | `to_json()` produces valid JSON with correct fields |
| `test_stream_response_escapes_special_chars` | Quotes in delta survive JSON round-trip |
| `test_tool_call_initial_state` | executed=false, result=None |
| `test_tool_call_execute` | After execute: result populated, executed=true |
| `test_tool_call_parses_json_arguments` | JSON string args deserialized and passed to tool |

### Provider tests (port of `test_llm_request_args.py` + `test_llm_stream_handlers.py` + `test_llm_tool_messages.py`)

Per-provider module tests. Use `include_str!` to load fixture SSE files.

#### Request builders (~17 tests)

| Provider | Test | What it verifies |
|----------|------|-----------------|
| openai_chat | `test_basic_structure` | model, stream, system+user messages |
| openai_chat | `test_bearer_auth` | Authorization header with env var |
| openai_chat | `test_no_auth_without_key` | No header when key name is None |
| openai_chat | `test_temperature_max_tokens` | Values passed through |
| openai_chat | `test_tools_included` | tools array in body |
| openai_chat | `test_reasoning_model_developer_role` | o3 → developer role, no temp |
| openai_chat | `test_gpt51_no_think` | reasoning_effort="none" |
| openai_chat | `test_gpt51_with_think` | reasoning_effort="high" |
| anthropic | `test_basic_structure` | system field, user message, stream |
| anthropic | `test_xapi_key_header` | x-api-key + anthropic-version |
| anthropic | `test_default_max_tokens` | 8192 |
| anthropic | `test_think_mode` | max_tokens=32000, thinking enabled |
| anthropic | `test_tools_included` | tools in body |
| vertexai | `test_anthropic_version_field` | anthropic_version="vertex-2023-10-16" |
| vertexai | `test_gcloud_auth` | Bearer token from subprocess |
| openai_responses | `test_basic_structure` | input, instructions, reasoning, store=false |
| openai_responses | `test_tools_not_supported` | Returns error when tools provided |

#### SSE parsers (~23 tests)

Feed fixture lines, assert `Vec<StreamResponse>` contents:

| Provider | Test | What it verifies |
|----------|------|-----------------|
| openai_chat | `test_ignores_non_data_lines` | Non-"data:" → empty vec |
| openai_chat | `test_done_marker` | `[DONE]` → BlockEnd |
| openai_chat | `test_response_start` | First chunk → ResponseStart with id |
| openai_chat | `test_content_delta` | delta.content → OutputText |
| openai_chat | `test_reasoning_content` | delta.reasoning → ReasoningContent |
| openai_chat | `test_tool_call_start` | Creates ToolCall entry, yields ToolCall response |
| openai_chat | `test_tool_call_accumulation` | Multi-chunk arguments concatenated |
| openai_chat | `test_empty_choices` | Empty choices → empty vec |
| openai_chat | `test_invalid_json` | Malformed JSON → empty vec |
| anthropic | `test_skips_event_lines` | `event:` → empty vec |
| anthropic | `test_message_start` | message_start → ResponseStart |
| anthropic | `test_tool_use_block` | content_block_start tool_use → ToolCall registered |
| anthropic | `test_text_delta` | delta.text → OutputText |
| anthropic | `test_thinking_delta` | delta.thinking → ReasoningContent |
| anthropic | `test_partial_json_accumulates` | partial_json appends to ToolCall.arguments |
| anthropic | `test_message_stop` | message_stop → BlockEnd |
| anthropic | `test_invalid_json` | Malformed → empty vec |
| openai_responses | `test_response_created` | → ResponseStart |
| openai_responses | `test_output_text_delta` | → OutputText |
| openai_responses | `test_reasoning_summary_delta` | → ReasoningContent |
| openai_responses | `test_response_completed` | → BlockEnd |
| openai_responses | `test_ignores_non_data` | → empty vec |
| openai_responses | `test_invalid_json` | → empty vec |

#### Message assemblers (~8 tests)

| Provider | Test | What it verifies |
|----------|------|-----------------|
| openai | `test_empty_noop` | Empty map → messages unchanged |
| openai | `test_single_tool` | 1 assistant msg + 1 tool msg appended |
| openai | `test_multiple_tools` | N tool_calls in assistant + N tool msgs |
| openai | `test_preserves_existing` | Prior messages untouched |
| anthropic | `test_empty_noop` | Empty map → messages unchanged |
| anthropic | `test_single_tool` | assistant with tool_use + user with tool_result |
| anthropic | `test_multiple_tools` | N tool_use blocks + N tool_result blocks |
| anthropic | `test_arguments_parsed` | JSON string → parsed object in input field |

### Tool system tests (port of `test_utils.py`)

| Test | What it verifies |
|------|-----------------|
| `test_register_tool` | Tool added to registry, retrievable by name |
| `test_register_overwrites` | Same name replaces previous |
| `test_openai_schema_structure` | type=function, function.name, parameters.properties |
| `test_anthropic_schema_structure` | name, description, input_schema |
| `test_unknown_tool_returns_none` | get("nonexistent") → None |

### Codesearch tests (port of `test_codesearch.py`, ~18 tests)

Use `tempdir` crate or `std::env::temp_dir()` for isolated test directories:

| Test | What it verifies |
|------|-----------------|
| `test_validate_path_relative` | Valid relative path passes |
| `test_validate_path_dot` | "." passes |
| `test_validate_path_outside_cwd` | "/etc/passwd" → error |
| `test_validate_path_traversal` | "../../../etc/passwd" → error |
| `test_paginate_no_limit` | limit=0 returns all |
| `test_paginate_with_limit` | limit=2 slices correctly |
| `test_paginate_with_offset` | offset skips items |
| `test_paginate_offset_beyond_end` | Returns empty, has_more=false |
| `test_bash_find_lists_files` | Finds files in temp dir |
| `test_bash_find_glob_filter` | name="*.py" filters |
| `test_bash_find_pagination` | limit=1, has_more=true |
| `test_bash_find_invalid_path` | Path outside CWD → error |
| `test_bash_find_directory_filter` | type_filter="d" works |
| `test_bash_ripgrep_finds_pattern` | Known pattern found |
| `test_bash_ripgrep_no_matches` | Returns error object |
| `test_bash_ripgrep_pagination` | limit=2 |
| `test_bash_read_content` | Reads file, content matches |
| `test_bash_read_pagination` | limit=1 returns 1 line |
| `test_bash_read_not_found` | Missing file → error |

### Websearch tests (port of `test_websearch.py`, ~12 tests)

Mock HTTP with `wiremock` crate or similar:

| Test | What it verifies |
|------|-----------------|
| `test_ddg_parses_html` | Fixture HTML → correct title/url/description |
| `test_ddg_empty_query` | "" → empty vec |
| `test_ddg_bot_detection_anomaly_js` | anomaly.js in response → DuckDuckGoBotDetected |
| `test_ddg_bot_detection_202` | HTTP 202 → DuckDuckGoBotDetected |
| `test_brave_parses_json` | Fixture JSON → correct results |
| `test_brave_empty_query` | "" → empty vec |
| `test_brave_missing_key` | No env var → error |
| `test_brave_retries_on_error` | 500 then 200 → succeeds |
| `test_search_web_uses_ddg` | Default path hits DDG |
| `test_search_web_falls_back_on_failure` | DDG 500 → falls back to Brave |
| `test_search_web_falls_back_on_bot` | Bot detected → Brave |
| `test_scrape_error_handling` | Bad URL → error string |

### Pythonexec tests (port of `test_pythonexec.py`, ~8 tests, mock Docker)

Mock `bollard::Docker` and socket:

| Test | What it verifies |
|------|-----------------|
| `test_find_port_success` | Returns available port |
| `test_find_port_none_available` | All occupied → error |
| `test_empty_code` | "" → stderr="No code provided" |
| `test_whitespace_code` | "  " → stderr="No code provided" |
| `test_sends_and_receives` | Mock socket returns stdout |
| `test_truncates_large_output` | >10000 chars truncated |
| `test_container_start_failure` | Docker error → "Execution failed" |
| `test_kill_container_cleans_up` | State reset after kill |

### CLI arg parsing tests (port of `test_main_parse_args.py`, ~12 tests)

Test `Cli::try_parse_from()` with argument vectors:

| Test | What it verifies |
|------|-----------------|
| `test_required_positional_args` | model, url, prompt parsed |
| `test_defaults` | max_tokens=8192, temp=0.8, think=false, sse=false, brief=false |
| `test_custom_max_tokens` | --max-tokens 4096 |
| `test_custom_temperature` | --temperature 0.5 |
| `test_think_flag` | --think → true |
| `test_sse_short_flag` | -S → sse=true |
| `test_sse_long_flag` | --sse → sse=true |
| `test_brief_flag` | -b → brief=true |
| `test_all_providers_accepted` | All 4 providers parse |
| `test_invalid_provider` | Unknown → parse error |
| `test_missing_provider` | → parse error |
| `test_missing_api_key_name` | → parse error |

### Output tests (port of `print_to_stdout` behavior, ~6 tests)

Capture stdout to a buffer:

| Test | What it verifies |
|------|-----------------|
| `test_sse_format` | event: + data: + blank line |
| `test_brief_hides_tool_call` | ToolCall type → no output |
| `test_brief_hides_tool_result` | ToolResult type → no output |
| `test_default_output_text` | OutputText → raw delta, no ANSI |
| `test_default_block_end` | BlockEnd → newline |
| `test_default_response_end` | ResponseEnd → "=== [ DONE ] ===" |

---

## Integration tests

In `rust/tests/integration/`. Gated by env vars — skip if key not set.

### Skip mechanism

```rust
fn require_env(var: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| {
        eprintln!("skipping: {} not set", var);
        std::process::exit(0); // or use #[ignore] + test runner flag
    })
}
```

Alternatively, use `#[ignore]` and run with `cargo test -- --ignored` when keys are available.

### OpenAI integration (~3 tests)

| Test | What it verifies |
|------|-----------------|
| `test_simple_completion` | Stream gpt-4o-mini, collect text, assert non-empty |
| `test_reasoning_model` | gpt-5.1 with think, assert reasoning_effort in request |
| `test_tool_calling` | Register a mock tool, assert tool_calls populated after streaming |

### Anthropic integration (~4 tests)

| Test | What it verifies |
|------|-----------------|
| `test_simple_completion` | Stream Haiku, assert text received |
| `test_thinking_mode` | Sonnet with think, assert reasoning content received |
| `test_tool_calling` | Register tool, assert tool_calls populated |
| `test_response_types` | Assert ResponseStart and BlockEnd received in stream |

---

## E2E tests

In `rust/tests/e2e/`. Run the compiled binary as a subprocess.

### Build requirement

E2E tests need `cargo build` first. Use `assert_cmd` crate:

```rust
use assert_cmd::Command;

#[test]
fn test_help() {
    Command::cargo_bin("buzzllm").unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("--provider"));
}
```

### Test matrix (~12 tests)

| Test | What it verifies |
|------|-----------------|
| `test_help_exits_zero` | --help → exit 0 |
| `test_help_shows_providers` | stdout contains all 4 provider names |
| `test_missing_args_fails` | No args → non-zero exit |
| `test_missing_provider_fails` | No --provider → non-zero, "provider" in stderr |
| `test_missing_api_key_fails` | No --api-key-name → non-zero |
| `test_invalid_provider_fails` | Bad provider → non-zero exit |
| `test_openai_simple_call` | Real API call, text in stdout (skip without key) |
| `test_sse_output_format` | -S → "event:" and "data:" in stdout |
| `test_websearch_template` | --system-prompt websearch completes (skip without key) |
| `test_anthropic_simple_call` | Real Anthropic call (skip without key) |
| `test_think_mode` | --think with Anthropic (skip without key) |
| `test_codesearch_template` | --system-prompt codesearch completes (skip without key) |

---

## Test dependencies (Cargo.toml)

```toml
[dev-dependencies]
assert_cmd = "2"
predicates = "3"
wiremock = "0.6"         # HTTP mocking for websearch tests
tempfile = "3"           # temp directories for codesearch
tokio-test = "0.4"       # async test utilities
serde_json = "1"         # fixture loading
```

## Running tests

```bash
# Unit tests only (fast, no deps)
cargo test --lib

# All unit + e2e (no API keys needed for e2e help/error tests)
cargo test

# Integration tests (need API keys)
cargo test -- --ignored
# or
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... cargo test -- --ignored

# Single test
cargo test test_ddg_parses_html
```

## Total test count

| Category | Tests | Requires |
|----------|-------|----------|
| Unit — types | 9 | nothing |
| Unit — providers (request builders) | 17 | nothing |
| Unit — providers (SSE parsers) | 23 | nothing |
| Unit — providers (message assemblers) | 8 | nothing |
| Unit — tool system | 5 | nothing |
| Unit — codesearch | 18 | `rg` in PATH |
| Unit — websearch | 12 | nothing (mocked HTTP) |
| Unit — pythonexec | 8 | nothing (mocked Docker) |
| Unit — CLI parsing | 12 | nothing |
| Unit — output | 6 | nothing |
| Integration — OpenAI | 3 | `OPENAI_API_KEY` |
| Integration — Anthropic | 4 | `ANTHROPIC_API_KEY` |
| E2E — CLI | 12 | binary built; some need API keys |
| **Total** | **~137** | |
