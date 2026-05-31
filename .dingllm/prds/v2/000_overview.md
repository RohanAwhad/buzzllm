# BuzzLLM v2 — Python to Rust Migration

## Goal

Rewrite BuzzLLM from Python to Rust. Produce a single static binary (`buzzllm`) with identical CLI interface, identical provider support, and identical tool capabilities. The Rust code lives in `rust/` within this repo.

## Current Python codebase (~1600 SLOC)

```
src/buzzllm/
  main.py            195 lines   CLI + orchestration
  llm.py             576 lines   types, 4 providers, SSE parsing, invoke_llm loop
  prompts/           ~255 lines  6 static prompt templates
  tools/
    utils.py          136 lines  tool registry + schema gen
    websearch.py      235 lines  DDG + Brave + crawl4ai scraping
    codesearch.py     256 lines  rg/find subprocess wrappers
    pythonexec.py     176 lines  Docker container + socket protocol
```

## Architecture (Rust)

```
rust/
  Cargo.toml
  src/
    main.rs           CLI (clap) + async main
    types.rs           LlmOptions, RequestArgs, StreamResponse, ToolCall
    providers/
      mod.rs           Provider trait + dispatch enum
      openai_chat.rs   request builder + SSE parser + message assembler
      openai_responses.rs
      anthropic.rs
      vertexai_anthropic.rs
    tools/
      mod.rs           Tool trait + registry
      schema.rs        OpenAI/Anthropic JSON schema generation
      codesearch.rs    bash_find, bash_ripgrep, bash_read
      websearch.rs     search_web (DDG + Brave), scrape_webpage (chromiumoxide)
      pythonexec.rs    Docker container lifecycle + socket protocol
    prompts/
      mod.rs           prompt registry (HashMap or match)
      *.txt            prompt templates loaded via include_str!
    output.rs          print_to_stdout (ANSI, SSE, brief modes)
```

## Phases

| Phase | PRD | Delivers | Depends on |
|-------|-----|----------|------------|
| 1 | [001_core_types_and_cli.md](001_core_types_and_cli.md) | Cargo project, all types, clap CLI, stubs | nothing |
| 2 | [002_provider_system.md](002_provider_system.md) | Provider trait, all 4 request builders + SSE parsers | Phase 1 |
| 3 | [003_streaming_and_llm_loop.md](003_streaming_and_llm_loop.md) | `invoke_llm` loop, reqwest streaming, stdout output | Phase 2 |
| 4 | [004_tool_system.md](004_tool_system.md) | Tool trait, registry, schema gen | Phase 1 |
| 5 | [005_codesearch.md](005_codesearch.md) | codesearch tools (rg, find, read) | Phase 4 |
| 6 | [006_websearch.md](006_websearch.md) | websearch tools (DDG, Brave, chromiumoxide) | Phase 4 |
| 7 | [007_pythonexec.md](007_pythonexec.md) | pythonexec tool (Docker + socket) | Phase 4 |
| 8 | [008_prompts_and_logging.md](008_prompts_and_logging.md) | Prompt templates, tracing, error handling, final wiring | Phase 3 + 4 |
| 9 | [009_testing.md](009_testing.md) | Test infrastructure, fixtures, unit/integration/e2e tests | All phases |

Phases 5, 6, 7 are independent of each other (all depend on Phase 4 only).
Phase 9 (testing) is written alongside each phase — the PRD defines the full test matrix.

## Crate dependencies

| Crate | Purpose |
|-------|---------|
| `clap` (derive) | CLI arg parsing |
| `serde`, `serde_json` | JSON serialization for all types and API payloads |
| `reqwest` (stream feature) | HTTP client with SSE streaming |
| `tokio` (full) | async runtime |
| `futures` | stream combinators |
| `tracing`, `tracing-subscriber`, `tracing-appender` | logging to `/tmp/buzzllm.logs` |
| `anyhow` | error propagation |
| `bollard` | Docker Engine API (pythonexec) |
| `chromiumoxide` | headless Chrome for webpage scraping |
| `scraper` | HTML parsing (DuckDuckGo results) |
| `tokio-retry` | retry with exponential backoff (websearch) |

## Acceptance criteria

1. `cargo run -- --help` output matches Python `buzzllm -h`
2. All 4 providers stream responses identically (text, reasoning, tool calls)
3. `--system-prompt websearch|codesearch|pythonexec` activates correct tools
4. `--sse`, `--brief`, `--think` flags behave identically
5. Tool results feed back into conversation loop correctly
6. Logs write to `/tmp/buzzllm.logs`
7. Single binary, no Python runtime required

## Non-goals

- No new features beyond current Python parity
- No GUI or TUI
- No config file system (args-only, like current Python)
- No plugin system beyond the existing tool registration pattern
