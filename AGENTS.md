# AGENTS.md

## Quick reference

```bash
# Build
cargo build                        # debug
cargo build --release              # release (~8.8 MB binary)

# Test (single-threaded required)
cargo test -- --test-threads=1

# Lint and format
cargo clippy -- -D warnings        # CI treats warnings as errors
cargo fmt -- --check               # format check
```

CI runs all four checks: build, test, clippy, fmt (`.github/workflows/ci.yml`).

## Key gotchas

- **Tests require `--test-threads=1`** — `set_current_dir` is process-global; parallel tests corrupt each other's CWD. CI enforces this.
- **Logs go to `/tmp/buzzllm.logs`**, not stdout. Controlled via `RUST_LOG` env var (e.g. `RUST_LOG=buzzllm=debug`).
- **Tool schema generation** uses the `Tool` trait's `openai_schema()` / `anthropic_schema()` methods. Each tool struct must implement both.
- **Prompt templates** are compiled in via `include_str!` in `src/prompts/mod.rs`. Adding a prompt means adding both a `.txt` file and a match arm in `get_prompt()`.
- **`pythonexec` tool needs Docker** — build the image first: `cd python_runtime_docker && bash build_docker.sh build-python-exec`
- **`codesearch` tools require `rg` (ripgrep)** — `brew install ripgrep` / `apt install ripgrep`.
- **`.cargo/config.toml`** sets `RUST_TEST_THREADS=8` (overridden by explicit `--test-threads=1` on CLI).

## Layout

```
src/
  main.rs              CLI (clap) + prompt resolution + tool registration
  llm.rs               invoke_llm() streaming loop, SSE parsing, tool execution
  types.rs             LlmOptions, RequestArgs, StreamResponse, ToolCallData
  output.rs            Colored stdout or SSE event format
  lib.rs               Crate root, re-exports
  providers/
    mod.rs             LlmClient trait + create_client() factory
    openai_chat.rs     /v1/chat/completions
    openai_responses.rs /v1/responses
    anthropic.rs       /v1/messages
    vertexai_anthropic.rs GCP Vertex AI (delegates SSE to anthropic)
  tools/
    mod.rs             Tool trait + ToolRegistry
    codesearch.rs      BashFind, BashRipgrep, BashRead
    websearch.rs       SearchWeb (DDG + Brave fallback), ScrapeWebpage
    pythonexec.rs      PythonExecute (Docker via bollard)
    write_file.rs      WriteFile (exact string replace)
    bash.rs            Bash (arbitrary shell commands)
  prompts/
    mod.rs             get_prompt() + prompt_names()
    *.txt              7 prompt templates (include_str! at compile time)

tests/                 Integration/e2e tests (cargo test)
```

## Adding a new provider

1. Create `src/providers/<name>.rs` — implement `LlmClient` trait
2. Add the struct + match arm in `create_client()` in `src/providers/mod.rs`
3. Add the provider name to `value_parser` in `src/main.rs` Cli struct

## Adding a new tool

1. Create a struct implementing `Tool` trait in `src/tools/<name>.rs`
2. Register it in the relevant prompt branch in `src/main.rs`
3. Add `mod <name>;` to `src/tools/mod.rs`
