# Phase 8: Prompts, Logging, and Final Wiring

## Goal

Port all prompt templates, set up file-based logging, implement error handling, and complete the final wiring so the Rust binary has full feature parity with the Python version.

## Source reference

- `src/buzzllm/prompts/__init__.py` — prompt registry
- `src/buzzllm/prompts/*.py` — 6 prompt templates
- `src/buzzllm/main.py:1-5` — loguru setup
- `src/buzzllm/llm.py:99-166` — error handling in `invoke_llm`

## Deliverables

### 1. Prompt templates

#### File structure

```
rust/src/prompts/
  mod.rs
  codesearch.txt       (~49 lines)
  generate.txt         (1 line)
  hackhub.txt          (~196 lines)
  helpful.txt          (1 line)
  replace.txt          (1 line)
  websearch.txt        (~3 lines)
```

#### Implementation

Load prompts at compile time with `include_str!`:

```rust
pub fn get_prompt(name: &str) -> Option<&'static str> {
    match name {
        "codesearch" => Some(include_str!("codesearch.txt")),
        "generate" => Some(include_str!("generate.txt")),
        "hackhub" => Some(include_str!("hackhub.txt")),
        "helpful" => Some(include_str!("helpful.txt")),
        "replace" => Some(include_str!("replace.txt")),
        "websearch" => Some(include_str!("websearch.txt")),
        _ => None,
    }
}

/// Returns known prompt names for --help text
pub fn prompt_names() -> &'static [&'static str] {
    &["codesearch", "generate", "hackhub", "helpful", "replace", "websearch"]
}
```

In `chat()`, resolve the system prompt:

```rust
let system_prompt = get_prompt(&args.system_prompt)
    .unwrap_or(&args.system_prompt);  // treat as literal text if not a known name
```

Copy prompt text verbatim from the Python `*_prompt.py` files into the `.txt` files. No reformatting.

### 2. Logging

#### Setup

Python: `loguru` → `/tmp/buzzllm.logs`, rotation 10MB, retention 5, enqueue=True, stderr removed.

Rust equivalent with `tracing`:

```rust
use tracing_appender::rolling;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn init_logging() {
    let file_appender = rolling::never("/tmp", "buzzllm.logs");
    // TODO: rotation can be added with tracing-appender's RollingFileAppender
    // For now, a single file is fine

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(file_appender)
                .with_ansi(false)
        )
        .with(EnvFilter::from_default_env().add_directive("buzzllm=debug".parse().unwrap()))
        .init();
}
```

Key behavior to match:
- No output to stderr/stdout from the logger (only `print_to_stdout` writes to stdout)
- Log file: `/tmp/buzzllm.logs`
- Log levels: debug, info, error (controlled by `RUST_LOG` env var, defaulting to `debug`)

#### Log points to port

From the Python source:
- `llm.py:110` — log HTTP error response body on non-200 status
- `llm.py:153` — log exceptions in invoke_llm
- `websearch.py:81` — warn on DDG bot detection
- `websearch.py:179-182` — warn on DDG failure, error on both failure
- `websearch.py:234` — error on scrape failure

Use `tracing::{debug, info, warn, error}` macros.

### 3. Error handling

#### Strategy

Use `anyhow::Result` throughout for error propagation. No custom error types needed — this is a CLI tool.

#### invoke_llm error behavior (must match Python)

```
1. HTTP error → log response body, propagate (response.error_for_status())
2. Any error during streaming → print error as StreamResponse(type: BlockEnd)
3. Finally (always) → cleanup pythonexec container (ignore cleanup errors)
4. Finally (always) → print StreamResponse(type: ResponseEnd)
```

No panics should reach the user. All errors become printed messages.

### 4. Final wiring checklist

Everything that needs to connect in `main.rs`:

- [ ] `init_logging()` at top of `main()`
- [ ] CLI parsing → `Cli` struct
- [ ] Prompt resolution: `get_prompt()` or passthrough
- [ ] Provider selection: `Provider::from(args.provider)`
- [ ] Tool registration based on `--system-prompt` name
- [ ] Schema selection (OpenAI vs Anthropic format) based on provider
- [ ] `LlmOptions` construction
- [ ] `invoke_llm()` call
- [ ] Cleanup on Ctrl-C: `tokio::signal::ctrl_c` → cleanup pythonexec

### 5. Build configuration

#### Cargo.toml final dependencies

```toml
[package]
name = "buzzllm"
version = "0.2.8"
edition = "2021"

[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.12", features = ["stream", "json"] }
tokio = { version = "1", features = ["full"] }
futures = "0.3"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
tracing-appender = "0.2"
anyhow = "1"
async-trait = "0.1"
bollard = "0.18"
chromiumoxide = "0.7"
scraper = "0.21"
tokio-retry = "0.3"
regex = "1"
```

#### Release build

```bash
cargo build --release
# Binary at rust/target/release/buzzllm
```

## Verification (full parity)

Run each command from the Python README and verify identical behavior:

1. **Help**: `cargo run -- -h` — all args, defaults, and descriptions match
2. **Basic generation**: OpenAI chat, plain prompt, no tools → streams text
3. **Websearch**: `--system-prompt websearch` → tool calls appear, results feed back, final answer
4. **Codesearch**: `--system-prompt codesearch` → rg/find tools used
5. **Pythonexec**: `--system-prompt pythonexec` → Docker container starts, code executes
6. **Hackhub**: `--system-prompt hackhub` → search-replace blocks generated
7. **Anthropic**: `--provider anthropic` → works with Claude
8. **VertexAI**: `--provider vertexai-anthropic` → gcloud auth, Vertex endpoint
9. **Think mode**: `--think` → reasoning content in yellow
10. **SSE mode**: `-S` → structured SSE output
11. **Brief mode**: `-b` → tool calls hidden
12. **Logs**: `/tmp/buzzllm.logs` contains debug entries
13. **Error recovery**: bad API key → error message printed, exits cleanly
14. **Ctrl-C**: kills container, exits cleanly
