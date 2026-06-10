# BuzzLLM

LLM gateway CLI — stream responses from any provider, with tools. Web search, codebase exploration, Python execution, file editing, shell commands.

```bash
buzzllm gpt-4.1-mini "" "Say hello" --provider openai-chat --api-key-name OPENAI_API_KEY
# → Hello! How can I help you today?
# === [ DONE ] ===
```

## Install

Requires Rust toolchain (1.75+).

```bash
git clone https://github.com/RohanAwhad/buzzllm.git
cd buzzllm
cargo build --release
# Binary: target/release/buzzllm (~8.8 MB)
```

Optionally copy to PATH:
```bash
cp target/release/buzzllm /usr/local/bin/
```

## Prerequisites

### Required

| Dependency | Used by | Install |
|-----------|---------|---------|
| Rust 1.75+ | Build | [rustup.rs](https://rustup.rs) |
| `rg` (ripgrep) | `codesearch`, `coding` tools | `brew install ripgrep` / `apt install ripgrep` |
| `/bin/sh` | `coding` tool (`bash`) | Pre-installed on macOS/Linux |
| API key env var | All providers (except Vertex AI) | See providers table below |

### Optional

| Dependency | Used by | Install |
|-----------|---------|---------|
| `gcloud` CLI | `vertexai-anthropic` provider | [gcloud SDK](https://cloud.google.com/sdk/docs/install) |
| Docker | `pythonexec` tool | [docker.com](https://docker.com) |
| `BRAVE_SEARCH_AI_API_KEY` env var | Brave fallback in `websearch` | [brave.com/search/api](https://brave.com/search/api/) |

### Platform notes

- **macOS**: `brew install ripgrep`
- **Ubuntu/Debian**: `sudo apt-get install ripgrep`
- `find` is pre-installed on macOS and Linux (used by `codesearch` for directory listings)

## Quick start

```bash
# OpenAI Chat Completions (default URL: https://api.openai.com/v1/chat/completions)
buzzllm gpt-4.1-mini "" "What is 2+2?" --provider openai-chat --api-key-name OPENAI_API_KEY

# Anthropic (default URL: https://api.anthropic.com/v1/messages)
buzzllm claude-sonnet-4-20250514 "" "Hello" --provider anthropic --api-key-name ANTHROPIC_API_KEY

# Override URL (vLLM, LiteLLM proxy, region endpoints)
buzzllm mistral-7b "http://localhost:8000/v1/chat/completions" "Hi" --provider openai-chat --api-key-name OPENAI_API_KEY

# SSE output mode
buzzllm gpt-4.1-mini "" "Hello" --provider openai-chat --api-key-name OPENAI_API_KEY -S

# Brief mode (hide tool calls, show only final output)
buzzllm gpt-4.1-mini "" "What is the weather?" --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt websearch -b

# Thinking mode (Claude extended thinking or OpenAI reasoning effort)
buzzllm claude-sonnet-4-20250514 "" "Solve this problem..." --provider anthropic --api-key-name ANTHROPIC_API_KEY --think
```

## Providers

| Provider | `--provider` value | Default URL |
|----------|-------------------|-------------|
| OpenAI Chat Completions | `openai-chat` | `https://api.openai.com/v1/chat/completions` |
| OpenAI Responses | `openai-responses` | `https://api.openai.com/v1/responses` |
| Anthropic Messages | `anthropic` | `https://api.anthropic.com/v1/messages` |
| Vertex AI Anthropic | `vertexai-anthropic` | none (pass `""` as URL, uses gcloud auth) |

URL is a required positional argument (2nd argument). Pass `""` to use the provider default URL. To override, pass a custom URL in its place.

Auth: `--api-key-name` specifies the environment variable holding the API key. For Vertex AI, the tool reads `gcloud auth print-access-token` automatically.

## System prompts & tools

BuzzLLM ships with 7 system prompts. Each registers a specific set of tools:

### `websearch`
Web search via DuckDuckGo (with Brave fallback) + webpage scraping.

**Tools:** `search_web`, `scrape_webpage`
```bash
buzzllm gpt-4.1-mini "" "Latest AI news?" --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt websearch
```

### `codesearch`
Explore local codebase: find files, grep text, read file contents.

**Tools:** `bash_find`, `bash_ripgrep`, `bash_read`
```bash
buzzllm gpt-4.1-mini "" "Where is error handling defined?" --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt codesearch
```

### `coding`
Full coding agent: read/edit files, execute shell commands, search the web.

**Tools:** `bash_read`, `write_file`, `bash`, `search_web`, `scrape_webpage`
```bash
# Read a file
buzzllm gpt-4.1-mini "" "Show me Cargo.toml" --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt coding

# Edit a file
buzzllm gpt-4.1-mini "" "Change version to 0.3.0 in Cargo.toml" --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt coding

# Create new files (pass empty old_string)
buzzllm gpt-4.1-mini "" "Create src/hello.rs with a hello world Rust program" --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt coding

# Run tests
buzzllm gpt-4.1-mini "" "Run the test suite" --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt coding

# Meta-agent: invoke another agent via bash
buzzllm gpt-4.1-mini "" "Research X using websearch" --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt coding
```

`write_file` uses exact string replacement — the `old_string` must match exactly once in the file. Pass empty `old_string` to create a new file.

### `pythonexec`
Execute Python code in a Docker container for safe sandboxed execution.

**Tools:** `python_execute`

**Prerequisites:** build the Docker image first:
```bash
cd python_runtime_docker
bash build_docker.sh build-python-exec
cd ..
```

```bash
buzzllm gpt-4.1-mini "" "Solve 5 = mx + c where m=4/2, x=1 using Python" \
  --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt pythonexec
```

### `replace`
Generates search-replace code change blocks. For code modification workflows that output structured patches.

### `generate`
General code generation.

### `helpful`
Default helpful assistant (no tools).

### Custom prompt

Pass any text as the system prompt:
```bash
buzzllm gpt-4.1-mini "" "Tell me a joke" --provider openai-chat --api-key-name OPENAI_API_KEY --system-prompt "You are a comedian."
```

## OpenCLI / BuzzLLM Gateway integration

BuzzLLM also serves as a gateway tool for OpenCLI and Claude Code via the [BuzzLLM Gateway skill](https://github.com/RohanAwhad/buzzllm). When used as a skill, BuzzLLM can perform web searches, execute Python code, and analyze codebases programmatically.

## Dev mode (debug logs)

Logs written to `/tmp/buzzllm.logs` with daily rotation. Controlled via `RUST_LOG` env var:
```bash
RUST_LOG=buzzllm=debug buzzllm ...
```

## Architecture

```
src/
  main.rs          CLI (clap) + prompt resolution + tool registration
  llm.rs           invoke_llm() streaming loop, SSE parsing, tool execution
  types.rs         LlmOptions, RequestArgs, StreamResponse, ToolCallData
  output.rs        Colored stdout or SSE event format
  providers/
    mod.rs         LlmClient trait + 4 implementations + create_client() factory
    openai_chat.rs        /v1/chat/completions
    openai_responses.rs   /v1/responses
    anthropic.rs          /v1/messages
    vertexai_anthropic.rs GCP Vertex AI (delegates SSE to anthropic)
  tools/
    mod.rs         Tool trait + ToolRegistry
    codesearch.rs  BashFind (rg --files / find -type d), BashRipgrep (rg), BashRead
    websearch.rs   SearchWeb (DDG → Brave fallback), ScrapeWebpage (reqwest + scraper)
    pythonexec.rs  PythonExecute (Docker via bollard)
    write_file.rs  WriteFile (exact string replace)
    bash.rs        Bash (arbitrary shell commands)
  prompts/
    mod.rs         get_prompt() + prompt_names()
    *.txt          7 prompt templates (include_str! at compile time)
```

See `.dingllm/specs/v2/` for C4 architecture and sequence diagrams.

## Testing

```bash
cargo test -- --test-threads=1   # single-threaded required (set_current_dir is process-global)
```

CI enforces `--test-threads=1`.
