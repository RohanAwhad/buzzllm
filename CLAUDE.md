# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BuzzLLM is a CLI gateway for LLM tasks including websearch, Python code execution, local code repository analysis, and code modifications. The project uses Python 3.10+ with UV for dependency management.

## Essential Commands

### Setup and Installation
```bash
uv venv -p 3.10
source .venv/bin/activate
uv pip install .

# For Python execution features, build Docker container:
cd python_runtime_docker && bash build_docker.sh build-python-exec && cd ..
```

### Running the CLI
```bash
# Basic usage
buzzllm "MODEL_NAME" "API_URL" "PROMPT" --provider PROVIDER --api-key-name API_KEY_ENV_VAR

# Example with OpenAI
buzzllm "gpt-4o-mini" \
    "https://api.openai.com/v1/chat/completions" \
    "hello, world" \
    --provider openai-chat \
    --api-key-name OPENAI_API_KEY \
    --system-prompt "You are a helpful agent"

# With extended thinking (Claude/OpenAI reasoning models)
buzzllm "claude-sonnet-4-20250514" \
    "https://api.anthropic.com/v1/messages" \
    "solve this problem" \
    --provider anthropic \
    --api-key-name ANTHROPIC_API_KEY \
    --think

# SSE output mode (for programmatic consumption)
buzzllm ... -S
```

### System Prompt Templates
Templates with tools:
- `websearch`: Web search (`search_web`, `scrape_webpage` tools). Falls back to Brave Search if DuckDuckGo fails (needs `BRAVE_SEARCH_AI_API_KEY`)
- `codesearch`: Codebase analysis (`bash_find`, `bash_ripgrep`, `bash_read` tools)
- `pythonexec`: Python execution in Docker (`python_execute` tool)

Templates without tools:
- `hackhub`: Search-Replace block generation for code modifications
- `generate`, `helpful`, `replace`: Other prompt templates

## Architecture

### Data Flow
```
main.py (CLI args)
  → chat() selects provider functions from provider_map
  → invoke_llm() handles streaming request/response loop
  → Tools executed via run_tools() when LLM returns tool_calls
  → Results fed back to LLM until no more tool calls
```

### Provider System
Each provider has 4 functions in `llm.py`:
1. `make_*_request_args()`: Build request headers/body
2. `handle_*_stream_response()`: Parse SSE chunks into `StreamResponse` objects
3. `tool_call_response_to_*_messages()`: Format tool results for conversation
4. Uses `callable_to_*_schema()` from `tools/utils.py` for tool schema conversion

Providers:
- `openai-chat`: OpenAI chat completions (also works with compatible APIs)
- `openai-responses`: OpenAI Responses API (reasoning models)
- `anthropic`: Anthropic Claude API
- `vertexai-anthropic`: Claude via Google Vertex AI (uses `gcloud auth`)

### Tool Registration
Tools are registered dynamically in `main.py` based on `--system-prompt`:
```python
utils.add_tool(websearch.search_web)  # registers to AVAILABLE_TOOLS dict
tools = [callable_to_schema(utils.AVAILABLE_TOOLS["search_web"])]
```

### Key Dataclasses
- `LLMOptions`: Request config (model, url, max_tokens, temperature, think, tools)
- `RequestArgs`: HTTP request data (headers, body)
- `StreamResponse`: Parsed streaming chunk (type: output_text|reasoning_content|tool_call|tool_result|...)
- `ToolCall`: Tracks tool execution state and results

## Development Notes

- Entry point: `buzzllm = "buzzllm.main:main"` in pyproject.toml
- Python exec container: `buzz/python-exec:latest`, uses persistent IPython kernel via socket on port 8787
- Websearch uses crawl4ai with Playwright (auto-installs Chromium on first use)