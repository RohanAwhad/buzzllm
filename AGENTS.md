# AGENTS.md

See `CLAUDE.md` for full architecture docs, provider system, and CLI usage examples.

## Quick reference

```bash
# Install
uv venv -p 3.10 && source .venv/bin/activate && uv pip install .

# Install with test deps
uv pip install -e ".[test]"

# Run tests
uv run pytest tests/unit -v              # unit (no network/docker needed)
uv run pytest tests/integration -v       # needs OPENAI_API_KEY / ANTHROPIC_API_KEY
uv run pytest tests/e2e -v               # CLI smoke tests

# Python exec feature requires docker container
cd python_runtime_docker && bash build_docker.sh build-python-exec && cd ..
```

## Key gotchas

- **Global mutable state**: `utils.AVAILABLE_TOOLS` and `llm.TOOL_CALLS` are module-level dicts. The `conftest.py` autouse fixture `reset_tool_state` clears them between tests. If you add new global state to `llm.py` or `tools/`, add cleanup to that fixture.
- **Logs go to `/tmp/buzzllm.logs`**, not stdout. `logger.remove()` is called at import time in `main.py:4`, so loguru never writes to stderr.
- **asyncio_mode = "auto"** in pytest config -- no need to decorate async tests with `@pytest.mark.asyncio`.
- **Tool schema generation** derives from function docstrings + type hints via `callable_to_*_schema()` in `tools/utils.py`. Functions registered as tools **must** have a docstring with `:param` entries or schema generation will break.

## Layout

```
src/buzzllm/
  main.py          # CLI entrypoint, arg parsing, tool registration, provider dispatch
  llm.py           # LLMOptions/RequestArgs/StreamResponse dataclasses, all provider
                   #   make_*/handle_*/tool_call_response_to_* functions, invoke_llm loop
  prompts/         # system prompt templates keyed by name (websearch, codesearch, etc.)
  tools/
    utils.py       # AVAILABLE_TOOLS registry, add_tool(), callable_to_*_schema()
    websearch.py   # search_web (DuckDuckGo + Brave fallback), scrape_webpage (crawl4ai)
    codesearch.py  # bash_find, bash_ripgrep, bash_read
    pythonexec.py  # python_execute (Docker container on port 8787)

tests/
  conftest.py      # shared fixtures, global state reset, skip markers
  unit/            # fast, no network/docker
  integration/     # needs real API keys in env
  e2e/             # CLI subprocess tests
```

## Adding a new provider

1. Add `make_<name>_request_args()`, `handle_<name>_stream_response()`, and `tool_call_response_to_<name>_messages()` in `llm.py`
2. Add entry to `provider_map` dict in `main.py:chat()` (~line 87)
3. Add the provider name to `--provider` choices in `parse_args()` (~line 40)

## Adding a new tool

1. Create the callable in the appropriate `tools/*.py` file (must have typed params + docstring with `:param`)
2. Register with `utils.add_tool(fn)` in the relevant `elif` branch of `main.py:chat()`
3. Add `callable_to_schema(utils.AVAILABLE_TOOLS["name"])` to the tools list in the same branch
