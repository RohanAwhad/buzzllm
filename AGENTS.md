# AGENTS.md

See `CLAUDE.md` for full architecture docs, provider system, and CLI usage examples.

## Two implementations

This repo has **parallel Python and Rust** implementations of the same CLI (`buzzllm`). CI only covers Rust. Python tests run locally.

## Quick reference — Python

```bash
# Install (Python 3.10, pinned in .python-version)
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

## Quick reference — Rust

```bash
# All Rust commands run from the rust/ directory
cargo build                        # debug build
cargo build --release              # release build
cargo test -- --test-threads=1     # tests MUST run single-threaded
cargo clippy -- -D warnings        # lint (CI treats warnings as errors)
cargo fmt -- --check               # format check
```

CI runs: build → test → clippy → fmt (all in `rust/`, see `.github/workflows/ci.yml`).

## Key gotchas

- **Global mutable state**: `utils.AVAILABLE_TOOLS`, `llm.TOOL_CALLS`, and `llm.current_tool_call_id` are module-level globals. The `conftest.py` autouse fixture `reset_tool_state` clears all three between tests. If you add new global state to `llm.py` or `tools/`, add cleanup to that fixture.
- **Logs go to `/tmp/buzzllm.logs`**, not stdout. `logger.remove()` is called at import time in `main.py:4`, so loguru never writes to stderr.
- **asyncio_mode = "auto"** in pytest config — no need to decorate async tests with `@pytest.mark.asyncio`.
- **Tool schema generation** derives from function docstrings + type hints via `callable_to_*_schema()` in `tools/utils.py`. Functions registered as tools **must** have a docstring with `:param` entries or schema generation will break.
- **Rust tests require `--test-threads=1`** — they share state that breaks under parallel execution.
- **codesearch CWD**: `codesearch.py` has a module-level `CWD` variable. Tests monkeypatch it via the `temp_cwd` fixture in `conftest.py`.

## Layout

```
src/buzzllm/                       # Python implementation
  main.py                          #   CLI entrypoint, provider dispatch, tool registration
  llm.py                           #   dataclasses, provider functions, invoke_llm loop
  prompts/                         #   system prompt templates (websearch, codesearch, etc.)
  tools/
    utils.py                       #   AVAILABLE_TOOLS registry, callable_to_*_schema()
    websearch.py                   #   search_web (DDG + Brave fallback), scrape_webpage
    codesearch.py                  #   bash_find, bash_ripgrep, bash_read
    pythonexec.py                  #   python_execute (Docker on port 8787)

rust/                              # Rust implementation (same CLI, CI-tested)
  src/
    main.rs                        #   CLI entrypoint
    lib.rs, llm.rs, types.rs       #   core logic
    providers/                     #   per-provider request/response handling
    tools/                         #   tool implementations
    prompts/                       #   prompt templates

tests/                             # Python tests only
  conftest.py                      #   shared fixtures, global state reset, skip markers
  unit/                            #   fast, no network/docker
  integration/                    #   needs real API keys in env
  e2e/                             #   CLI subprocess tests
```

## Adding a new provider

1. Add `make_<name>_request_args()`, `handle_<name>_stream_response()`, and `tool_call_response_to_<name>_messages()` in `llm.py`
2. Add entry to `provider_map` dict in `main.py:chat()` (~line 87)
3. Add the provider name to `--provider` choices in `parse_args()` (~line 40)

## Adding a new tool

1. Create the callable in the appropriate `tools/*.py` file (must have typed params + docstring with `:param`)
2. Register with `utils.add_tool(fn)` in the relevant `elif` branch of `main.py:chat()`
3. Add `callable_to_schema(utils.AVAILABLE_TOOLS["name"])` to the tools list in the same branch
