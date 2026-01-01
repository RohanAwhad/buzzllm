# BuzzLLM

This is a gateway for all llm tasks that I need to do. Examples:
1. Websearch
2. Python Code Execution
3. Ask questions about local code repo
4. Make changes

### How to Run:

1. Setup:
    ```bash
    git clone https://github.com/RohanAwhad/buzzllm.git
    cd buzzllm
    uv venv -p 3.10
    source .venv/bin/activate
    uv pip install .
    ```

2. Run generation:
    ```bash
    buzzllm "gpt-4o-mini" \
        "https://api.openai.com/v1/chat/completions" \
        "hello, world" \
        --provider openai-chat \
        --api-key-name OPENAI_API_KEY \
        --system-prompt "You are a helpful agent" 
    ```

### Notes:

1. `--provider openai-chat`: It uses openai chat completion compatible api endpoint for llm calls
2. `--system-prompt websearch`: It uses the pre-built websearch system prompt and tools
    - Available templates:
        1. `websearch`: Used for answering general questions that require websearch. Uses `search_web`, and `scrape_webpage` tools
        2. `codesearch`: Used for answering questions about current codebase. Uses `bash_find`, `bash_grep`, and `bash_read` tools
        3. `pythonexec`: Used for executing python code in a kernel as a tool for the llm answer generation.
        4. `hackhub`: Used for "Apply Changes" functionality from Cursor in neovim. Generates changes in Search-Replace blocks.


### Usage examples:

- help
    ```bash
    buzzllm -h
    ```
- websearch
    ```bash
    buzzllm "gpt-4o-mini" \
        "https://api.openai.com/v1/chat/completions" \
        "What was low for Meta's stock price yesterday? Today is July 1, 2025" \
        --provider openai-chat \
        --api-key-name OPENAI_API_KEY \
        --system-prompt websearch 
    ```
- codesearch
    ```bash
    buzzllm "gpt-4o-mini" \
        "https://api.openai.com/v1/chat/completions" \
        "in the current repo for scraping url contents what do we use?" \
        --provider openai-chat \
        --api-key-name OPENAI_API_KEY \
        --system-prompt codesearch 
    ```
- pythonexec
    ```bash
    # before running python execution template,
    # we need a docker container to execute code safely

    cd python_runtime_docker
    bash build_docker.sh build-python-exec
    cd ../
    ```
    ```bash
    # Now we can run the python execution

    buzzllm "gpt-4o-mini" \
        "https://api.openai.com/v1/chat/completions" \
        "solve the equation 5 = mx + c, where m = 4/2 and x = 1. use python to write code and execute" \
        --provider openai-chat \
        --api-key-name OPENAI_API_KEY \
        --system-prompt pythonexec
    ```
- hackhub: We will be using claude sonnet 4 here. Its pretty easy to change provider.
    ```bash
    buzzllm "claude-sonnet-4-20250514" \
        "https://api.anthropic.com/v1/messages" \
        "$(cat src/buzzllm/main.py)\nI need you to add a new tools argument to cli. There will be multiple tools, and in help provide a list of available tools" \
        --provider anthropic \
        --api-key-name ANTHROPIC_API_KEY \
        --system-prompt hackhub
    ```

### Testing

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Run all tests
uv run pytest

# Run by category
uv run pytest tests/unit -v          # unit tests
uv run pytest tests/integration -v   # requires API keys
uv run pytest tests/e2e -v           # CLI tests

# With coverage
uv run pytest --cov=buzzllm
```
