# Phase 3: Streaming HTTP and invoke_llm Loop

## Goal

Implement the core LLM invocation loop: make a streaming HTTP request, parse SSE lines through the provider, execute tool calls, feed results back, repeat until done. Wire up stdout printing with ANSI colors, SSE mode, and brief mode.

## Source reference

- `src/buzzllm/llm.py:84-166` — `invoke_llm()`
- `src/buzzllm/llm.py:74-81` — `run_tools()`
- `src/buzzllm/llm.py:170-194` — `print_to_stdout()`
- `src/buzzllm/main.py:65-171` — `chat()` orchestrator

## Deliverables

### File structure

```
rust/src/
  llm.rs               invoke_llm() + run_tools()
  output.rs            print_to_stdout()
  main.rs              chat() wiring (update from Phase 1)
```

### `invoke_llm()` loop

Port the Python loop. Key behavior:

```
loop {
    1. POST request to opts.url with request_args (streaming)
    2. Read response line-by-line
    3. For each line: parse via provider.parse_sse_line()
    4. For each StreamResponse: print_to_stdout()
    5. After stream ends: if no tool_calls → return
    6. Execute all pending tool calls concurrently
    7. Print tool results
    8. Assemble tool messages via provider.assemble_tool_messages()
    9. Clear tool_calls, loop
}
```

Important details:
- HTTP timeout: 900 seconds (Python: `timeout=900`)
- Max iterations: `opts.max_infer_iters` (default 10) — Python doesn't enforce this currently but the field exists
- On HTTP error: log error body, raise/propagate
- On any exception: print error as `StreamResponse(type: BlockEnd)`, then print `ResponseEnd`
- Finally block: call `pythonexec::cleanup()` (ignore errors), print `ResponseEnd`

### HTTP streaming with reqwest

```rust
let response = client
    .post(&opts.url)
    .headers(request_args.headers)
    .json(&request_args.data)
    .timeout(Duration::from_secs(900))
    .send()
    .await?;

let mut stream = response.bytes_stream();
let mut buffer = String::new();

// Process byte chunks into lines (split on \n)
// Feed complete lines to provider.parse_sse_line()
```

Handle partial lines: buffer bytes until `\n` is seen, then process the complete line.

### `run_tools()`

Execute all unexecuted tool calls concurrently using `tokio::join!` or `futures::future::join_all`:

```rust
async fn run_tools(
    tool_calls: &mut HashMap<String, ToolCall>,
    registry: &ToolRegistry,
) {
    let futures: Vec<_> = tool_calls.values_mut()
        .filter(|tc| !tc.executed)
        .map(|tc| tc.execute(registry))
        .collect();
    futures::future::join_all(futures).await;
}
```

### `print_to_stdout()` (`output.rs`)

Three modes, checked in order:

1. **Brief mode** (`--brief`): skip `ToolCall` and `ToolResult` types entirely
2. **SSE mode** (`-S`): print `event: {type}\ndata: {json}\n\n`
3. **Default**: ANSI-colored output
   - `ToolCall` → cyan (`\x1b[96m`)
   - `ToolResult` → green (`\x1b[92m`)
   - `ReasoningContent` → yellow (`\x1b[93m`)
   - `BlockEnd` → newline
   - `ResponseEnd` → `\n\n=== [ DONE ] ===`
   - `OutputText` → raw text, no color

All prints use `flush=true` equivalent (`stdout().flush()` or `print!` with explicit flush).

### `chat()` wiring in `main.rs`

```rust
async fn chat(args: Cli) {
    let provider = Provider::from_str(&args.provider);
    let system_prompt = resolve_prompt(&args.system_prompt);  // lookup or passthrough
    let tools = register_tools(&args.system_prompt, &provider); // Phase 4+ stub: returns None

    let opts = LlmOptions { ... };
    invoke_llm(opts, &args.prompt, &system_prompt, provider, tools).await;
}
```

## Verification

1. **No-tools streaming**: `cargo run -- "gpt-4o-mini" "https://api.openai.com/v1/chat/completions" "say hello" --provider openai-chat --api-key-name OPENAI_API_KEY` — streams colored text, ends with `=== [ DONE ] ===`
2. **SSE mode**: same with `-S` — outputs `event:/data:` format
3. **Brief mode**: same with `-b` — identical to default for no-tool calls
4. **Anthropic provider**: test with Claude model
5. **Think mode**: `--think` with Anthropic — yellow reasoning content appears
6. **Error handling**: use a bad URL — prints error and `=== [ DONE ] ===`
