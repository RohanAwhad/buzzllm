# Phase 1: Core Types and CLI

## Goal

Set up the Cargo project in `rust/`, define all core data types, and implement the CLI argument parser. The binary should compile and print parsed args.

## Source reference

- `src/buzzllm/llm.py:15-68` — `LLMOptions`, `RequestArgs`, `StreamResponse`, `ToolCall`
- `src/buzzllm/main.py:26-62` — `parse_args()`

## Deliverables

### 1. Cargo project scaffold

```
rust/
  Cargo.toml
  src/
    main.rs
    types.rs
```

`Cargo.toml` dependencies for this phase:
- `clap` (derive feature)
- `serde`, `serde_json`
- `tokio` (full)

### 2. Types (`types.rs`)

Port these Python dataclasses to Rust structs with `serde::Serialize` + `Deserialize`:

#### `LlmOptions`

```rust
pub struct LlmOptions {
    pub model: String,
    pub url: String,
    pub api_key_name: Option<String>,
    pub max_tokens: Option<u32>,      // default 8192
    pub temperature: f64,              // default 0.8
    pub think: bool,                   // default false
    pub tools: Option<Vec<serde_json::Value>>,
    pub max_infer_iters: u32,          // default 10
}
```

#### `RequestArgs`

```rust
pub struct RequestArgs {
    pub data: serde_json::Value,       // JSON body
    pub headers: HashMap<String, String>,
}
```

#### `StreamResponse`

```rust
pub enum StreamResponseType {
    ResponseStart,
    OutputText,
    ReasoningContent,
    ToolCall,
    ToolResult,
    BlockEnd,
    ResponseEnd,
}

pub struct StreamResponse {
    pub id: String,
    pub delta: String,
    pub response_type: StreamResponseType,
}
```

Must implement `to_json(&self) -> String` for SSE output mode.

#### `ToolCall`

```rust
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,          // raw JSON string, parsed at execution time
    pub executed: bool,
    pub result: Option<serde_json::Value>,
}
```

### 3. CLI (`main.rs`)

Use `clap` derive to mirror the exact Python interface:

```
buzzllm <MODEL> <URL> <PROMPT> \
    --provider <openai-chat|openai-responses|anthropic|vertexai-anthropic> \
    --api-key-name <ENV_VAR> \
    [--system-prompt <TEXT>] \
    [--max-tokens <N>] \
    [--temperature <F>] \
    [--think] \
    [-S | --sse] \
    [-b | --brief]
```

Positional args: `model`, `url`, `prompt` (in that order).

Defaults to match Python:
- `--system-prompt`: `"scream at mee for not setting your system prompt"`
- `--max-tokens`: `8192`
- `--temperature`: `0.8`

Provider is required. `--api-key-name` is required.

### 4. Stub main

```rust
#[tokio::main]
async fn main() {
    let args = Cli::parse();
    // TODO: wire to chat()
    println!("parsed: {:?}", args);
}
```

## Verification

- `cargo build` compiles with no errors
- `cargo run -- --help` prints usage matching Python's `buzzllm -h`
- `cargo run -- "gpt-4o-mini" "https://api.openai.com/v1/chat/completions" "hello" --provider openai-chat --api-key-name OPENAI_API_KEY` parses without error
- Unit test: construct each type, serialize to JSON, verify fields
