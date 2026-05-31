# Phase 4: Tool System

## Goal

Define the `Tool` trait, tool registry, and JSON schema generation for both OpenAI and Anthropic formats. This phase provides the infrastructure that Phases 5-7 plug into.

## Source reference

- `src/buzzllm/tools/utils.py:1-8` — `AVAILABLE_TOOLS` dict, `add_tool()`
- `src/buzzllm/tools/utils.py:11-42` — `callable_to_openai_schema()`
- `src/buzzllm/tools/utils.py:45-73` — `callable_to_anthropic_schema()`
- `src/buzzllm/tools/utils.py:76-136` — `_python_type_to_json_schema()`
- `src/buzzllm/main.py:120-147` — tool registration per system-prompt
- `src/buzzllm/llm.py:60-67` — `ToolCall.execute()`

## Deliverables

### File structure

```
rust/src/tools/
  mod.rs             Tool trait + ToolRegistry
  schema.rs          schema generation (not needed if schema is per-tool)
```

### Tool trait

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name as it appears in the API schema
    fn name(&self) -> &str;

    /// JSON schema for this tool in OpenAI function-calling format
    fn openai_schema(&self) -> serde_json::Value;

    /// JSON schema for this tool in Anthropic tool format
    fn anthropic_schema(&self) -> serde_json::Value;

    /// Execute the tool with JSON arguments, return JSON result
    async fn execute(&self, args: serde_json::Value) -> serde_json::Value;
}
```

### Why not reflection-based schema gen

Python generates schemas from docstrings + type hints at runtime. Rust has no runtime reflection. Two options:

1. **Hardcode schemas per tool** — each tool struct returns its schema as a `serde_json::json!()` literal
2. **Derive macro** — overkill for 6 tools

Use option 1. Each tool implementation defines its own schema. This is explicit and avoids macro complexity.

### ToolRegistry

```rust
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, tool: Box<dyn Tool>);
    pub fn get(&self, name: &str) -> Option<&dyn Tool>;

    /// Get schemas for all registered tools in the provider's format
    pub fn openai_schemas(&self) -> Vec<Value>;
    pub fn anthropic_schemas(&self) -> Vec<Value>;
}
```

### Tool registration dispatch

In `main.rs` `chat()`, mirror the Python pattern:

```rust
let mut registry = ToolRegistry::new();
let tools = match system_prompt_name {
    "websearch" => {
        registry.register(Box::new(SearchWeb));
        registry.register(Box::new(ScrapeWebpage));
        Some(match provider {
            Provider::OpenaiChat | Provider::OpenaiResponses => registry.openai_schemas(),
            Provider::Anthropic | Provider::VertexaiAnthropic => registry.anthropic_schemas(),
        })
    }
    "codesearch" => {
        registry.register(Box::new(BashFind));
        registry.register(Box::new(BashRipgrep));
        registry.register(Box::new(BashRead));
        // same schema dispatch
    }
    "pythonexec" => {
        registry.register(Box::new(PythonExecute));
        // same
    }
    _ => None,
};
```

### ToolCall execution

Update `ToolCall` to work with the registry:

```rust
impl ToolCall {
    pub async fn execute(&mut self, registry: &ToolRegistry) -> Result<()> {
        let tool = registry.get(&self.name)
            .ok_or_else(|| anyhow!("unknown tool: {}", self.name))?;
        let args: Value = serde_json::from_str(&self.arguments)?;
        self.result = Some(tool.execute(args).await);
        self.executed = true;
        Ok(())
    }
}
```

### Schema format reference

**OpenAI format** (from `callable_to_openai_schema`):
```json
{
  "type": "function",
  "function": {
    "name": "search_web",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": { "query": { "type": "string" } },
      "required": ["query"]
    }
  }
}
```

**Anthropic format** (from `callable_to_anthropic_schema`):
```json
{
  "name": "search_web",
  "description": "...",
  "input_schema": {
    "type": "object",
    "properties": { "query": { "type": "string" } },
    "required": ["query"]
  }
}
```

Only difference: OpenAI wraps in `{type: "function", function: {...}}` and uses `parameters`. Anthropic uses `input_schema` at top level.

## Verification

1. Register a mock tool, get OpenAI schema, verify JSON structure
2. Register a mock tool, get Anthropic schema, verify JSON structure
3. Execute a mock tool with JSON args, verify result comes back
4. Registry lookup for unknown tool returns None
5. Integration: wire into `invoke_llm` from Phase 3, verify tool call → execute → result → feed back cycle works with a real LLM + a simple mock tool
