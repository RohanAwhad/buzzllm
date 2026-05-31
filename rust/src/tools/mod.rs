pub mod codesearch;
pub mod websearch;
pub mod pythonexec;

use std::collections::HashMap;
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name as it appears in the API schema
    fn name(&self) -> &str;

    /// JSON schema for this tool in OpenAI function-calling format
    fn openai_schema(&self) -> Value;

    /// JSON schema for this tool in Anthropic tool format
    fn anthropic_schema(&self) -> Value;

    /// Execute the tool with JSON arguments, return JSON result
    async fn execute(&self, args: Value) -> Value;

    /// Cleanup any resources held by this tool (default: no-op)
    async fn cleanup(&self) {}
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn openai_schemas(&self) -> Vec<Value> {
        self.tools.values().map(|t| t.openai_schema()).collect()
    }

    pub fn anthropic_schemas(&self) -> Vec<Value> {
        self.tools.values().map(|t| t.anthropic_schema()).collect()
    }

    /// Cleanup any stateful tools (e.g. pythonexec container)
    pub async fn cleanup(&self) {
        for tool in self.tools.values() {
            tool.cleanup().await;
        }
    }
}
