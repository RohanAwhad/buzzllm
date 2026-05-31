use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct LlmOptions {
    pub model: String,
    pub url: String,
    pub api_key_name: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: f64,
    pub think: bool,
    pub tools: Option<Vec<serde_json::Value>>,
    pub max_infer_iters: u32,
}

impl Default for LlmOptions {
    fn default() -> Self {
        Self {
            model: String::new(),
            url: String::new(),
            api_key_name: None,
            max_tokens: Some(8192),
            temperature: 0.8,
            think: false,
            tools: None,
            max_infer_iters: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestArgs {
    pub data: serde_json::Value,
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamResponseType {
    ResponseStart,
    OutputText,
    ReasoningContent,
    ToolCall,
    ToolResult,
    BlockEnd,
    ResponseEnd,
}

impl fmt::Display for StreamResponseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResponseStart => write!(f, "response_start"),
            Self::OutputText => write!(f, "output_text"),
            Self::ReasoningContent => write!(f, "reasoning_content"),
            Self::ToolCall => write!(f, "tool_call"),
            Self::ToolResult => write!(f, "tool_result"),
            Self::BlockEnd => write!(f, "block_end"),
            Self::ResponseEnd => write!(f, "response_end"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResponse {
    pub id: String,
    pub delta: String,
    #[serde(rename = "type")]
    pub response_type: StreamResponseType,
}

impl StreamResponse {
    pub fn new(id: impl Into<String>, delta: impl Into<String>, response_type: StreamResponseType) -> Self {
        Self {
            id: id.into(),
            delta: delta.into(),
            response_type,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct ToolCallData {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub executed: bool,
    pub result: Option<serde_json::Value>,
}

impl ToolCallData {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: String::new(),
            executed: false,
            result: None,
        }
    }
}
