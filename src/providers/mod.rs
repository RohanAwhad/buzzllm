pub mod anthropic;
pub mod openai_chat;
pub mod openai_responses;
pub mod vertexai_anthropic;

use crate::types::{LlmOptions, RequestArgs, StreamResponse, ToolCallData};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSchemaFormat {
    OpenAI,
    Anthropic,
}

pub trait LlmClient {
    fn build_request(
        &self,
        opts: &LlmOptions,
        prompt: &str,
        system_prompt: &str,
    ) -> anyhow::Result<RequestArgs>;

    fn parse_sse_line(
        &self,
        line: &str,
        message_started: bool,
        tool_calls: &mut HashMap<String, ToolCallData>,
        current_tool_call_id: &mut String,
    ) -> Vec<StreamResponse>;

    fn assemble_tool_messages(
        &self,
        messages: &mut Vec<serde_json::Value>,
        tool_calls: &HashMap<String, ToolCallData>,
    );

    fn default_api_url(&self, model: &str) -> String;

    fn tool_schema_format(&self) -> ToolSchemaFormat;
}

pub struct OpenAIChatClient;
pub struct OpenAIResponsesClient;
pub struct AnthropicClient;
pub struct VertexAIAnthropicClient;

pub fn create_client(name: &str) -> anyhow::Result<Box<dyn LlmClient>> {
    match name {
        "openai-chat" => Ok(Box::new(OpenAIChatClient)),
        "openai-responses" => Ok(Box::new(OpenAIResponsesClient)),
        "anthropic" => Ok(Box::new(AnthropicClient)),
        "vertexai-anthropic" => Ok(Box::new(VertexAIAnthropicClient)),
        _ => Err(anyhow::anyhow!("unknown provider: {}", name)),
    }
}
