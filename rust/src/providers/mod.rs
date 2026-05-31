pub mod openai_chat;
pub mod openai_responses;
pub mod anthropic;
pub mod vertexai_anthropic;

use std::collections::HashMap;
use crate::types::{LlmOptions, RequestArgs, StreamResponse, ToolCallData};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenaiChat,
    OpenaiResponses,
    Anthropic,
    VertexaiAnthropic,
}

impl Provider {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "openai-chat" => Ok(Self::OpenaiChat),
            "openai-responses" => Ok(Self::OpenaiResponses),
            "anthropic" => Ok(Self::Anthropic),
            "vertexai-anthropic" => Ok(Self::VertexaiAnthropic),
            _ => Err(anyhow::anyhow!("unknown provider: {}", s)),
        }
    }

    pub fn make_request_args(&self, opts: &LlmOptions, prompt: &str, system_prompt: &str) -> anyhow::Result<RequestArgs> {
        match self {
            Self::OpenaiChat => Ok(openai_chat::make_request_args(opts, prompt, system_prompt)),
            Self::OpenaiResponses => openai_responses::make_request_args(opts, prompt, system_prompt),
            Self::Anthropic => Ok(anthropic::make_request_args(opts, prompt, system_prompt)),
            Self::VertexaiAnthropic => vertexai_anthropic::make_request_args(opts, prompt, system_prompt),
        }
    }

    pub fn parse_sse_line(
        &self,
        line: &str,
        message_started: bool,
        tool_calls: &mut HashMap<String, ToolCallData>,
        current_tool_call_id: &mut String,
    ) -> Vec<StreamResponse> {
        match self {
            Self::OpenaiChat => openai_chat::parse_sse_line(line, message_started, tool_calls, current_tool_call_id),
            Self::OpenaiResponses => openai_responses::parse_sse_line(line, message_started),
            Self::Anthropic | Self::VertexaiAnthropic => {
                anthropic::parse_sse_line(line, message_started, tool_calls, current_tool_call_id)
            }
        }
    }

    pub fn assemble_tool_messages(
        &self,
        messages: &mut Vec<serde_json::Value>,
        tool_calls: &HashMap<String, ToolCallData>,
    ) {
        match self {
            Self::OpenaiChat | Self::OpenaiResponses => {
                openai_chat::assemble_tool_messages(messages, tool_calls)
            }
            Self::Anthropic | Self::VertexaiAnthropic => {
                anthropic::assemble_tool_messages(messages, tool_calls)
            }
        }
    }

    pub fn is_anthropic_format(&self) -> bool {
        matches!(self, Self::Anthropic | Self::VertexaiAnthropic)
    }
}
