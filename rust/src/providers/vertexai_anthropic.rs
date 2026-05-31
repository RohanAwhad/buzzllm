use crate::providers::{LlmClient, ToolSchemaFormat, VertexAIAnthropicClient};
use crate::types::{LlmOptions, RequestArgs, StreamResponse, ToolCallData};
use serde_json::json;
use std::collections::HashMap;

impl LlmClient for VertexAIAnthropicClient {
    fn build_request(
        &self,
        opts: &LlmOptions,
        prompt: &str,
        system_prompt: &str,
    ) -> anyhow::Result<RequestArgs> {
        make_request_args(opts, prompt, system_prompt)
    }

    fn parse_sse_line(
        &self,
        line: &str,
        message_started: bool,
        tool_calls: &mut HashMap<String, ToolCallData>,
        current_tool_call_id: &mut String,
    ) -> Vec<StreamResponse> {
        super::anthropic::parse_sse_line(line, message_started, tool_calls, current_tool_call_id)
    }

    fn assemble_tool_messages(
        &self,
        messages: &mut Vec<serde_json::Value>,
        tool_calls: &HashMap<String, ToolCallData>,
    ) {
        super::anthropic::assemble_tool_messages(messages, tool_calls);
    }

    fn default_api_url(&self, _model: &str) -> String {
        String::new()
    }

    fn tool_schema_format(&self) -> ToolSchemaFormat {
        ToolSchemaFormat::Anthropic
    }
}

pub fn make_request_args(
    opts: &LlmOptions,
    prompt: &str,
    system_prompt: &str,
) -> anyhow::Result<RequestArgs> {
    let mut data = json!({
        "anthropic_version": "vertex-2023-10-16",
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
    });

    if opts.think {
        data["max_tokens"] = json!(32000);
        data["thinking"] = json!({"type": "enabled", "budget_tokens": 24000});
    } else {
        data["max_tokens"] = json!(opts.max_tokens.unwrap_or(8192));
    }

    if let Some(ref tools) = opts.tools {
        data["tools"] = json!(tools);
    }

    let mut headers = HashMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());

    let output = std::process::Command::new("gcloud")
        .args(["auth", "print-access-token"])
        .output()
        .map_err(|e| anyhow::anyhow!("failed to run gcloud: {}", e))?;

    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    headers.insert("Authorization".to_string(), format!("Bearer {}", token));

    Ok(RequestArgs { data, headers })
}
