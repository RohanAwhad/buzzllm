use crate::providers::{LlmClient, OpenAIResponsesClient, ToolSchemaFormat};
use crate::types::{LlmOptions, RequestArgs, StreamResponse, StreamResponseType, ToolCallData};
use serde_json::json;
use std::collections::HashMap;

impl LlmClient for OpenAIResponsesClient {
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
        _message_started: bool,
        _tool_calls: &mut HashMap<String, ToolCallData>,
        _current_tool_call_id: &mut String,
    ) -> Vec<StreamResponse> {
        parse_sse_line(line)
    }

    fn assemble_tool_messages(
        &self,
        _messages: &mut Vec<serde_json::Value>,
        _tool_calls: &HashMap<String, ToolCallData>,
    ) {
    }

    fn default_api_url(&self, _model: &str) -> String {
        "https://api.openai.com/v1/responses".into()
    }

    fn tool_schema_format(&self) -> ToolSchemaFormat {
        ToolSchemaFormat::OpenAI
    }
}

pub fn make_request_args(
    opts: &LlmOptions,
    prompt: &str,
    system_prompt: &str,
) -> anyhow::Result<RequestArgs> {
    let mut data = json!({
        "model": opts.model,
        "input": prompt,
        "stream": true,
        "store": false,
        "reasoning": {
            "effort": "high",
            "summary": "detailed",
        }
    });

    if !system_prompt.is_empty() {
        data["instructions"] = json!(system_prompt);
    }

    if opts.tools.is_some() {
        return Err(anyhow::anyhow!(
            "Tools with OpenAI Responses API has not yet been implemented"
        ));
    }

    let mut headers = HashMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    if let Some(ref key_name) = opts.api_key_name {
        if let Ok(api_key) = std::env::var(key_name) {
            headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
        }
    }

    Ok(RequestArgs { data, headers })
}

pub fn parse_sse_line(line: &str) -> Vec<StreamResponse> {
    let mut responses = Vec::new();

    if !line.starts_with("data: ") {
        return responses;
    }
    let data_content = &line[6..];

    let chunk_data: serde_json::Value = match serde_json::from_str(data_content) {
        Ok(v) => v,
        Err(_) => return responses,
    };

    let event_type = chunk_data
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match event_type {
        "response.created" => {
            let response_id = chunk_data
                .get("response")
                .and_then(|r| r.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            responses.push(StreamResponse::new(
                response_id,
                "",
                StreamResponseType::ResponseStart,
            ));
        }
        "response.output_text.delta" => {
            let delta_text = chunk_data
                .get("delta")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            responses.push(StreamResponse::new(
                "",
                delta_text,
                StreamResponseType::OutputText,
            ));
        }
        "response.reasoning_summary_text.delta" => {
            let delta_text = chunk_data
                .get("delta")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            responses.push(StreamResponse::new(
                "",
                delta_text,
                StreamResponseType::ReasoningContent,
            ));
        }
        "response.completed" => {
            responses.push(StreamResponse::new("", "", StreamResponseType::BlockEnd));
        }
        _ => {}
    }

    responses
}
