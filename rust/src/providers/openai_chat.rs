use crate::providers::{LlmClient, OpenAIChatClient, ToolSchemaFormat};
use crate::types::{LlmOptions, RequestArgs, StreamResponse, StreamResponseType, ToolCallData};
use serde_json::json;
use std::collections::HashMap;

const REASONING_MODELS: &[&str] = &[
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "o4-mini",
    "o3",
    "o3-pro",
    "gpt-5-pro",
];

fn is_reasoning_model(model: &str) -> bool {
    REASONING_MODELS.contains(&model) || model.starts_with("gpt-5")
}

impl LlmClient for OpenAIChatClient {
    fn build_request(
        &self,
        opts: &LlmOptions,
        prompt: &str,
        system_prompt: &str,
    ) -> anyhow::Result<RequestArgs> {
        Ok(make_request_args(opts, prompt, system_prompt))
    }

    fn parse_sse_line(
        &self,
        line: &str,
        message_started: bool,
        tool_calls: &mut HashMap<String, ToolCallData>,
        current_tool_call_id: &mut String,
    ) -> Vec<StreamResponse> {
        parse_sse_line(line, message_started, tool_calls, current_tool_call_id)
    }

    fn assemble_tool_messages(
        &self,
        messages: &mut Vec<serde_json::Value>,
        tool_calls: &HashMap<String, ToolCallData>,
    ) {
        assemble_tool_messages(messages, tool_calls);
    }

    fn default_api_url(&self, _model: &str) -> String {
        "https://api.openai.com/v1/chat/completions".into()
    }

    fn tool_schema_format(&self) -> ToolSchemaFormat {
        ToolSchemaFormat::OpenAI
    }
}

pub fn make_request_args(opts: &LlmOptions, prompt: &str, system_prompt: &str) -> RequestArgs {
    let mut data = json!({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "model": opts.model,
        "stream": true,
    });

    if is_reasoning_model(&opts.model) {
        data["messages"][0]["role"] = json!("developer");
        data["response_format"] = json!({"type": "text"});
        let effort = if opts.model == "gpt-5.1" && !opts.think {
            "none"
        } else {
            "high"
        };
        data["reasoning_effort"] = json!(effort);
    } else {
        data["temperature"] = json!(opts.temperature);
        data["max_tokens"] = json!(opts.max_tokens.unwrap_or(8192));
    }

    if let Some(ref tools) = opts.tools {
        data["tools"] = json!(tools);
    }

    let mut headers = HashMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    if let Some(ref key_name) = opts.api_key_name {
        if let Ok(api_key) = std::env::var(key_name) {
            headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
        }
    }

    RequestArgs { data, headers }
}

pub fn parse_sse_line(
    line: &str,
    message_started: bool,
    tool_calls: &mut HashMap<String, ToolCallData>,
    current_tool_call_id: &mut String,
) -> Vec<StreamResponse> {
    let mut responses = Vec::new();

    if !line.starts_with("data: ") {
        return responses;
    }
    let data_content = &line[6..];

    if data_content == "[DONE]" {
        responses.push(StreamResponse::new("", "", StreamResponseType::BlockEnd));
        return responses;
    }

    let chunk_data: serde_json::Value = match serde_json::from_str(data_content) {
        Ok(v) => v,
        Err(_) => return responses,
    };

    if !message_started {
        if let Some(id) = chunk_data.get("id").and_then(|v| v.as_str()) {
            if !id.is_empty() {
                responses.push(StreamResponse::new(
                    id,
                    "",
                    StreamResponseType::ResponseStart,
                ));
            }
        }
    }

    let choices = match chunk_data.get("choices").and_then(|v| v.as_array()) {
        Some(c) if !c.is_empty() => c,
        _ => return responses,
    };

    let choice = &choices[0];
    let delta = match choice.get("delta") {
        Some(d) => d,
        None => return responses,
    };

    if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
        if !content.is_empty() {
            responses.push(StreamResponse::new(
                "",
                content,
                StreamResponseType::OutputText,
            ));
        }
    }

    let reasoning = delta
        .get("reasoning")
        .and_then(|v| v.as_str())
        .or_else(|| delta.get("reasoning_content").and_then(|v| v.as_str()));
    if let Some(r) = reasoning {
        if !r.is_empty() {
            responses.push(StreamResponse::new(
                "",
                r,
                StreamResponseType::ReasoningContent,
            ));
        }
    }

    if let Some(tc_array) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tc_array {
            let mut tool_call_content = String::new();

            if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                *current_tool_call_id = id.to_string();
                tool_calls.insert(id.to_string(), ToolCallData::new(id, ""));
            }

            if let Some(function) = tc.get("function") {
                if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
                    if !name.is_empty() {
                        if let Some(entry) = tool_calls.get_mut(current_tool_call_id.as_str()) {
                            entry.name = name.to_string();
                        }
                        tool_call_content.push_str(&format!("\nFunction: {} ", name));
                    }
                }

                if let Some(args) = function.get("arguments").and_then(|v| v.as_str()) {
                    if !args.is_empty() {
                        if let Some(entry) = tool_calls.get_mut(current_tool_call_id.as_str()) {
                            entry.arguments.push_str(args);
                        }
                        tool_call_content.push_str(args);
                    }
                }
            }

            if !tool_call_content.is_empty() {
                responses.push(StreamResponse::new(
                    "",
                    tool_call_content,
                    StreamResponseType::ToolCall,
                ));
            }
        }
    }

    responses
}

pub fn assemble_tool_messages(
    messages: &mut Vec<serde_json::Value>,
    tool_calls: &HashMap<String, ToolCallData>,
) {
    if tool_calls.is_empty() {
        return;
    }

    let tool_calls_list: Vec<serde_json::Value> = tool_calls
        .values()
        .map(|tc| {
            json!({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
            })
        })
        .collect();

    messages.push(json!({"role": "assistant", "tool_calls": tool_calls_list}));

    for tc in tool_calls.values() {
        messages.push(json!({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": tc.result.as_ref().map(|r| r.to_string()).unwrap_or_default(),
        }));
    }
}
