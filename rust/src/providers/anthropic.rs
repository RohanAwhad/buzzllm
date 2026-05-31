use std::collections::HashMap;
use serde_json::json;
use crate::types::{LlmOptions, RequestArgs, StreamResponse, StreamResponseType, ToolCallData};

pub fn make_request_args(opts: &LlmOptions, prompt: &str, system_prompt: &str) -> RequestArgs {
    let mut data = json!({
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "model": opts.model,
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
    if let Some(ref key_name) = opts.api_key_name {
        if let Ok(api_key) = std::env::var(key_name) {
            headers.insert("x-api-key".to_string(), api_key);
            headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
        }
    }

    RequestArgs { data, headers }
}

pub fn parse_sse_line(
    line: &str,
    _message_started: bool,
    tool_calls: &mut HashMap<String, ToolCallData>,
    current_tool_call_id: &mut String,
) -> Vec<StreamResponse> {
    let mut responses = Vec::new();

    // Skip event lines
    if line.starts_with("event: ") {
        return responses;
    }

    if !line.starts_with("data: ") {
        return responses;
    }
    let data_content = &line[6..];

    let chunk_data: serde_json::Value = match serde_json::from_str(data_content) {
        Ok(v) => v,
        Err(_) => return responses,
    };

    let event_type = chunk_data.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match event_type {
        "message_start" => {
            let message_id = chunk_data
                .get("message")
                .and_then(|m| m.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            responses.push(StreamResponse::new(message_id, "", StreamResponseType::ResponseStart));
        }

        "content_block_start" => {
            let content_block = chunk_data.get("content_block").unwrap_or(&serde_json::Value::Null);
            if content_block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                let tool_id = content_block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let tool_name = content_block.get("name").and_then(|v| v.as_str()).unwrap_or("");

                *current_tool_call_id = tool_id.to_string();
                tool_calls.insert(
                    tool_id.to_string(),
                    ToolCallData::new(tool_id, tool_name),
                );

                let content = format!("Function: {}\n", tool_name);
                responses.push(StreamResponse::new("", content, StreamResponseType::ToolCall));
            }
        }

        "content_block_delta" => {
            let delta = chunk_data.get("delta").unwrap_or(&serde_json::Value::Null);

            // Thinking content
            if let Some(thinking) = delta.get("thinking").and_then(|v| v.as_str()) {
                if !thinking.is_empty() {
                    responses.push(StreamResponse::new("", thinking, StreamResponseType::ReasoningContent));
                }
            }
            // Regular text
            else if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    responses.push(StreamResponse::new("", text, StreamResponseType::OutputText));
                }
            }
            // Tool input JSON delta
            else if let Some(partial_json) = delta.get("partial_json").and_then(|v| v.as_str()) {
                if !partial_json.is_empty() {
                    if let Some(entry) = tool_calls.get_mut(current_tool_call_id.as_str()) {
                        entry.arguments.push_str(partial_json);
                    }
                    responses.push(StreamResponse::new("", partial_json, StreamResponseType::ToolCall));
                }
            }
        }

        "message_stop" => {
            responses.push(StreamResponse::new("", "", StreamResponseType::BlockEnd));
        }

        _ => {}
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

    // Assistant message with tool_use content blocks
    let content: Vec<serde_json::Value> = tool_calls.values().map(|tc| {
        let input: serde_json::Value = serde_json::from_str(&tc.arguments)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        json!({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": input,
        })
    }).collect();

    messages.push(json!({"role": "assistant", "content": content}));

    // User message with tool_result content blocks
    let results: Vec<serde_json::Value> = tool_calls.values().map(|tc| {
        json!({
            "type": "tool_result",
            "tool_use_id": tc.id,
            "content": tc.result.as_ref().map(|r| r.to_string()).unwrap_or_default(),
        })
    }).collect();

    messages.push(json!({"role": "user", "content": results}));
}
