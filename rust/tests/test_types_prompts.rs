use buzzllm::types::{LlmOptions, StreamResponse, StreamResponseType, ToolCallData};
use serde_json::json;

#[test]
fn test_llm_options_defaults() {
    let opts = LlmOptions::default();
    assert_eq!(opts.max_tokens, Some(8192));
    assert_eq!(opts.temperature, 0.8);
    assert!(!opts.think);
    assert_eq!(opts.max_infer_iters, 10);
    assert!(opts.tools.is_none());
}

#[test]
fn test_llm_options_custom() {
    let opts = LlmOptions {
        model: "gpt-4".into(),
        url: "https://example.com".into(),
        api_key_name: Some("KEY".into()),
        max_tokens: Some(4096),
        temperature: 0.5,
        think: true,
        tools: Some(vec![json!({"type": "function"})]),
        max_infer_iters: 5,
    };
    assert_eq!(opts.model, "gpt-4");
    assert_eq!(opts.url, "https://example.com");
    assert_eq!(opts.api_key_name, Some("KEY".into()));
    assert_eq!(opts.max_tokens, Some(4096));
    assert_eq!(opts.temperature, 0.5);
    assert!(opts.think);
    assert_eq!(opts.max_infer_iters, 5);
    assert!(opts.tools.is_some());
}

#[test]
fn test_stream_response_new() {
    let sr = StreamResponse::new("id1", "hello", StreamResponseType::OutputText);
    assert_eq!(sr.id, "id1");
    assert_eq!(sr.delta, "hello");
    assert_eq!(sr.response_type, StreamResponseType::OutputText);
}

#[test]
fn test_stream_response_to_json() {
    let sr = StreamResponse::new("xyz", "hi", StreamResponseType::OutputText);
    let json_str = sr.to_json();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed["id"], "xyz");
    assert_eq!(parsed["delta"], "hi");
    assert_eq!(parsed["type"], "output_text");
}

#[test]
fn test_stream_response_escapes_quotes() {
    let sr = StreamResponse::new("", "he said \"hello\"", StreamResponseType::OutputText);
    let json_str = sr.to_json();
    assert!(json_str.contains("he said \\\"hello\\\""));
}

#[test]
fn test_stream_response_types_display() {
    assert_eq!(StreamResponseType::ResponseStart.to_string(), "response_start");
    assert_eq!(StreamResponseType::OutputText.to_string(), "output_text");
    assert_eq!(StreamResponseType::ReasoningContent.to_string(), "reasoning_content");
    assert_eq!(StreamResponseType::ToolCall.to_string(), "tool_call");
    assert_eq!(StreamResponseType::ToolResult.to_string(), "tool_result");
    assert_eq!(StreamResponseType::BlockEnd.to_string(), "block_end");
    assert_eq!(StreamResponseType::ResponseEnd.to_string(), "response_end");
}

#[test]
fn test_tool_call_data_new() {
    let tc = ToolCallData::new("id1", "test_func");
    assert_eq!(tc.id, "id1");
    assert_eq!(tc.name, "test_func");
    assert!(tc.arguments.is_empty());
    assert!(!tc.executed);
    assert!(tc.result.is_none());
}

#[test]
fn test_tool_call_data_execute() {
    let mut tc = ToolCallData::new("id1", "func");
    tc.arguments = "{\"key\":\"val\"}".into();
    tc.result = Some(json!({"ok": true}));
    tc.executed = true;
    assert_eq!(tc.arguments, "{\"key\":\"val\"}");
    assert!(tc.executed);
    assert_eq!(tc.result, Some(json!({"ok": true})));
}

// --- Prompt tests ---

#[test]
fn test_get_prompt_valid() {
    let p = buzzllm::prompts::get_prompt("websearch");
    assert!(p.is_some());
    assert!(!p.unwrap().is_empty());
}

#[test]
fn test_get_prompt_coding_exists() {
    let p = buzzllm::prompts::get_prompt("coding");
    assert!(p.is_some());
}

#[test]
fn test_get_prompt_invalid() {
    let p = buzzllm::prompts::get_prompt("nonexistent");
    assert!(p.is_none());
}

#[test]
fn test_prompt_names() {
    let names = buzzllm::prompts::prompt_names();
    assert!(names.contains(&"websearch"));
    assert!(names.contains(&"codesearch"));
    assert!(names.contains(&"coding"));
    assert!(names.contains(&"replace"));
    assert!(names.contains(&"generate"));
    assert!(names.contains(&"helpful"));
    assert!(names.contains(&"hackhub"));
}

#[test]
fn test_prompt_names_count() {
    let names = buzzllm::prompts::prompt_names();
    assert_eq!(names.len(), 7);
}
