use buzzllm::providers::{self, ToolSchemaFormat};
use buzzllm::types::{LlmOptions, StreamResponseType, ToolCallData};
use std::collections::HashMap;

fn test_opts() -> LlmOptions {
    LlmOptions {
        model: "gpt-4.1-mini".into(),
        url: String::new(),
        api_key_name: None,
        max_tokens: Some(8192),
        temperature: 0.8,
        think: false,
        tools: None,
        max_infer_iters: 10,
    }
}

fn test_opts_with_tools() -> LlmOptions {
    let mut opts = test_opts();
    opts.tools = Some(vec![
        serde_json::json!({"type": "function", "function": {"name": "test_tool", "parameters": {}}}),
    ]);
    opts
}

fn test_opts_with_api_key() -> LlmOptions {
    let mut opts = test_opts();
    opts.api_key_name = Some("TEST_API_KEY".into());
    opts
}

#[test]
fn test_create_client_valid() {
    assert!(providers::create_client("openai-chat").is_ok());
    assert!(providers::create_client("openai-responses").is_ok());
    assert!(providers::create_client("anthropic").is_ok());
    assert!(providers::create_client("vertexai-anthropic").is_ok());
}

#[test]
fn test_create_client_invalid() {
    assert!(providers::create_client("unknown").is_err());
}

#[test]
fn test_tool_schema_format_openai() {
    let c = providers::create_client("openai-chat").unwrap();
    assert_eq!(c.tool_schema_format(), ToolSchemaFormat::OpenAI);
}

#[test]
fn test_tool_schema_format_anthropic() {
    let c = providers::create_client("anthropic").unwrap();
    assert_eq!(c.tool_schema_format(), ToolSchemaFormat::Anthropic);
}

#[test]
fn test_tool_schema_format_vertexai() {
    let c = providers::create_client("vertexai-anthropic").unwrap();
    assert_eq!(c.tool_schema_format(), ToolSchemaFormat::Anthropic);
}

// ============================================================
// OpenAI Chat — Request Builder
// ============================================================

#[test]
fn test_openai_chat_basic_structure() {
    let c = providers::create_client("openai-chat").unwrap();
    let args = c.build_request(&test_opts(), "hello", "sys").unwrap();

    assert_eq!(args.data["model"], "gpt-4.1-mini");
    assert_eq!(args.data["stream"], true);
    assert_eq!(args.data["messages"][0]["role"], "system");
    assert_eq!(args.data["messages"][0]["content"], "sys");
    assert_eq!(args.data["messages"][1]["role"], "user");
    assert_eq!(args.data["messages"][1]["content"], "hello");
    assert_eq!(args.data["temperature"], 0.8);
    assert_eq!(args.data["max_tokens"], 8192);
    assert!(args.headers.contains_key("Content-Type"));
}

#[test]
fn test_openai_chat_bearer_auth() {
    unsafe {
        std::env::set_var("TEST_API_KEY", "sk-test123");
    }
    let c = providers::create_client("openai-chat").unwrap();
    let args = c
        .build_request(&test_opts_with_api_key(), "hi", "sys")
        .unwrap();
    assert_eq!(
        args.headers.get("Authorization").unwrap(),
        "Bearer sk-test123"
    );
}

#[test]
fn test_openai_chat_no_auth_without_key() {
    let c = providers::create_client("openai-chat").unwrap();
    let args = c.build_request(&test_opts(), "hi", "sys").unwrap();
    assert!(!args.headers.contains_key("Authorization"));
}

#[test]
fn test_openai_chat_tools_included() {
    let c = providers::create_client("openai-chat").unwrap();
    let args = c
        .build_request(&test_opts_with_tools(), "hi", "sys")
        .unwrap();
    let tools = args.data["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
}

#[test]
fn test_openai_chat_reasoning_model() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut opts = test_opts();
    opts.model = "o3".into();
    let args = c.build_request(&opts, "hi", "sys").unwrap();
    assert_eq!(args.data["messages"][0]["role"], "developer");
    assert_eq!(args.data["response_format"]["type"], "text");
    assert!(args.data.get("temperature").is_none());
    assert_eq!(args.data["reasoning_effort"], "high");
}

#[test]
fn test_openai_chat_gpt51_no_think() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut opts = test_opts();
    opts.model = "gpt-5.1".into();
    opts.think = false;
    let args = c.build_request(&opts, "hi", "sys").unwrap();
    assert_eq!(args.data["reasoning_effort"], "none");
}

#[test]
fn test_openai_chat_gpt5_mini_reasoning_effort() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut opts = test_opts();
    opts.model = "gpt-5-mini".into();
    let args = c.build_request(&opts, "hi", "sys").unwrap();
    assert_eq!(args.data["reasoning_effort"], "high");
}

// ============================================================
// OpenAI Chat — SSE Parser
// ============================================================

#[test]
fn test_openai_chat_ignores_non_data_lines() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let res = c.parse_sse_line("not a data line", false, &mut tc, &mut cid);
    assert!(res.is_empty());
}

#[test]
fn test_openai_chat_done_marker() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let res = c.parse_sse_line("data: [DONE]", false, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::BlockEnd);
}

#[test]
fn test_openai_chat_response_start() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"id":"chatcmpl-123","choices":[{"delta":{}}]}"#;
    let res = c.parse_sse_line(line, false, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::ResponseStart);
    assert_eq!(res[0].id, "chatcmpl-123");
}

#[test]
fn test_openai_chat_content_delta() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"}}]}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::OutputText);
    assert_eq!(res[0].delta, "Hello");
}

#[test]
fn test_openai_chat_reasoning_content() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"id":"x","choices":[{"delta":{"reasoning":"Let me think..."}}]}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::ReasoningContent);
    assert_eq!(res[0].delta, "Let me think...");
}

#[test]
fn test_openai_chat_reasoning_content_alt_key() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"id":"x","choices":[{"delta":{"reasoning_content":"Hmm"}}]}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].delta, "Hmm");
}

#[test]
fn test_openai_chat_tool_call_start() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"id":"x","choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"name":"search_web"}}]}}]}"#;
    let _res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(cid, "call_1");
    assert!(tc.contains_key("call_1"));
    assert_eq!(tc["call_1"].name, "search_web");
}

#[test]
fn test_openai_chat_tool_call_accumulation() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = "call_1".into();
    tc.insert("call_1".into(), ToolCallData::new("call_1", "search_web"));

    let line = r#"data: {"id":"x","choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"arguments":"some args"}}]}}]}"#;
    let _res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(tc["call_1"].arguments, "some args");
    assert!(!_res.is_empty());
}

#[test]
fn test_openai_chat_empty_choices() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"id":"x","choices":[]}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert!(res.is_empty());
}

#[test]
fn test_openai_chat_invalid_json() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let res = c.parse_sse_line("data: not json", true, &mut tc, &mut cid);
    assert!(res.is_empty());
}

// ============================================================
// OpenAI Chat — Message Assembler
// ============================================================

#[test]
fn test_openai_chat_assemble_empty_noop() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "hi"})];
    let tc = HashMap::new();
    c.assemble_tool_messages(&mut messages, &tc);
    assert_eq!(messages.len(), 1);
}

#[test]
fn test_openai_chat_assemble_single_tool() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "hi"})];
    let mut tc = HashMap::new();
    let mut tcd = ToolCallData::new("call_1", "search_web");
    tcd.arguments = r#"{"query":"test"}"#.into();
    tcd.result = Some(serde_json::json!({"results": []}));
    tc.insert("call_1".into(), tcd);

    c.assemble_tool_messages(&mut messages, &tc);
    assert_eq!(messages.len(), 3);

    assert_eq!(messages[1]["role"], "assistant");
    let calls = messages[1]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["id"], "call_1");

    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call_1");
}

#[test]
fn test_openai_chat_assemble_multi_tool() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "hi"})];
    let mut tc = HashMap::new();

    let mut tcd1 = ToolCallData::new("call_1", "search_web");
    tcd1.arguments = r#"{"query":"test"}"#.into();
    tcd1.result = Some(serde_json::json!({"results": []}));
    tc.insert("call_1".into(), tcd1);

    let mut tcd2 = ToolCallData::new("call_2", "bash_read");
    tcd2.arguments = r#"{"path":"src/main.rs"}"#.into();
    tcd2.result = Some(serde_json::json!("file contents"));
    tc.insert("call_2".into(), tcd2);

    c.assemble_tool_messages(&mut messages, &tc);
    assert_eq!(messages.len(), 4);

    assert_eq!(messages[1]["role"], "assistant");
    let calls = messages[1]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2);
    let call_ids: Vec<&str> = calls.iter().map(|c| c["id"].as_str().unwrap()).collect();
    assert!(call_ids.contains(&"call_1"));
    assert!(call_ids.contains(&"call_2"));

    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call_1");
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "call_2");
}

#[test]
fn test_openai_chat_assemble_preserves_existing() {
    let c = providers::create_client("openai-chat").unwrap();
    let mut messages: Vec<serde_json::Value> = vec![
        serde_json::json!({"role": "system", "content": "You are helpful."}),
        serde_json::json!({"role": "user", "content": "hi"}),
    ];
    let orig_len = messages.len();
    let mut tc = HashMap::new();
    let mut tcd = ToolCallData::new("call_1", "search_web");
    tcd.arguments = r#"{"query":"test"}"#.into();
    tcd.result = Some(serde_json::json!({"results": []}));
    tc.insert("call_1".into(), tcd);

    c.assemble_tool_messages(&mut messages, &tc);
    assert_eq!(messages.len(), orig_len + 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
}
// Anthropic — Request Builder
// ============================================================

fn anthropic_env_setup() {
    unsafe {
        std::env::set_var("TEST_API_KEY", "sk-ant-test123");
    }
}

#[test]
fn test_anthropic_basic_structure() {
    anthropic_env_setup();
    let c = providers::create_client("anthropic").unwrap();
    let args = c
        .build_request(&test_opts_with_api_key(), "hello", "sys prompt")
        .unwrap();

    assert_eq!(args.data["system"], "sys prompt");
    assert_eq!(args.data["messages"][0]["role"], "user");
    assert_eq!(args.data["messages"][0]["content"], "hello");
    assert_eq!(args.data["model"], "gpt-4.1-mini");
    assert_eq!(args.data["stream"], true);
    assert_eq!(args.data["max_tokens"], 8192);
}

#[test]
fn test_anthropic_xapi_key_header() {
    anthropic_env_setup();
    let c = providers::create_client("anthropic").unwrap();
    let args = c
        .build_request(&test_opts_with_api_key(), "hi", "sys")
        .unwrap();
    assert_eq!(args.headers.get("x-api-key").unwrap(), "sk-ant-test123");
    assert!(args.headers.contains_key("anthropic-version"));
}

#[test]
fn test_anthropic_think_mode() {
    anthropic_env_setup();
    let c = providers::create_client("anthropic").unwrap();
    let mut opts = test_opts_with_api_key();
    opts.think = true;
    let args = c.build_request(&opts, "hi", "sys").unwrap();
    assert_eq!(args.data["max_tokens"], 32000);
    assert_eq!(args.data["thinking"]["type"], "enabled");
    assert_eq!(args.data["thinking"]["budget_tokens"], 24000);
}

#[test]
fn test_anthropic_tools_included() {
    anthropic_env_setup();
    let c = providers::create_client("anthropic").unwrap();
    let args = c
        .build_request(&test_opts_with_tools(), "hi", "sys")
        .unwrap();
    let tools = args.data["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
}

// ============================================================
// Anthropic — SSE Parser
// ============================================================

#[test]
fn test_anthropic_skips_event_lines() {
    let c = providers::create_client("anthropic").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let res = c.parse_sse_line("event: ping", false, &mut tc, &mut cid);
    assert!(res.is_empty());
}

#[test]
fn test_anthropic_message_start() {
    let c = providers::create_client("anthropic").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"type":"message_start","message":{"id":"msg_123"}}"#;
    let res = c.parse_sse_line(line, false, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::ResponseStart);
    assert_eq!(res[0].id, "msg_123");
}

#[test]
fn test_anthropic_tool_use_block() {
    let c = providers::create_client("anthropic").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_1","name":"search_web"}}"#;
    let res = c.parse_sse_line(line, false, &mut tc, &mut cid);
    assert_eq!(cid, "toolu_1");
    assert!(tc.contains_key("toolu_1"));
    assert_eq!(tc["toolu_1"].name, "search_web");
    assert!(!res.is_empty());
}

#[test]
fn test_anthropic_text_delta() {
    let c = providers::create_client("anthropic").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"type":"content_block_delta","delta":{"text":"Hello"}}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::OutputText);
    assert_eq!(res[0].delta, "Hello");
}

#[test]
fn test_anthropic_thinking_delta() {
    let c = providers::create_client("anthropic").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"type":"content_block_delta","delta":{"thinking":"Let me consider..."}}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::ReasoningContent);
    assert_eq!(res[0].delta, "Let me consider...");
}

#[test]
fn test_anthropic_partial_json_accumulates() {
    let c = providers::create_client("anthropic").unwrap();
    let mut tc = HashMap::new();
    let mut cid = "toolu_1".into();
    tc.insert("toolu_1".into(), ToolCallData::new("toolu_1", "search_web"));

    let line = r#"data: {"type":"content_block_delta","delta":{"partial_json":"{\"q"}}"#;
    let _ = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(tc["toolu_1"].arguments, "{\"q");
}

#[test]
fn test_anthropic_message_stop() {
    let c = providers::create_client("anthropic").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"type":"message_stop"}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::BlockEnd);
}

#[test]
fn test_anthropic_invalid_json() {
    let c = providers::create_client("anthropic").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let res = c.parse_sse_line("data: {invalid", true, &mut tc, &mut cid);
    assert!(res.is_empty());
}

// ============================================================
// Anthropic — Message Assembler
// ============================================================

#[test]
fn test_anthropic_assemble_empty_noop() {
    let c = providers::create_client("anthropic").unwrap();
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "hi"})];
    let tc = HashMap::new();
    c.assemble_tool_messages(&mut messages, &tc);
    assert_eq!(messages.len(), 1);
}

#[test]
fn test_anthropic_assemble_single_tool() {
    let c = providers::create_client("anthropic").unwrap();
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "hi"})];
    let mut tc = HashMap::new();
    let mut tcd = ToolCallData::new("toolu_1", "search_web");
    tcd.arguments = r#"{"query":"rust"}"#.into();
    tcd.result = Some(serde_json::json!({"results": []}));
    tc.insert("toolu_1".into(), tcd);

    c.assemble_tool_messages(&mut messages, &tc);
    assert_eq!(messages.len(), 3);

    assert_eq!(messages[1]["role"], "assistant");
    let content = messages[1]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "tool_use");
    assert_eq!(content[0]["id"], "toolu_1");
    assert_eq!(content[0]["input"]["query"], "rust");

    assert_eq!(messages[2]["role"], "user");
    let results = messages[2]["content"].as_array().unwrap();
    assert_eq!(results[0]["type"], "tool_result");
    assert_eq!(results[0]["tool_use_id"], "toolu_1");
}

#[test]
fn test_anthropic_assemble_multi_tool() {
    let c = providers::create_client("anthropic").unwrap();
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "user", "content": "hi"})];
    let mut tc = HashMap::new();

    let mut tcd1 = ToolCallData::new("toolu_1", "search_web");
    tcd1.arguments = r#"{"query":"rust"}"#.into();
    tcd1.result = Some(serde_json::json!({"results": []}));
    tc.insert("toolu_1".into(), tcd1);

    let mut tcd2 = ToolCallData::new("toolu_2", "bash_read");
    tcd2.arguments = r#"{"path":"Cargo.toml"}"#.into();
    tcd2.result = Some(serde_json::json!("[package]"));
    tc.insert("toolu_2".into(), tcd2);

    c.assemble_tool_messages(&mut messages, &tc);
    assert_eq!(messages.len(), 3);

    assert_eq!(messages[1]["role"], "assistant");
    let content = messages[1]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    let use_ids: Vec<&str> = content
        .iter()
        .map(|b| b["id"].as_str().unwrap())
        .collect();
    assert!(use_ids.contains(&"toolu_1"));
    assert!(use_ids.contains(&"toolu_2"));

    assert_eq!(messages[2]["role"], "user");
    let results = messages[2]["content"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    let ids: Vec<&str> = results
        .iter()
        .map(|r| r["tool_use_id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"toolu_1"));
    assert!(ids.contains(&"toolu_2"));
}

#[test]
fn test_anthropic_assemble_preserves_existing() {
    let c = providers::create_client("anthropic").unwrap();
    let mut messages: Vec<serde_json::Value> = vec![
        serde_json::json!({"role": "user", "content": "hi"}),
    ];
    let orig_len = messages.len();
    let mut tc = HashMap::new();
    let mut tcd = ToolCallData::new("toolu_1", "search_web");
    tcd.arguments = r#"{"query":"rust"}"#.into();
    tcd.result = Some(serde_json::json!({"results": []}));
    tc.insert("toolu_1".into(), tcd);

    c.assemble_tool_messages(&mut messages, &tc);
    assert_eq!(messages.len(), orig_len + 2);
    assert_eq!(messages[0]["role"], "user");
}

// ============================================================
// OpenAI Responses — Request Builder
// ============================================================

#[test]
fn test_openai_responses_basic_structure() {
    let c = providers::create_client("openai-responses").unwrap();
    let args = c
        .build_request(&test_opts(), "hello", "sys prompt")
        .unwrap();

    assert_eq!(args.data["model"], "gpt-4.1-mini");
    assert_eq!(args.data["input"], "hello");
    assert_eq!(args.data["instructions"], "sys prompt");
    assert_eq!(args.data["stream"], true);
    assert_eq!(args.data["store"], false);
    assert_eq!(args.data["reasoning"]["effort"], "high");
}

#[test]
fn test_openai_responses_tools_not_supported() {
    let c = providers::create_client("openai-responses").unwrap();
    let result = c.build_request(&test_opts_with_tools(), "hi", "sys");
    assert!(result.is_err());
}

#[test]
fn test_openai_responses_empty_system_prompt() {
    let c = providers::create_client("openai-responses").unwrap();
    let args = c.build_request(&test_opts(), "hello", "").unwrap();
    assert!(args.data.get("instructions").is_none());
}

// ============================================================
// OpenAI Responses — SSE Parser
// ============================================================

#[test]
fn test_openai_responses_response_created() {
    let c = providers::create_client("openai-responses").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"type":"response.created","response":{"id":"resp_123"}}"#;
    let res = c.parse_sse_line(line, false, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::ResponseStart);
    assert_eq!(res[0].id, "resp_123");
}

#[test]
fn test_openai_responses_output_text_delta() {
    let c = providers::create_client("openai-responses").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"type":"response.output_text.delta","delta":"Hello"}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::OutputText);
    assert_eq!(res[0].delta, "Hello");
}

#[test]
fn test_openai_responses_reasoning_delta() {
    let c = providers::create_client("openai-responses").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"type":"response.reasoning_summary_text.delta","delta":"Summary"}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::ReasoningContent);
}

#[test]
fn test_openai_responses_completed() {
    let c = providers::create_client("openai-responses").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let line = r#"data: {"type":"response.completed"}"#;
    let res = c.parse_sse_line(line, true, &mut tc, &mut cid);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].response_type, StreamResponseType::BlockEnd);
}

#[test]
fn test_openai_responses_ignores_non_data() {
    let c = providers::create_client("openai-responses").unwrap();
    let mut tc = HashMap::new();
    let mut cid = String::new();
    let res = c.parse_sse_line("not data", false, &mut tc, &mut cid);
    assert!(res.is_empty());
}

// ============================================================
// VertexAI Anthropic — Request Builder
// ============================================================

#[test]
fn test_vertexai_anthropic_version_field() {
    let c = providers::create_client("vertexai-anthropic").unwrap();
    let mut opts = test_opts();
    opts.model = "claude-sonnet-4@default".into();
    let args = c.build_request(&opts, "hi", "sys").unwrap();
    assert_eq!(args.data["anthropic_version"], "vertex-2023-10-16");
    assert_eq!(args.data["system"], "sys");
}

#[test]
fn test_vertexai_model_in_body() {
    let c = providers::create_client("vertexai-anthropic").unwrap();
    let mut opts = test_opts();
    opts.model = "claude-opus-4-6@default".into();
    let args = c.build_request(&opts, "hi", "sys").unwrap();
    assert_eq!(args.data["messages"][0]["content"], "hi");
}

#[test]
fn test_vertexai_think_mode() {
    let c = providers::create_client("vertexai-anthropic").unwrap();
    let mut opts = test_opts();
    opts.think = true;
    let args = c.build_request(&opts, "hi", "sys").unwrap();
    assert_eq!(args.data["max_tokens"], 32000);
    assert_eq!(args.data["thinking"]["type"], "enabled");
}

// ============================================================
// LlmClient — default_api_url
// ============================================================

#[test]
fn test_default_api_url_openai_chat() {
    let c = providers::create_client("openai-chat").unwrap();
    let url = c.default_api_url("gpt-4.1-mini");
    assert!(url.contains("api.openai.com"));
    assert!(url.contains("chat/completions"));
}

#[test]
fn test_default_api_url_anthropic() {
    let c = providers::create_client("anthropic").unwrap();
    let url = c.default_api_url("claude-sonnet-4");
    assert!(url.contains("api.anthropic.com"));
}

#[test]
fn test_default_api_url_openai_responses() {
    let c = providers::create_client("openai-responses").unwrap();
    let url = c.default_api_url("gpt-4.1-mini");
    assert!(url.contains("responses"));
}

#[test]
fn test_default_api_url_vertexai() {
    let c = providers::create_client("vertexai-anthropic").unwrap();
    let url = c.default_api_url("any");
    assert!(url.is_empty());
}
