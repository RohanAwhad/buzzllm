use buzzllm::output::print_to_writer;
use buzzllm::types::{StreamResponse, StreamResponseType};

fn capture(data: &StreamResponse, sse: bool, brief: bool) -> String {
    let mut buf = Vec::new();
    print_to_writer(data, sse, brief, &mut buf);
    String::from_utf8(buf).unwrap()
}

#[test]
fn test_output_text_normal() {
    let sr = StreamResponse::new("", "hello", StreamResponseType::OutputText);
    assert_eq!(capture(&sr, false, false), "hello");
}

#[test]
fn test_output_text_sse() {
    let sr = StreamResponse::new("", "hi", StreamResponseType::OutputText);
    let out = capture(&sr, true, false);
    assert!(out.contains("event: output_text"));
    assert!(out.contains("data: {\"id\":\"\",\"delta\":\"hi\",\"type\":\"output_text\"}"));
}

#[test]
fn test_output_response_start() {
    let sr = StreamResponse::new("id1", "", StreamResponseType::ResponseStart);
    assert_eq!(capture(&sr, false, false), "");
}

#[test]
fn test_output_response_end_default() {
    let sr = StreamResponse::new("", "", StreamResponseType::ResponseEnd);
    let out = capture(&sr, false, false);
    assert!(out.contains("=== [ DONE ] ==="));
}

#[test]
fn test_output_response_end_sse() {
    let sr = StreamResponse::new("", "", StreamResponseType::ResponseEnd);
    let out = capture(&sr, true, false);
    assert!(out.contains("event: response_end"));
    assert!(out.contains("data: {\"id\":\"\",\"delta\":\"\",\"type\":\"response_end\"}"));
}

#[test]
fn test_output_block_end() {
    let sr = StreamResponse::new("", "", StreamResponseType::BlockEnd);
    assert_eq!(capture(&sr, false, false), "\n");
}

#[test]
fn test_output_tool_call() {
    let sr = StreamResponse::new("", "fn: test", StreamResponseType::ToolCall);
    let out = capture(&sr, false, false);
    assert!(out.contains("fn: test"));
}

#[test]
fn test_output_tool_result() {
    let sr = StreamResponse::new("", "result", StreamResponseType::ToolResult);
    let out = capture(&sr, false, false);
    assert!(out.contains("result"));
}

#[test]
fn test_output_reasoning_content() {
    let sr = StreamResponse::new("", "thinking...", StreamResponseType::ReasoningContent);
    let out = capture(&sr, false, false);
    assert!(out.contains("thinking..."));
}

#[test]
fn test_output_brief_hides_tool_call() {
    let sr = StreamResponse::new("", "hidden", StreamResponseType::ToolCall);
    assert_eq!(capture(&sr, false, true), "");
}

#[test]
fn test_output_brief_hides_tool_result() {
    let sr = StreamResponse::new("", "hidden", StreamResponseType::ToolResult);
    assert_eq!(capture(&sr, false, true), "");
}

#[test]
fn test_output_brief_shows_text() {
    let sr = StreamResponse::new("", "visible", StreamResponseType::OutputText);
    assert_eq!(capture(&sr, false, true), "visible");
}
