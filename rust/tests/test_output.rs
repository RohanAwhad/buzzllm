use buzzllm::types::{StreamResponse, StreamResponseType};
use buzzllm::output::print_to_stdout;

#[test]
fn test_output_text_normal() {
    let sr = StreamResponse::new("", "hello", StreamResponseType::OutputText);
    print_to_stdout(&sr, false, false);
}

#[test]
fn test_output_text_sse() {
    let sr = StreamResponse::new("", "hi", StreamResponseType::OutputText);
    print_to_stdout(&sr, true, false);
}

#[test]
fn test_output_response_start() {
    let sr = StreamResponse::new("id1", "", StreamResponseType::ResponseStart);
    print_to_stdout(&sr, false, false);
}

#[test]
fn test_output_response_end_default() {
    let sr = StreamResponse::new("", "", StreamResponseType::ResponseEnd);
    print_to_stdout(&sr, false, false);
}

#[test]
fn test_output_response_end_sse() {
    let sr = StreamResponse::new("", "", StreamResponseType::ResponseEnd);
    print_to_stdout(&sr, true, false);
}

#[test]
fn test_output_block_end() {
    let sr = StreamResponse::new("", "", StreamResponseType::BlockEnd);
    print_to_stdout(&sr, false, false);
}

#[test]
fn test_output_tool_call() {
    let sr = StreamResponse::new("", "fn: test", StreamResponseType::ToolCall);
    print_to_stdout(&sr, false, false);
}

#[test]
fn test_output_tool_result() {
    let sr = StreamResponse::new("", "result", StreamResponseType::ToolResult);
    print_to_stdout(&sr, false, false);
}

#[test]
fn test_output_reasoning_content() {
    let sr = StreamResponse::new("", "thinking...", StreamResponseType::ReasoningContent);
    print_to_stdout(&sr, false, false);
}

#[test]
fn test_output_brief_hides_tool_call() {
    let sr = StreamResponse::new("", "hidden", StreamResponseType::ToolCall);
    print_to_stdout(&sr, false, true);
}

#[test]
fn test_output_brief_hides_tool_result() {
    let sr = StreamResponse::new("", "hidden", StreamResponseType::ToolResult);
    print_to_stdout(&sr, false, true);
}

#[test]
fn test_output_brief_shows_text() {
    let sr = StreamResponse::new("", "visible", StreamResponseType::OutputText);
    print_to_stdout(&sr, false, true);
}
