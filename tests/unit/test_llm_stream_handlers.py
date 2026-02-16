import pytest
import json

from buzzllm.llm import (
    handle_openai_stream_response,
    handle_anthropic_stream_response,
    handle_openai_responses_stream_response,
    TOOL_CALLS,
    ToolCall,
)


class TestHandleOpenaiStreamResponse:
    def test_ignores_non_data_lines(self):
        results = list(handle_openai_stream_response("some random line", False))
        assert all(r is None for r in results)

    def test_handles_done_marker(self):
        results = list(handle_openai_stream_response("data: [DONE]", True))
        assert any(r is not None and r.type == "block_end" for r in results)

    def test_response_start(self):
        line = 'data: {"id":"chatcmpl-123","choices":[]}'
        results = list(handle_openai_stream_response(line, False))
        response_starts = [r for r in results if r and r.type == "response_start"]
        assert len(response_starts) == 1
        assert response_starts[0].id == "chatcmpl-123"

    def test_content_delta(self):
        line = 'data: {"id":"123","choices":[{"delta":{"content":"Hello"}}]}'
        results = list(handle_openai_stream_response(line, True))
        text_responses = [r for r in results if r and r.type == "output_text"]
        assert len(text_responses) == 1
        assert text_responses[0].delta == "Hello"

    def test_reasoning_content(self):
        line = 'data: {"id":"123","choices":[{"delta":{"reasoning":"thinking..."}}]}'
        results = list(handle_openai_stream_response(line, True))
        reasoning = [r for r in results if r and r.type == "reasoning_content"]
        assert len(reasoning) == 1

    def test_tool_call_start(self):
        line = 'data: {"id":"123","choices":[{"delta":{"tool_calls":[{"id":"call_abc","function":{"name":"search_web","arguments":""}}]}}]}'
        results = list(handle_openai_stream_response(line, True))

        tool_calls = [r for r in results if r and r.type == "tool_call"]
        assert len(tool_calls) == 1
        assert "call_abc" in TOOL_CALLS
        assert TOOL_CALLS["call_abc"].name == "search_web"

    def test_tool_call_arguments_accumulate(self):
        # First chunk with id and name
        line1 = 'data: {"id":"123","choices":[{"delta":{"tool_calls":[{"id":"call_xyz","function":{"name":"test","arguments":"{\\"q"}}]}}]}'
        list(handle_openai_stream_response(line1, True))

        # Second chunk with more arguments
        line2 = 'data: {"id":"123","choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\\": \\"value\\"}"}}]}}]}'
        list(handle_openai_stream_response(line2, True))

        assert "call_xyz" in TOOL_CALLS
        assert TOOL_CALLS["call_xyz"].arguments == '{"q": "value"}'

    def test_empty_choices_yields_none(self):
        line = 'data: {"id":"123","choices":[]}'
        results = list(handle_openai_stream_response(line, True))
        # Should not crash, may yield None
        assert all(r is None or r.type == "response_start" for r in results)

    def test_invalid_json_yields_none(self):
        line = "data: {invalid json}"
        results = list(handle_openai_stream_response(line, True))
        assert all(r is None for r in results)


class TestHandleAnthropicStreamResponse:
    def test_skips_event_lines(self):
        results = list(handle_anthropic_stream_response("event: message_start", False))
        assert all(r is None for r in results)

    def test_ignores_non_data_lines(self):
        results = list(handle_anthropic_stream_response("random line", False))
        assert all(r is None for r in results)

    def test_message_start(self):
        data = {"type": "message_start", "message": {"id": "msg_123"}}
        line = f"data: {json.dumps(data)}"
        results = list(handle_anthropic_stream_response(line, False))
        starts = [r for r in results if r and r.type == "response_start"]
        assert len(starts) == 1
        assert starts[0].id == "msg_123"

    def test_content_block_start_tool_use(self):
        data = {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "tool_123", "name": "search"},
        }
        line = f"data: {json.dumps(data)}"
        results = list(handle_anthropic_stream_response(line, True))

        tool_calls = [r for r in results if r and r.type == "tool_call"]
        assert len(tool_calls) == 1
        assert "tool_123" in TOOL_CALLS
        assert TOOL_CALLS["tool_123"].name == "search"

    def test_content_block_delta_text(self):
        data = {"type": "content_block_delta", "delta": {"text": "Hello world"}}
        line = f"data: {json.dumps(data)}"
        results = list(handle_anthropic_stream_response(line, True))

        texts = [r for r in results if r and r.type == "output_text"]
        assert len(texts) == 1
        assert texts[0].delta == "Hello world"

    def test_content_block_delta_thinking(self):
        data = {"type": "content_block_delta", "delta": {"thinking": "pondering..."}}
        line = f"data: {json.dumps(data)}"
        results = list(handle_anthropic_stream_response(line, True))

        thinking = [r for r in results if r and r.type == "reasoning_content"]
        assert len(thinking) == 1
        assert thinking[0].delta == "pondering..."

    def test_content_block_delta_partial_json(self):
        # First set up a tool call
        start_data = {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "tool_abc", "name": "func"},
        }
        list(handle_anthropic_stream_response(f"data: {json.dumps(start_data)}", True))

        # Then send partial JSON
        delta_data = {
            "type": "content_block_delta",
            "delta": {"partial_json": '{"x": 1}'},
        }
        results = list(
            handle_anthropic_stream_response(f"data: {json.dumps(delta_data)}", True)
        )

        tool_calls = [r for r in results if r and r.type == "tool_call"]
        assert len(tool_calls) == 1
        assert TOOL_CALLS["tool_abc"].arguments == '{"x": 1}'

    def test_message_stop(self):
        data = {"type": "message_stop"}
        line = f"data: {json.dumps(data)}"
        results = list(handle_anthropic_stream_response(line, True))

        ends = [r for r in results if r and r.type == "block_end"]
        assert len(ends) == 1

    def test_invalid_json_yields_none(self):
        line = "data: {not valid json"
        results = list(handle_anthropic_stream_response(line, True))
        assert all(r is None for r in results)


class TestHandleOpenaiResponsesStreamResponse:
    def test_ignores_non_data_lines(self):
        results = list(handle_openai_responses_stream_response("random", False))
        assert all(r is None for r in results)

    def test_response_created(self):
        data = {"type": "response.created", "response": {"id": "resp_123"}}
        line = f"data: {json.dumps(data)}"
        results = list(handle_openai_responses_stream_response(line, False))

        starts = [r for r in results if r and r.type == "response_start"]
        assert len(starts) == 1
        assert starts[0].id == "resp_123"

    def test_output_text_delta(self):
        data = {"type": "response.output_text.delta", "delta": "Hello"}
        line = f"data: {json.dumps(data)}"
        results = list(handle_openai_responses_stream_response(line, True))

        texts = [r for r in results if r and r.type == "output_text"]
        assert len(texts) == 1
        assert texts[0].delta == "Hello"

    def test_reasoning_summary_delta(self):
        data = {"type": "response.reasoning_summary_text.delta", "delta": "Summary"}
        line = f"data: {json.dumps(data)}"
        results = list(handle_openai_responses_stream_response(line, True))

        reasoning = [r for r in results if r and r.type == "reasoning_content"]
        assert len(reasoning) == 1
        assert reasoning[0].delta == "Summary"

    def test_response_completed(self):
        data = {"type": "response.completed"}
        line = f"data: {json.dumps(data)}"
        results = list(handle_openai_responses_stream_response(line, True))

        ends = [r for r in results if r and r.type == "block_end"]
        assert len(ends) == 1

    def test_tool_call_added(self):
        data = {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "search_web",
            },
        }
        line = f"data: {json.dumps(data)}"
        list(handle_openai_responses_stream_response(line, True))

        assert "call_1" in TOOL_CALLS
        assert TOOL_CALLS["call_1"].name == "search_web"

    def test_tool_call_arguments_delta(self):
        list(
            handle_openai_responses_stream_response(
                f"data: {json.dumps({'type': 'response.output_item.added', 'item': {'type': 'function_call', 'id': 'fc_2', 'call_id': 'call_2', 'name': 'tool'}})}",
                True,
            )
        )
        data = {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_2",
            "delta": '{"q": "test"}',
        }
        line = f"data: {json.dumps(data)}"
        results = list(handle_openai_responses_stream_response(line, True))

        assert TOOL_CALLS["call_2"].arguments == '{"q": "test"}'
        tool_calls = [r for r in results if r and r.type == "tool_call"]
        assert len(tool_calls) == 1

    def test_tool_call_done_sets_arguments(self):
        added = {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_3",
                "call_id": "call_3",
                "name": "tool",
            },
        }
        list(
            handle_openai_responses_stream_response(
                f"data: {json.dumps(added)}",
                True,
            )
        )
        data = {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_3",
                "call_id": "call_3",
                "arguments": "{}",
            },
        }
        line = f"data: {json.dumps(data)}"
        list(handle_openai_responses_stream_response(line, True))

        assert TOOL_CALLS["call_3"].arguments == "{}"

    def test_invalid_json_yields_none(self):
        line = "data: broken{"
        results = list(handle_openai_responses_stream_response(line, True))
        assert all(r is None for r in results)
