import pytest

from buzzllm.llm import (
    ToolCall,
    tool_call_response_to_openai_messages,
    tool_call_response_to_anthropic_messages,
)


class TestToolCallResponseToOpenaiMessages:
    def test_empty_tool_calls_no_op(self):
        messages = [{"role": "user", "content": "hello"}]
        tool_call_response_to_openai_messages(messages, {})
        assert len(messages) == 1

    def test_single_tool_call(self):
        messages = [{"role": "user", "content": "hello"}]
        tool_calls = {
            "call_1": ToolCall(
                id="call_1",
                name="search_web",
                arguments='{"query": "test"}',
                executed=True,
                result="Found results",
            )
        }

        tool_call_response_to_openai_messages(messages, tool_calls)

        assert len(messages) == 3
        # Assistant message with tool_calls
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 1
        assert messages[1]["tool_calls"][0]["id"] == "call_1"
        assert messages[1]["tool_calls"][0]["type"] == "function"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "search_web"
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"query": "test"}'

        # Tool result message
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"
        assert messages[2]["content"] == "Found results"

    def test_multiple_tool_calls(self):
        messages = []
        tool_calls = {
            "call_1": ToolCall(
                id="call_1", name="func1", arguments="{}", executed=True, result="res1"
            ),
            "call_2": ToolCall(
                id="call_2", name="func2", arguments="{}", executed=True, result="res2"
            ),
        }

        tool_call_response_to_openai_messages(messages, tool_calls)

        # 1 assistant message + 2 tool messages
        assert len(messages) == 3
        assert len(messages[0]["tool_calls"]) == 2
        assert messages[1]["role"] == "tool"
        assert messages[2]["role"] == "tool"

    def test_preserves_existing_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]
        tool_calls = {
            "call_1": ToolCall(
                id="call_1", name="test", arguments="{}", executed=True, result="ok"
            )
        }

        tool_call_response_to_openai_messages(messages, tool_calls)

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "tool"


class TestToolCallResponseToAnthropicMessages:
    def test_empty_tool_calls_no_op(self):
        messages = [{"role": "user", "content": "hello"}]
        tool_call_response_to_anthropic_messages(messages, {})
        assert len(messages) == 1

    def test_single_tool_call(self):
        messages = [{"role": "user", "content": "hello"}]
        tool_calls = {
            "tool_1": ToolCall(
                id="tool_1",
                name="search",
                arguments='{"q": "test"}',
                executed=True,
                result="Search result",
            )
        }

        tool_call_response_to_anthropic_messages(messages, tool_calls)

        assert len(messages) == 3
        # Assistant message with tool_use content blocks
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["content"]) == 1
        assert messages[1]["content"][0]["type"] == "tool_use"
        assert messages[1]["content"][0]["id"] == "tool_1"
        assert messages[1]["content"][0]["name"] == "search"
        assert messages[1]["content"][0]["input"] == {"q": "test"}

        # User message with tool_result content blocks
        assert messages[2]["role"] == "user"
        assert len(messages[2]["content"]) == 1
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_1"
        assert messages[2]["content"][0]["content"] == "Search result"

    def test_multiple_tool_calls(self):
        messages = []
        tool_calls = {
            "t1": ToolCall(
                id="t1", name="f1", arguments='{"a": 1}', executed=True, result="r1"
            ),
            "t2": ToolCall(
                id="t2", name="f2", arguments='{"b": 2}', executed=True, result="r2"
            ),
        }

        tool_call_response_to_anthropic_messages(messages, tool_calls)

        # 1 assistant + 1 user message
        assert len(messages) == 2
        assert len(messages[0]["content"]) == 2  # 2 tool_use blocks
        assert len(messages[1]["content"]) == 2  # 2 tool_result blocks

    def test_arguments_parsed_to_dict(self):
        messages = []
        tool_calls = {
            "t1": ToolCall(
                id="t1",
                name="func",
                arguments='{"nested": {"key": "value"}, "list": [1, 2, 3]}',
                executed=True,
                result="ok",
            )
        }

        tool_call_response_to_anthropic_messages(messages, tool_calls)

        input_obj = messages[0]["content"][0]["input"]
        assert input_obj["nested"]["key"] == "value"
        assert input_obj["list"] == [1, 2, 3]
