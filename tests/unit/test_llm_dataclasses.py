import json
import pytest
import asyncio

from buzzllm.llm import LLMOptions, RequestArgs, StreamResponse, ToolCall


class TestLLMOptions:
    def test_default_values(self):
        opts = LLMOptions(model="gpt-4", url="https://api.example.com")
        assert opts.model == "gpt-4"
        assert opts.url == "https://api.example.com"
        assert opts.api_key_name is None
        assert opts.max_tokens == 8192
        assert opts.temperature == 0.8
        assert opts.think is False
        assert opts.tools is None
        assert opts.max_infer_iters == 10
        assert opts.output_mode is None
        assert opts.output_schema is None

    def test_custom_values(self):
        opts = LLMOptions(
            model="claude-3",
            url="https://api.anthropic.com",
            api_key_name="MY_KEY",
            max_tokens=4096,
            temperature=0.5,
            think=True,
            tools=[{"name": "tool1"}],
            max_infer_iters=5,
            output_mode="json_schema",
            output_schema={"type": "object"},
        )
        assert opts.max_tokens == 4096
        assert opts.temperature == 0.5
        assert opts.think is True
        assert opts.tools == [{"name": "tool1"}]
        assert opts.max_infer_iters == 5
        assert opts.output_mode == "json_schema"
        assert opts.output_schema == {"type": "object"}


class TestRequestArgs:
    def test_structure(self):
        args = RequestArgs(
            data={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Content-Type": "application/json"},
        )
        assert args.data["messages"][0]["content"] == "hello"
        assert args.headers["Content-Type"] == "application/json"


class TestStreamResponse:
    def test_output_text_type(self):
        resp = StreamResponse(id="123", delta="Hello", type="output_text")
        assert resp.id == "123"
        assert resp.delta == "Hello"
        assert resp.type == "output_text"

    def test_reasoning_content_type(self):
        resp = StreamResponse(id="", delta="thinking...", type="reasoning_content")
        assert resp.type == "reasoning_content"

    def test_tool_call_type(self):
        resp = StreamResponse(id="", delta="search_web", type="tool_call")
        assert resp.type == "tool_call"

    def test_output_structured_type(self):
        resp = StreamResponse(id="", delta="{}", type="output_structured")
        assert resp.type == "output_structured"

    def test_to_json(self):
        resp = StreamResponse(id="abc", delta="test", type="output_text")
        json_str = resp.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "abc"
        assert parsed["delta"] == "test"
        assert parsed["type"] == "output_text"

    def test_to_json_escapes_special_chars(self):
        resp = StreamResponse(id="", delta='quote: "hello"', type="output_text")
        json_str = resp.to_json()
        parsed = json.loads(json_str)
        assert parsed["delta"] == 'quote: "hello"'


class TestToolCall:
    def test_initial_state(self):
        tc = ToolCall(id="call_1", name="search", arguments='{"q": "test"}')
        assert tc.id == "call_1"
        assert tc.name == "search"
        assert tc.arguments == '{"q": "test"}'
        assert tc.executed is False
        assert tc.result is None

    def test_execute_sync_function(self):
        def sync_func(x: int, y: int) -> int:
            return x + y

        tc = ToolCall(id="1", name="add", arguments='{"x": 2, "y": 3}')
        asyncio.run(tc.execute(sync_func))

        assert tc.executed is True
        assert tc.result == 5

    def test_execute_async_function(self):
        async def async_func(name: str) -> str:
            return f"Hello, {name}"

        tc = ToolCall(id="2", name="greet", arguments='{"name": "World"}')
        asyncio.run(tc.execute(async_func))

        assert tc.executed is True
        assert tc.result == "Hello, World"

    def test_execute_parses_json_arguments(self):
        def func(data: dict) -> dict:
            return {"received": data}

        tc = ToolCall(id="3", name="process", arguments='{"data": {"key": "value"}}')
        asyncio.run(tc.execute(func))

        assert tc.result == {"received": {"key": "value"}}
