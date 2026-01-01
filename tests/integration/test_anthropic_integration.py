import pytest

from tests.conftest import skip_if_no_anthropic


@pytest.mark.integration
class TestAnthropicIntegration:
    """Integration tests against real Anthropic API.

    Requires ANTHROPIC_API_KEY environment variable to be set.
    Uses claude-3-5-haiku-20241022 for all tests.
    """

    MODEL = "claude-3-5-haiku-20241022"

    @skip_if_no_anthropic
    @pytest.mark.asyncio
    async def test_simple_completion(self):
        """Test basic message with Haiku"""
        from buzzllm.llm import (
            LLMOptions,
            make_anthropic_request_args,
            handle_anthropic_stream_response,
        )
        import requests

        opts = LLMOptions(
            model=self.MODEL,
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
            max_tokens=50,
            temperature=0.0,
        )

        request_args = make_anthropic_request_args(
            opts, "Say 'hello' and nothing else.", "You are helpful."
        )

        response = requests.post(
            opts.url,
            headers=request_args.headers,
            json=request_args.data,
            stream=True,
            timeout=30,
        )
        response.raise_for_status()

        collected_text = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            for stream_response in handle_anthropic_stream_response(line, True):
                if stream_response and stream_response.type == "output_text":
                    collected_text.append(stream_response.delta)

        full_text = "".join(collected_text).lower()
        assert "hello" in full_text

    @skip_if_no_anthropic
    @pytest.mark.asyncio
    async def test_thinking_mode(self):
        """Test extended thinking with Claude Sonnet (supports thinking)"""
        from buzzllm.llm import (
            LLMOptions,
            make_anthropic_request_args,
            handle_anthropic_stream_response,
        )
        import requests

        # Use Sonnet 4 which supports extended thinking
        opts = LLMOptions(
            model="claude-sonnet-4-20250514",
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
            think=True,
        )

        request_args = make_anthropic_request_args(
            opts, "What is 15 * 7? Just give the number.", "You are helpful."
        )

        # Verify thinking is configured
        assert request_args.data["thinking"]["type"] == "enabled"
        assert request_args.data["max_tokens"] == 32000

        response = requests.post(
            opts.url,
            headers=request_args.headers,
            json=request_args.data,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

        collected_text = []
        collected_thinking = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            for stream_response in handle_anthropic_stream_response(line, True):
                if stream_response:
                    if stream_response.type == "output_text":
                        collected_text.append(stream_response.delta)
                    elif stream_response.type == "reasoning_content":
                        collected_thinking.append(stream_response.delta)

        full_text = "".join(collected_text)
        assert "105" in full_text

    @skip_if_no_anthropic
    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """Test tool calling with Haiku 4.5"""
        from buzzllm.llm import (
            LLMOptions,
            make_anthropic_request_args,
            handle_anthropic_stream_response,
            TOOL_CALLS,
        )
        from buzzllm.tools.utils import callable_to_anthropic_schema
        import requests

        def calculate(expression: str) -> str:
            """Calculate a math expression

            :param expression: The math expression to evaluate
            """
            return str(eval(expression))

        tools = [callable_to_anthropic_schema(calculate)]

        opts = LLMOptions(
            model=self.MODEL,
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
            max_tokens=200,
            temperature=0.0,
            tools=tools,
        )

        request_args = make_anthropic_request_args(
            opts,
            "Calculate 123 * 456 using the calculate tool.",
            "You must use the calculate tool to answer math questions.",
        )

        response = requests.post(
            opts.url,
            headers=request_args.headers,
            json=request_args.data,
            stream=True,
            timeout=30,
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            list(handle_anthropic_stream_response(line, True))

        # Should have triggered a tool call
        assert len(TOOL_CALLS) >= 1
        tool_call = list(TOOL_CALLS.values())[0]
        assert tool_call.name == "calculate"

    @skip_if_no_anthropic
    @pytest.mark.asyncio
    async def test_streaming_response_types(self):
        """Test that we receive expected streaming response types"""
        from buzzllm.llm import (
            LLMOptions,
            make_anthropic_request_args,
            handle_anthropic_stream_response,
        )
        import requests

        opts = LLMOptions(
            model=self.MODEL,
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
            max_tokens=20,
        )

        request_args = make_anthropic_request_args(opts, "Hi", "Be brief.")

        response = requests.post(
            opts.url,
            headers=request_args.headers,
            json=request_args.data,
            stream=True,
            timeout=30,
        )
        response.raise_for_status()

        response_types = set()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            for stream_response in handle_anthropic_stream_response(line, False):
                if stream_response:
                    response_types.add(stream_response.type)

        # Should have received start and end events
        assert "response_start" in response_types
        assert "block_end" in response_types
