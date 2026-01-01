import pytest
import asyncio

from tests.conftest import skip_if_no_openai


@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests against real OpenAI API.

    Requires OPENAI_API_KEY environment variable to be set.
    Uses gpt-4.1-mini for standard tests and gpt-5.1-nano for reasoning tests.
    """

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_simple_completion_gpt41_mini(self):
        """Test basic chat completion with gpt-4.1-mini"""
        from buzzllm.llm import (
            LLMOptions,
            make_openai_request_args,
            handle_openai_stream_response,
        )
        import requests

        opts = LLMOptions(
            model="gpt-4.1-mini",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            max_tokens=50,
            temperature=0.0,
        )

        request_args = make_openai_request_args(
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
            for stream_response in handle_openai_stream_response(line, True):
                if stream_response and stream_response.type == "output_text":
                    collected_text.append(stream_response.delta)

        full_text = "".join(collected_text).lower()
        assert "hello" in full_text

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_reasoning_model_gpt51(self):
        """Test reasoning model gpt-5.1 with think mode"""
        from buzzllm.llm import (
            LLMOptions,
            make_openai_request_args,
            handle_openai_stream_response,
        )
        import requests

        opts = LLMOptions(
            model="gpt-5.1",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            think=True,
        )

        request_args = make_openai_request_args(
            opts, "What is 2+2? Answer with just the number.", "You are helpful."
        )

        # Check that reasoning_effort is set for reasoning models
        assert request_args.data.get("reasoning_effort") == "high"
        assert request_args.data["messages"][0]["role"] == "developer"

        response = requests.post(
            opts.url,
            headers=request_args.headers,
            json=request_args.data,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

        collected_text = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            for stream_response in handle_openai_stream_response(line, True):
                if stream_response and stream_response.type == "output_text":
                    collected_text.append(stream_response.delta)

        full_text = "".join(collected_text)
        assert "4" in full_text

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """Test tool calling functionality with gpt-4.1-mini"""
        from buzzllm.llm import (
            LLMOptions,
            make_openai_request_args,
            handle_openai_stream_response,
            TOOL_CALLS,
        )
        from buzzllm.tools.utils import callable_to_openai_schema
        import requests

        def get_weather(city: str) -> str:
            """Get the weather for a city

            :param city: The city name
            """
            return f"Weather in {city}: Sunny, 72°F"

        tools = [callable_to_openai_schema(get_weather)]

        opts = LLMOptions(
            model="gpt-4.1-mini",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            max_tokens=100,
            temperature=0.0,
            tools=tools,
        )

        request_args = make_openai_request_args(
            opts, "What's the weather in Paris?", "You are helpful. Use tools when needed."
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
            list(handle_openai_stream_response(line, True))

        # Should have triggered a tool call
        assert len(TOOL_CALLS) >= 1
        tool_call = list(TOOL_CALLS.values())[0]
        assert tool_call.name == "get_weather"
        assert "Paris" in tool_call.arguments or "paris" in tool_call.arguments.lower()
