import pytest

from tests.conftest import skip_if_no_openai


@pytest.mark.integration
class TestOpenAIResponsesIntegration:
    @skip_if_no_openai
    def test_simple_responses_output_text(self):
        from buzzllm.llm import (
            LLMOptions,
            make_openai_responses_request_args,
            handle_openai_responses_stream_response,
        )
        import requests

        opts = LLMOptions(
            model="gpt-5.1",
            url="https://api.openai.com/v1/responses",
            api_key_name="OPENAI_API_KEY",
            max_tokens=50,
            temperature=0.0,
        )

        request_args = make_openai_responses_request_args(
            opts, "Say 'hello' and nothing else.", "You are helpful."
        )

        response = requests.post(
            opts.url,
            headers=request_args.headers,
            json=request_args.data,
            stream=True,
            timeout=30,
        )
        assert response.status_code == 200, response.text

        collected_text = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            for stream_response in handle_openai_responses_stream_response(line, True):
                if stream_response and stream_response.type == "output_text":
                    collected_text.append(stream_response.delta)

        full_text = "".join(collected_text).lower()
        assert "hello" in full_text

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_tool_calling_flow(self):
        from buzzllm.llm import (
            LLMOptions,
            invoke_llm,
            make_openai_responses_request_args,
            handle_openai_responses_stream_response,
            tool_call_response_to_openai_responses_messages,
        )
        from buzzllm.tools import utils

        tool_called = {"count": 0}

        def get_weather(city: str) -> str:
            """Get weather for a city"""
            tool_called["count"] += 1
            return f"Weather in {city}: Sunny"

        utils.add_tool(get_weather)
        tools = [utils.callable_to_openai_schema(utils.AVAILABLE_TOOLS["get_weather"])]

        opts = LLMOptions(
            model="gpt-5.1",
            url="https://api.openai.com/v1/responses",
            api_key_name="OPENAI_API_KEY",
            max_tokens=100,
            temperature=0.0,
            tools=tools,
        )

        def make_request_args_required(_opts, _prompt, _system_prompt):
            args = make_openai_responses_request_args(_opts, _prompt, _system_prompt)
            args.data["tool_choice"] = "required"
            return args

        await invoke_llm(
            opts,
            "Call get_weather with city=Paris before responding.",
            "Always call tools when a tool is available.",
            make_request_args_required,
            handle_openai_responses_stream_response,
            tool_call_response_to_openai_responses_messages,
            sse=False,
            brief=True,
        )

        assert tool_called["count"] >= 1
