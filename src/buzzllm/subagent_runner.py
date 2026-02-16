import asyncio
import json
import sys

from .llm import (
    LLMOptions,
    invoke_llm,
    make_openai_request_args,
    handle_openai_stream_response,
    tool_call_response_to_openai_messages,
    make_openai_responses_request_args,
    handle_openai_responses_stream_response,
    tool_call_response_to_openai_responses_messages,
    make_anthropic_request_args,
    handle_anthropic_stream_response,
    tool_call_response_to_anthropic_messages,
    make_vertexai_anthropic_request_args,
)
from .tools import utils


def main() -> None:
    payload = json.loads(sys.stdin.read())

    provider = payload["provider"]
    tool_subset = payload.get("tool_subset") or []

    provider_map = {
        "openai-chat": (
            make_openai_request_args,
            handle_openai_stream_response,
            utils.callable_to_openai_schema,
            tool_call_response_to_openai_messages,
        ),
        "openai-responses": (
            make_openai_responses_request_args,
            handle_openai_responses_stream_response,
            utils.callable_to_openai_schema,
            tool_call_response_to_openai_responses_messages,
        ),
        "anthropic": (
            make_anthropic_request_args,
            handle_anthropic_stream_response,
            utils.callable_to_anthropic_schema,
            tool_call_response_to_anthropic_messages,
        ),
        "vertexai-anthropic": (
            make_vertexai_anthropic_request_args,
            handle_anthropic_stream_response,
            utils.callable_to_anthropic_schema,
            tool_call_response_to_anthropic_messages,
        ),
    }

    (
        make_request_args_fn,
        handle_stream_response_fn,
        callable_to_schema,
        add_tool_response,
    ) = provider_map[provider]

    tools = None
    if tool_subset:
        tools = utils.build_tool_schemas(tool_subset, callable_to_schema)

    opts = LLMOptions(
        model=payload["model"],
        url=payload["url"],
        api_key_name=payload.get("api_key_name"),
        max_tokens=payload.get("max_tokens"),
        temperature=payload.get("temperature", 0.8),
        think=payload.get("think", False),
        tools=tools,
    )

    system_prompt = payload.get("system_prompt") or ""

    asyncio.run(
        invoke_llm(
            opts,
            payload["prompt"],
            system_prompt,
            make_request_args_fn,
            handle_stream_response_fn,
            add_tool_response,
            sse=True,
            brief=False,
        )
    )


if __name__ == "__main__":
    main()
