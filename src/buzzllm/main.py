import argparse
import asyncio
from .llm import (
    LLMOptions,
    invoke_llm,
    make_openai_request_args,
    handle_openai_stream_response,
    tool_call_response_to_openai_messages,
    make_anthropic_request_args,
    handle_anthropic_stream_response,
    tool_call_response_to_anthropic_messages,
)
from .prompts import prompts
from .tools import utils, websearch, codesearch


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Invoke LLM with streaming response")

    parser.add_argument("model", help="LLM model name")
    parser.add_argument("url", help="LLM API URL")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument(
        "--system-prompt",
        default="scream at mee for not setting your system prompt",
        help=f"System prompt. Use a predefined prompt name or provide your own custom system prompt text. Avaiable prompts: {', '.join(list(prompts.keys()))}",
    )

    parser.add_argument(
        "--provider",
        choices=["openai-chat", "anthropic"],
        required=True,
        help="LLM provider type",
    )
    parser.add_argument(
        "--api-key-name", required=True, help="Environment variable name for API key"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192, help="Maximum tokens in response"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Response temperature"
    )
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    parser.add_argument(
        "-S", "--sse", action="store_true", help="Enable SSE mode for printing"
    )

    return parser.parse_args()


async def chat(
    model,
    url,
    prompt,
    system_prompt,
    provider,
    api_key_name,
    max_tokens,
    temperature,
    think,
    sse,
):
    """Invoke LLM with streaming response"""

    if prompt is None:
        return

    original_system_prompt = system_prompt
    if system_prompt in prompts:
        system_prompt = prompts[system_prompt]

    provider_map = {
        "openai-chat": (
            make_openai_request_args,
            handle_openai_stream_response,
            utils.callable_to_openai_schema,
            tool_call_response_to_openai_messages,
        ),
        "anthropic": (
            make_anthropic_request_args,
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
    if original_system_prompt == "websearch":
        # Add websearch tools
        utils.add_tool(websearch.search_web)
        utils.add_tool(websearch.scrape_webpage)
        # TODO: this is boilerplate need to remove this
        tools = [
            callable_to_schema(utils.AVAILABLE_TOOLS["search_web"]),
            callable_to_schema(utils.AVAILABLE_TOOLS["scrape_webpage"]),
        ]
    elif original_system_prompt == "codesearch":
        utils.add_tool(codesearch.bash_find)
        utils.add_tool(codesearch.bash_ripgrep)
        utils.add_tool(codesearch.bash_read)
        tools = [
            callable_to_schema(
                utils.AVAILABLE_TOOLS["bash_find"], codesearch.bash_find_tool_desc
            ),
            callable_to_schema(
                utils.AVAILABLE_TOOLS["bash_ripgrep"], codesearch.bash_ripgrep_tool_desc
            ),
            callable_to_schema(utils.AVAILABLE_TOOLS["bash_read"]),
        ]

    # Create LLM options
    opts = LLMOptions(
        model=model,
        url=url,
        api_key_name=api_key_name,
        max_tokens=max_tokens,
        temperature=temperature,
        think=think,
        tools=tools,
    )

    # Invoke LLM - this will print streaming responses to stdout
    await invoke_llm(
        opts,
        prompt,
        system_prompt,
        make_request_args_fn,
        handle_stream_response_fn,
        add_tool_response,
        sse,
    )


def main():
    """Main function that parses args and calls chat"""
    args = parse_args()
    asyncio.run(
        chat(
            model=args.model,
            url=args.url,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            provider=args.provider,
            api_key_name=args.api_key_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            think=args.think,
            sse=args.sse,
        )
    )


if __name__ == "__main__":
    main()
