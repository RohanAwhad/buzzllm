import argparse
from src.llm import (
    LLMOptions,
    invoke_llm,
    make_openai_request_args,
    handle_openai_stream_response,
    make_anthropic_request_args,
    handle_anthropic_stream_response,
)
from src.prompts import prompts


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

    return parser.parse_args()


def chat(
    model,
    url,
    prompt,
    system_prompt,
    provider,
    api_key_name,
    max_tokens,
    temperature,
    think,
):
    """Invoke LLM with streaming response"""

    if prompt is None:
        return

    # Check if system_prompt is a predefined prompt name
    if system_prompt in prompts:
        system_prompt = prompts[system_prompt]

    provider_map = {
        "openai-chat": (make_openai_request_args, handle_openai_stream_response),
        "anthropic": (make_anthropic_request_args, handle_anthropic_stream_response),
    }
    make_request_args_fn, handle_stream_response_fn = provider_map[provider]

    # Create LLM options
    opts = LLMOptions(
        model=model,
        url=url,
        api_key_name=api_key_name,
        max_tokens=max_tokens,
        temperature=temperature,
        think=think,
    )

    # Invoke LLM - this will print streaming responses to stdout
    invoke_llm(
        opts, prompt, system_prompt, make_request_args_fn, handle_stream_response_fn
    )


def main():
    """Main function that parses args and calls chat"""
    args = parse_args()
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
    )


if __name__ == "__main__":
    main()
