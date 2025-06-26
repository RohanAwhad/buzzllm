import asyncio
import dataclasses
import json
import os
import requests

from typing import Generator, Optional, Callable, Literal, Any
from dataclasses import dataclass

from .tools import utils


@dataclass
class LLMOptions:
    model: str
    url: str
    api_key_name: Optional[str] = None
    max_tokens: Optional[int] = 8192
    temperature: float = 0.8
    think: bool = False
    tools: Optional[list[dict]] = None
    max_infer_iters: int = 10


@dataclass
class RequestArgs:
    data: dict
    headers: dict


@dataclass
class StreamResponse:
    id: str
    delta: str
    type: Literal[
        "response_start",
        "output_text",
        "reasoning_content",
        "tool_call",
        "response_end",
    ]

    def to_json(self):
        return json.dumps(dataclasses.asdict(self))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str
    executed: bool = False
    result: Any = None

    async def execute(self, func: Callable):
        args = json.loads(self.arguments)
        res = func(**args)
        if asyncio.iscoroutine(res):
            self.result = await res
        else:
            self.result = res
        self.executed = True


TOOL_CALLS: dict[str, ToolCall] = {}
current_tool_call_id: str = ""


async def run_tools():
    tasks = []
    for tc in TOOL_CALLS.values():
        if tc.executed:
            continue
        tasks.append(tc.execute(utils.AVAILABLE_TOOLS[tc.name]))

    await asyncio.gather(*tasks)


async def invoke_llm(
    opts: LLMOptions,
    prompt: str,
    system_prompt: str,
    make_request_args: Callable,
    handle_stream_response: Callable,
) -> None:
    """Invoke LLM with streaming response, printing StreamResponse objects to stdout as JSON"""

    request_args = make_request_args(opts, prompt, system_prompt)
    messages = request_args.data.get("messages", [])

    while True:
        try:

            print(request_args.data["messages"])
            input()
            message_started = False
            # Make streaming request
            response = requests.post(
                opts.url,
                headers=request_args.headers,
                json=request_args.data,
                stream=True,
                timeout=30,
            )
            response.raise_for_status()

            # Process streaming response
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue

                # Handle the streaming response
                stream_responses_gen = handle_stream_response(line, message_started)
                for stream_response in stream_responses_gen:
                    if not stream_response:
                        continue
                    if stream_response.type == "response_start":
                        message_started = True
                    print_to_stdout(stream_response)

            # Perform tool calls
            if not TOOL_CALLS:
                return

            await run_tools()

            # Add tool call and response messages
            tool_call_response_to_openai_messages(messages, TOOL_CALLS)
            request_args.data["messages"] = messages

            # Clear tool calls for next iteration
            TOOL_CALLS.clear()

        except Exception as e:
            print(e)
            # Print error as StreamResponse
            error_response = StreamResponse(
                id="", delta=f"Error: {str(e)}", type="response_end"
            )
            print_to_stdout(error_response)


def tool_call_response_to_openai_messages(
    messages: list, tool_calls: dict[str, ToolCall]
):
    if not tool_calls:
        return

    # Add assistant message with tool calls
    tool_calls_list = []
    for tc in tool_calls.values():
        tool_calls_list.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
        )

    messages.append({"role": "assistant", "tool_calls": tool_calls_list})

    # Add tool result messages
    for tc in tool_calls.values():
        messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": str(tc.result)}
        )


def print_to_stdout(data: StreamResponse) -> None:
    # Print to stdout in SSE format
    print(f"event: {data.type}")
    print(f"data: {data.to_json()}")
    print("", flush=True)  # Empty line for SSE format


# LLM Specific funcs


# ===
# OpenAI Chat completeion api
# ===
def make_openai_request_args(
    opts: LLMOptions, prompt: str, system_prompt: str
) -> RequestArgs:
    # json body
    role = "system"
    if opts.model == "o3":
        role = "developer"

    data = {
        "messages": [
            {"role": role, "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "model": opts.model,
        "stream": True,
    }

    if opts.model == "o3":
        data["reasoning_effort"] = "high"
        data["response_format"] = {"type": "text"}
    else:
        data["temperature"] = opts.temperature
        data["max_tokens"] = opts.max_tokens

    if opts.tools:
        data["tools"] = opts.tools

    # headers
    headers = {"Content-Type": "application/json"}
    if opts.api_key_name:
        api_key = os.environ.get(opts.api_key_name, None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    return RequestArgs(data=data, headers=headers)


def handle_openai_stream_response(
    line: str, message_started: bool
) -> Generator[StreamResponse | None, None, None]:

    if not line.startswith("data: "):
        yield None
    data_content = line[len("data: ") :]  # Remove 'data: ' prefix

    if data_content == "[DONE]":
        yield StreamResponse(delta="", type="response_end", id="")

    try:
        chunk_data = json.loads(data_content)

        # Handle response start
        if not message_started and chunk_data.get("id"):
            response_id = chunk_data["id"]
            yield StreamResponse(id=response_id, type="response_start", delta="")

        # Handle content deltas
        if not chunk_data.get("choices") or len(chunk_data["choices"]) == 0:
            yield None

        choice = chunk_data["choices"][0]
        delta = choice.get("delta", {})

        # Handle regular content
        if "content" in delta and delta["content"]:
            yield StreamResponse(id="", delta=delta["content"], type="output_text")

        # Handle reasoning content (for o1 models)
        if "reasoning" in delta and delta["reasoning"]:
            yield StreamResponse(
                id="", delta=delta["reasoning"], type="reasoning_content"
            )

        # Handle tool calls
        if "tool_calls" in delta:
            global current_tool_call_id
            for tool_call in delta["tool_calls"]:
                tool_call_content = ""

                # first chunk - create new tool call
                if "id" in tool_call:
                    current_tool_call_id = tool_call["id"]
                    TOOL_CALLS[current_tool_call_id] = ToolCall(
                        id=current_tool_call_id, name="", arguments="", executed=False
                    )

                if "function" in tool_call:
                    function = tool_call["function"]

                    if "name" in function and function["name"]:
                        if current_tool_call_id in TOOL_CALLS:
                            TOOL_CALLS[current_tool_call_id].name = function["name"]
                        tool_call_content += f"Function: {function['name']}\n"

                    if "arguments" in function and function["arguments"]:
                        if current_tool_call_id in TOOL_CALLS:
                            TOOL_CALLS[current_tool_call_id].arguments += function[
                                "arguments"
                            ]
                        tool_call_content += function["arguments"]

                if tool_call_content:
                    yield StreamResponse(
                        id="", delta=tool_call_content, type="tool_call"
                    )

    except Exception:
        yield None


# ===
# Anthropic messages api
# ===


def make_anthropic_request_args(
    opts: LLMOptions, prompt: str, system_prompt: str
) -> RequestArgs:
    data = {
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "model": opts.model,
        "stream": True,
    }

    if opts.think:
        data["max_tokens"] = 40000
        data["thinking"] = {"type": "enabled", "budget_tokens": 32000}
    else:
        data["max_tokens"] = opts.max_tokens or 8192

    headers = {"Content-Type": "application/json"}
    if opts.api_key_name:
        api_key = os.environ.get(opts.api_key_name, None)
        if api_key:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"

    return RequestArgs(data=data, headers=headers)


def handle_anthropic_stream_response(
    line: str, message_started: bool
) -> Generator[StreamResponse | None, None, None]:
    # Skip event lines, only process data lines
    if line.startswith("event: "):
        yield None
        return

    if not line.startswith("data: "):
        yield None
        return

    data_content = line[len("data: ") :]

    try:
        chunk_data = json.loads(data_content)

        # Handle message start
        if chunk_data.get("type") == "message_start":
            message_id = chunk_data.get("message", {}).get("id", "")
            yield StreamResponse(id=message_id, type="response_start", delta="")

        # Handle content block delta
        elif chunk_data.get("type") == "content_block_delta":
            delta = chunk_data.get("delta", {})

            # Handle thinking content
            if "thinking" in delta and delta["thinking"]:
                yield StreamResponse(
                    id="", delta=delta["thinking"], type="reasoning_content"
                )

            # Handle regular text content
            elif "text" in delta and delta["text"]:
                yield StreamResponse(id="", delta=delta["text"], type="output_text")

        # Handle message stop
        elif chunk_data.get("type") == "message_stop":
            yield StreamResponse(id="", delta="", type="response_end")

    except Exception:
        yield None
