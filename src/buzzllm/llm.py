import asyncio
import dataclasses
import json
import os
import requests
import subprocess

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
    output_mode: Optional[Literal["json_schema", "json_object"]] = None
    output_schema: Optional[dict] = None


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
        "tool_result",
        "block_end",
        "response_end",
        "output_structured",
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
last_openai_response_id: str = ""
openai_responses_item_id_to_call_id: dict[str, str] = {}


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
    add_tool_response: Callable,
    sse: bool,
    brief: bool = False,
) -> None:
    """Invoke LLM with streaming response, printing StreamResponse objects to stdout as JSON"""

    request_args = make_request_args(opts, prompt, system_prompt)
    messages = request_args.data.get("messages")
    inputs = request_args.data.get("input")

    try:
        while True:
            message_started = False
            structured_output_buffer = "" if opts.output_mode else None
            # Make streaming request
            response = requests.post(
                opts.url,
                headers=request_args.headers,
                json=request_args.data,
                stream=True,
                timeout=900,
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
                    if opts.output_mode and stream_response.type == "output_text":
                        structured_output_buffer = (
                            structured_output_buffer or ""
                        ) + stream_response.delta
                        continue
                    if opts.output_mode and stream_response.type == "block_end":
                        continue
                    print_to_stdout(stream_response, sse, brief)

            # Perform tool calls
            if not TOOL_CALLS:
                if opts.output_mode:
                    structured_response = StreamResponse(
                        id="",
                        delta=structured_output_buffer or "",
                        type="output_structured",
                    )
                    print_to_stdout(structured_response, sse, brief)
                return

            await run_tools()

            # Print tool results
            for tc in TOOL_CALLS.values():
                if tc.executed and tc.result is not None:
                    result_response = StreamResponse(
                        id=tc.id,
                        delta=f"\n\nTool Result ({tc.name}):\n{str(tc.result)}\n",
                        type="tool_result",
                    )
                    print_to_stdout(result_response, sse, brief)

            # Add tool call and response messages
            if messages is not None:
                add_tool_response(messages, TOOL_CALLS)
                request_args.data["messages"] = messages
            elif inputs is not None:
                add_tool_response(request_args, TOOL_CALLS)
                inputs = request_args.data.get("input")

            # Clear tool calls for next iteration
            TOOL_CALLS.clear()
            openai_responses_item_id_to_call_id.clear()

    except Exception as e:
        print(e)
        # Print error as StreamResponse
        error_response = StreamResponse(
            id="", delta=f"Error: {str(e)}", type="block_end"
        )
        print_to_stdout(error_response, sse, brief)
    finally:
        # Cleanup any running containers
        try:
            from .tools import pythonexec

            pythonexec.cleanup_python_exec()
        except ImportError:
            pass
        print_to_stdout(
            StreamResponse(id="", delta="", type="response_end"), sse, brief
        )


def print_to_stdout(data: StreamResponse, sse: bool, brief: bool = False) -> None:
    # In brief mode, skip tool calls and results
    if brief and data.type in ("tool_call", "tool_result"):
        return

    if sse:
        # Print to stdout in SSE format
        print(f"event: {data.type}")
        print(f"data: {data.to_json()}")
        print("", flush=True)  # Empty line for SSE format
        return

    # Add colors for different content types
    if data.type == "tool_call":
        print(f"\033[96m{data.delta}\033[0m", end="", flush=True)  # Cyan
    elif data.type == "tool_result":
        print(f"\033[92m{data.delta}\033[0m", end="", flush=True)  # Green
    elif data.type == "reasoning_content":
        print(f"\033[93m{data.delta}\033[0m", end="", flush=True)  # Yellow
    elif data.type == "block_end":
        print("\n", end="", flush=True)
    elif data.type == "response_end":
        print("\n\n=== [ DONE ] ===")
    else:
        print(data.delta, end="", flush=True)


# LLM Specific funcs


# ===
# OpenAI Chat completeion api
# ===
def make_openai_request_args(
    opts: LLMOptions, prompt: str, system_prompt: str
) -> RequestArgs:
    OPENAI_REASONING_MODELS = [
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "o4-mini",
        "o3",
        "o3-pro",
        "gpt-5-pro",
    ]
    # json body
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "model": opts.model,
        "stream": True,
    }

    if opts.model in OPENAI_REASONING_MODELS:
        data["messages"][0]["role"] = "developer"
        data["response_format"] = {"type": "text"}
        data["reasoning_effort"] = (
            "none" if opts.model == "gpt-5.1" and not opts.think else "high"
        )
    else:
        data["temperature"] = opts.temperature
        data["max_tokens"] = opts.max_tokens

    if opts.output_mode == "json_schema" and opts.output_schema:
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": opts.output_schema,
                "strict": True,
            },
        }
    elif opts.output_mode == "json_object":
        data["response_format"] = {"type": "json_object"}

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
        yield StreamResponse(delta="", type="block_end", id="")

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
        if ("reasoning" in delta and delta["reasoning"]) or (
            "reasoning_content" in delta and delta["reasoning_content"]
        ):
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
                        tool_call_content += f"\nFunction: {function['name']} "

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
        data["max_tokens"] = 32000
        data["thinking"] = {"type": "enabled", "budget_tokens": 24000}
    else:
        data["max_tokens"] = opts.max_tokens or 8192

    if opts.tools:
        data["tools"] = opts.tools

    if opts.output_mode and opts.output_schema:
        data["output_config"] = {
            "format": {"type": "json_schema", "schema": opts.output_schema}
        }

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
    global current_tool_call_id
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

        # Handle content block start (for tool use)
        elif chunk_data.get("type") == "content_block_start":
            content_block = chunk_data.get("content_block", {})
            if content_block.get("type") == "tool_use":
                tool_id = content_block.get("id", "")
                tool_name = content_block.get("name", "")

                current_tool_call_id = tool_id
                TOOL_CALLS[tool_id] = ToolCall(
                    id=tool_id, name=tool_name, arguments="", executed=False
                )

                tool_call_content = f"Function: {tool_name}\n"
                yield StreamResponse(id="", delta=tool_call_content, type="tool_call")

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

            # Handle tool input JSON delta
            elif "partial_json" in delta and delta["partial_json"]:
                if current_tool_call_id in TOOL_CALLS:
                    TOOL_CALLS[current_tool_call_id].arguments += delta["partial_json"]

                yield StreamResponse(
                    id="", delta=delta["partial_json"], type="tool_call"
                )

        # Handle message stop
        elif chunk_data.get("type") == "message_stop":
            yield StreamResponse(id="", delta="", type="block_end")

    except Exception:
        yield None


def tool_call_response_to_anthropic_messages(
    messages: list, tool_calls: dict[str, ToolCall]
):
    if not tool_calls:
        return

    # Add assistant message with tool uses
    content = []
    for tc in tool_calls.values():
        content.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": json.loads(tc.arguments),
            }
        )

    messages.append({"role": "assistant", "content": content})

    # Add tool result message
    content = []
    for tc in tool_calls.values():
        content.append(
            {"type": "tool_result", "tool_use_id": tc.id, "content": str(tc.result)}
        )

    messages.append({"role": "user", "content": content})


# ===
# OpenAI Responses api
# ===


def _convert_openai_responses_tools(tools: list[dict]) -> list[dict]:
    converted = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            function = tool.get("function", {})
            converted.append(
                {
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description"),
                    "parameters": function.get("parameters"),
                }
            )
        else:
            converted.append(tool)
    return converted


def make_openai_responses_request_args(
    opts: LLMOptions, prompt: str, system_prompt: str
) -> RequestArgs:
    data = {
        "model": opts.model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "stream": True,
        "store": False,
        "temperature": opts.temperature,
    }

    if opts.think:
        data["reasoning"] = {
            "effort": "high",
            "summary": "detailed",
        }
    else:
        data["reasoning"] = {"effort": "none"}

    if opts.max_tokens:
        data["max_output_tokens"] = opts.max_tokens

    if system_prompt:
        data["instructions"] = system_prompt

    if opts.tools:
        data["tools"] = _convert_openai_responses_tools(opts.tools)
        data["tool_choice"] = "auto"

    headers = {"Content-Type": "application/json"}
    if opts.api_key_name:
        api_key = os.environ.get(opts.api_key_name, None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    return RequestArgs(data=data, headers=headers)


def handle_openai_responses_stream_response(
    line: str, message_started: bool
) -> Generator[StreamResponse | None, None, None]:
    if not line.startswith("data: "):
        yield None
        return

    data_content = line[len("data: ") :]

    try:
        chunk_data = json.loads(data_content)
        event_type = chunk_data.get("type", "")

        # Handle response start
        if event_type == "response.created":
            response_id = chunk_data.get("response", {}).get("id", "")
            global last_openai_response_id
            last_openai_response_id = response_id
            yield StreamResponse(id=response_id, type="response_start", delta="")

        # Handle regular text content
        elif event_type == "response.output_text.delta":
            delta_text = chunk_data.get("delta", "")
            yield StreamResponse(id="", delta=delta_text, type="output_text")

        # Handle reasoning summary content
        elif event_type == "response.reasoning_summary_text.delta":
            delta_text = chunk_data.get("delta", "")
            yield StreamResponse(id="", delta=delta_text, type="reasoning_content")

        elif event_type == "response.output_item.added":
            item = chunk_data.get("item", {})
            item_type = item.get("type")
            if item_type in ("tool_call", "function_call"):
                item_id = item.get("id", "")
                call_id = item.get("call_id") or item_id
                tool_name = item.get("name", "")
                if item_id:
                    openai_responses_item_id_to_call_id[item_id] = call_id
                TOOL_CALLS[call_id] = ToolCall(
                    id=call_id, name=tool_name, arguments="", executed=False
                )

        elif event_type in (
            "response.tool_call_arguments.delta",
            "response.function_call_arguments.delta",
        ):
            item_id = (
                chunk_data.get("item_id")
                or chunk_data.get("tool_call_id")
                or chunk_data.get("call_id")
                or ""
            )
            call_id = openai_responses_item_id_to_call_id.get(item_id, item_id)
            delta_text = chunk_data.get("delta", "")
            if call_id in TOOL_CALLS:
                TOOL_CALLS[call_id].arguments += delta_text
            if delta_text:
                yield StreamResponse(id="", delta=delta_text, type="tool_call")

        elif event_type == "response.output_item.done":
            item = chunk_data.get("item", {})
            item_type = item.get("type")
            if item_type in ("tool_call", "function_call"):
                item_id = item.get("id", "")
                call_id = item.get(
                    "call_id"
                ) or openai_responses_item_id_to_call_id.get(item_id, item_id)
                arguments = item.get("arguments")
                if call_id in TOOL_CALLS and arguments:
                    TOOL_CALLS[call_id].arguments = arguments

        # Handle response completion
        elif event_type == "response.completed":
            yield StreamResponse(id="", delta="", type="block_end")

    except Exception:
        yield None


def tool_call_response_to_openai_responses_messages(
    request_args: RequestArgs, tool_calls: dict[str, ToolCall]
):
    if not tool_calls:
        return

    request_args.data["previous_response_id"] = last_openai_response_id
    tool_results = []
    for tc in tool_calls.values():
        tool_results.append(
            {
                "type": "tool_result",
                "tool_call_id": tc.id,
                "content": str(tc.result),
            }
        )

    request_args.data["input"] = tool_results


# ===
# Vertex AI Anthropic messages api
# ===


def make_vertexai_anthropic_request_args(
    opts: LLMOptions, prompt: str, system_prompt: str
) -> RequestArgs:
    data = {
        "anthropic_version": "vertex-2023-10-16",
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    if opts.think:
        data["max_tokens"] = 32000
        data["thinking"] = {"type": "enabled", "budget_tokens": 24000}
    else:
        data["max_tokens"] = opts.max_tokens or 8192

    if opts.tools:
        data["tools"] = opts.tools

    if opts.output_mode and opts.output_schema:
        data["output_config"] = {
            "format": {"type": "json_schema", "schema": opts.output_schema}
        }

    headers = {"Content-Type": "application/json"}
    api_key = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    )
    headers["Authorization"] = f"Bearer {api_key.stdout.strip()}"
    return RequestArgs(data=data, headers=headers)
