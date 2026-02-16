import json
import subprocess
import sys
from typing import Iterable, Optional

from .tools.catalog import TOOL_NAMES


def parse_sse_output_text(lines: Iterable[str]) -> str:
    output_chunks = []
    current_event = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event: "):
            current_event = line[len("event: ") :].strip()
            continue
        if line.startswith("data: ") and current_event == "output_text":
            data_content = line[len("data: ") :]
            payload = json.loads(data_content)
            delta = payload.get("delta", "")
            if delta:
                output_chunks.append(delta)

    return "".join(output_chunks)


_SUBAGENT_CONTEXT: dict[str, object] = {}


def configure_subagent_context(
    model: str,
    provider: str,
    url: str,
    api_key_name: Optional[str],
    think: bool,
    temperature: float,
    max_tokens: Optional[int],
) -> None:
    _SUBAGENT_CONTEXT.clear()
    _SUBAGENT_CONTEXT.update(
        {
            "model": model,
            "provider": provider,
            "url": url,
            "api_key_name": api_key_name,
            "think": think,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    )


def build_subagent_payload(
    prompt: str,
    system_prompt: Optional[str],
    tool_subset: Optional[list[str]],
) -> dict:
    payload = {
        "model": _SUBAGENT_CONTEXT["model"],
        "provider": _SUBAGENT_CONTEXT["provider"],
        "url": _SUBAGENT_CONTEXT["url"],
        "api_key_name": _SUBAGENT_CONTEXT["api_key_name"],
        "prompt": prompt,
        "system_prompt": system_prompt or "",
        "tool_subset": tool_subset or [],
        "think": _SUBAGENT_CONTEXT["think"],
        "temperature": _SUBAGENT_CONTEXT["temperature"],
        "max_tokens": _SUBAGENT_CONTEXT["max_tokens"],
    }
    return payload


def raw_call_subagent(payload: dict) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "buzzllm.subagent_runner"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        return f"Error: Subagent exited with code {result.returncode}"

    return parse_sse_output_text(result.stdout.splitlines())


def call_subagent(
    prompt: str,
    system_prompt: Optional[str] = None,
    tool_subset: Optional[list[str]] = None,
) -> str:
    """Run a subagent with an optional tool subset."""
    if not _SUBAGENT_CONTEXT:
        return "Error: Subagent context is not configured"

    requested_tools = tool_subset or []
    unknown_tools = sorted(set(requested_tools) - TOOL_NAMES)
    if unknown_tools:
        return f"Error: Unknown tool names: {', '.join(unknown_tools)}"

    payload = build_subagent_payload(prompt, system_prompt, requested_tools)
    return raw_call_subagent(payload)
