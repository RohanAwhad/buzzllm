import pytest

from buzzllm.llm import LLMOptions, RequestArgs, StreamResponse, invoke_llm


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        return self._lines


@pytest.mark.asyncio
async def test_structured_output_non_sse(capsys, monkeypatch):
    opts = LLMOptions(
        model="gpt-4o-mini",
        url="https://api.openai.com/v1/chat/completions",
        api_key_name="OPENAI_API_KEY",
        output_mode="json_schema",
        output_schema={"type": "object"},
    )

    def make_request_args(_opts, _prompt, _system_prompt):
        return RequestArgs(data={"messages": []}, headers={})

    def handle_stream_response(_line, _started):
        yield StreamResponse(id="", delta='{"ok": true}', type="output_text")
        yield StreamResponse(id="", delta="", type="block_end")

    def add_tool_response(_messages, _tool_calls):
        return None

    monkeypatch.setattr(
        "buzzllm.llm.requests.post",
        lambda *args, **kwargs: FakeResponse(["data: {}"]),
    )

    await invoke_llm(
        opts,
        "Hello",
        "System",
        make_request_args,
        handle_stream_response,
        add_tool_response,
        sse=False,
        brief=False,
    )

    captured = capsys.readouterr().out
    assert '{"ok": true}' in captured
    assert "=== [ DONE ] ===" in captured


@pytest.mark.asyncio
async def test_structured_output_sse(capsys, monkeypatch):
    opts = LLMOptions(
        model="gpt-4o-mini",
        url="https://api.openai.com/v1/chat/completions",
        api_key_name="OPENAI_API_KEY",
        output_mode="json_schema",
        output_schema={"type": "object"},
    )

    def make_request_args(_opts, _prompt, _system_prompt):
        return RequestArgs(data={"messages": []}, headers={})

    def handle_stream_response(_line, _started):
        yield StreamResponse(id="", delta='{"ok": true}', type="output_text")
        yield StreamResponse(id="", delta="", type="block_end")

    def add_tool_response(_messages, _tool_calls):
        return None

    monkeypatch.setattr(
        "buzzllm.llm.requests.post",
        lambda *args, **kwargs: FakeResponse(["data: {}"]),
    )

    await invoke_llm(
        opts,
        "Hello",
        "System",
        make_request_args,
        handle_stream_response,
        add_tool_response,
        sse=True,
        brief=False,
    )

    captured = capsys.readouterr().out
    assert "event: output_structured" in captured
    assert "event: output_text" not in captured
