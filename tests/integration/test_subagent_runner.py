import json

from buzzllm import subagent


class TestSubagentRunner:
    def test_call_subagent_parses_output_text(self, monkeypatch):
        subagent.configure_subagent_context(
            model="gpt-4o-mini",
            provider="openai-chat",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            think=False,
            temperature=0.8,
            max_tokens=1024,
        )

        sse_text = "\n".join(
            [
                "event: output_text",
                'data: {"id":"","delta":"Hello ","type":"output_text"}',
                "",
                "event: output_text",
                'data: {"id":"","delta":"world","type":"output_text"}',
                "",
            ]
        )

        captured = {}

        def fake_run(cmd, input, text, capture_output):
            captured["input"] = input

            class Result:
                returncode = 0
                stdout = sse_text
                stderr = ""

            return Result()

        monkeypatch.setattr(subagent.subprocess, "run", fake_run)

        output = subagent.call_subagent(
            prompt="Hi",
            system_prompt="System",
            tool_subset=["search_web"],
        )

        payload = json.loads(captured["input"])
        assert payload["tool_subset"] == ["search_web"]
        assert output == "Hello world"
