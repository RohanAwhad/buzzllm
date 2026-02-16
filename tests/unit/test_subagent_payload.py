from unittest.mock import patch

from buzzllm import subagent


class TestSubagentPayload:
    def test_payload_includes_context_fields(self):
        subagent.configure_subagent_context(
            model="gpt-4o-mini",
            provider="openai-chat",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            think=True,
            temperature=0.4,
            max_tokens=1234,
        )

        payload = subagent.build_subagent_payload(
            prompt="Hello",
            system_prompt="System",
            tool_subset=["search_web"],
        )

        assert payload["model"] == "gpt-4o-mini"
        assert payload["provider"] == "openai-chat"
        assert payload["url"] == "https://api.openai.com/v1/chat/completions"
        assert payload["api_key_name"] == "OPENAI_API_KEY"
        assert payload["prompt"] == "Hello"
        assert payload["system_prompt"] == "System"
        assert payload["tool_subset"] == ["search_web"]
        assert payload["think"] is True
        assert payload["temperature"] == 0.4
        assert payload["max_tokens"] == 1234

    def test_unknown_tool_subset_returns_error(self):
        subagent.configure_subagent_context(
            model="gpt-4o-mini",
            provider="openai-chat",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            think=False,
            temperature=0.8,
            max_tokens=1024,
        )

        with patch("buzzllm.subagent.subprocess.run") as mock_run:
            result = subagent.call_subagent(
                prompt="Hello",
                system_prompt="System",
                tool_subset=["unknown_tool"],
            )

        assert result.startswith("Error: Unknown tool names")
        mock_run.assert_not_called()
