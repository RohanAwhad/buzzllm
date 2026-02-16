import pytest
from unittest.mock import patch, MagicMock

from buzzllm.llm import (
    LLMOptions,
    make_openai_request_args,
    make_anthropic_request_args,
    make_vertexai_anthropic_request_args,
    make_openai_responses_request_args,
)


class TestMakeOpenaiRequestArgs:
    def test_basic_request_structure(self, env_with_api_keys):
        opts = LLMOptions(
            model="gpt-4o-mini",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
        )
        args = make_openai_request_args(opts, "Hello", "You are helpful")

        assert args.data["model"] == "gpt-4o-mini"
        assert args.data["stream"] is True
        assert len(args.data["messages"]) == 2
        assert args.data["messages"][0]["role"] == "system"
        assert args.data["messages"][0]["content"] == "You are helpful"
        assert args.data["messages"][1]["role"] == "user"
        assert args.data["messages"][1]["content"] == "Hello"

    def test_headers_with_api_key(self, env_with_api_keys):
        opts = LLMOptions(
            model="gpt-4o-mini",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
        )
        args = make_openai_request_args(opts, "Hello", "System")

        assert args.headers["Content-Type"] == "application/json"
        assert args.headers["Authorization"] == "Bearer test-openai-key-xxx"

    def test_headers_without_api_key(self):
        opts = LLMOptions(
            model="gpt-4o-mini",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name=None,
        )
        args = make_openai_request_args(opts, "Hello", "System")

        assert "Authorization" not in args.headers

    def test_temperature_and_max_tokens(self, env_with_api_keys):
        opts = LLMOptions(
            model="gpt-4o-mini",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            temperature=0.5,
            max_tokens=2048,
        )
        args = make_openai_request_args(opts, "Hello", "System")

        assert args.data["temperature"] == 0.5
        assert args.data["max_tokens"] == 2048

    def test_with_tools(self, env_with_api_keys):
        tools = [{"type": "function", "function": {"name": "test"}}]
        opts = LLMOptions(
            model="gpt-4o-mini",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            tools=tools,
        )
        args = make_openai_request_args(opts, "Hello", "System")

        assert args.data["tools"] == tools

    def test_reasoning_model_uses_developer_role(self, env_with_api_keys):
        opts = LLMOptions(
            model="o3",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
        )
        args = make_openai_request_args(opts, "Hello", "System")

        assert args.data["messages"][0]["role"] == "developer"
        assert "temperature" not in args.data
        assert "max_tokens" not in args.data

    def test_gpt5_1_no_think_mode(self, env_with_api_keys):
        opts = LLMOptions(
            model="gpt-5.1",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            think=False,
        )
        args = make_openai_request_args(opts, "Hello", "System")

        assert args.data["reasoning_effort"] == "none"

    def test_gpt5_1_with_think_mode(self, env_with_api_keys):
        opts = LLMOptions(
            model="gpt-5.1",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            think=True,
        )
        args = make_openai_request_args(opts, "Hello", "System")

        assert args.data["reasoning_effort"] == "high"

    def test_structured_output_json_schema(self, env_with_api_keys):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        opts = LLMOptions(
            model="gpt-4o-mini",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            output_mode="json_schema",
            output_schema=schema,
        )
        args = make_openai_request_args(opts, "Hello", "System")

        response_format = args.data["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"] == schema

    def test_structured_output_json_object(self, env_with_api_keys):
        opts = LLMOptions(
            model="gpt-4o-mini",
            url="https://api.openai.com/v1/chat/completions",
            api_key_name="OPENAI_API_KEY",
            output_mode="json_object",
        )
        args = make_openai_request_args(opts, "Hello", "System")

        assert args.data["response_format"] == {"type": "json_object"}


class TestMakeAnthropicRequestArgs:
    def test_basic_request_structure(self, env_with_api_keys):
        opts = LLMOptions(
            model="claude-sonnet-4-20250514",
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
        )
        args = make_anthropic_request_args(opts, "Hello", "You are helpful")

        assert args.data["model"] == "claude-sonnet-4-20250514"
        assert args.data["stream"] is True
        assert args.data["system"] == "You are helpful"
        assert len(args.data["messages"]) == 1
        assert args.data["messages"][0]["role"] == "user"
        assert args.data["messages"][0]["content"] == "Hello"

    def test_headers_with_api_key(self, env_with_api_keys):
        opts = LLMOptions(
            model="claude-sonnet-4-20250514",
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
        )
        args = make_anthropic_request_args(opts, "Hello", "System")

        assert args.headers["x-api-key"] == "test-anthropic-key-xxx"
        assert args.headers["anthropic-version"] == "2023-06-01"

    def test_max_tokens_default(self, env_with_api_keys):
        opts = LLMOptions(
            model="claude-sonnet-4-20250514",
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
        )
        args = make_anthropic_request_args(opts, "Hello", "System")

        assert args.data["max_tokens"] == 8192

    def test_thinking_mode_enabled(self, env_with_api_keys):
        opts = LLMOptions(
            model="claude-sonnet-4-20250514",
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
            think=True,
        )
        args = make_anthropic_request_args(opts, "Hello", "System")

        assert args.data["max_tokens"] == 32000
        assert args.data["thinking"]["type"] == "enabled"
        assert args.data["thinking"]["budget_tokens"] == 24000

    def test_with_tools(self, env_with_api_keys):
        tools = [{"name": "test", "input_schema": {}}]
        opts = LLMOptions(
            model="claude-sonnet-4-20250514",
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
            tools=tools,
        )
        args = make_anthropic_request_args(opts, "Hello", "System")

        assert args.data["tools"] == tools

    def test_structured_output_config(self, env_with_api_keys):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        opts = LLMOptions(
            model="claude-sonnet-4-20250514",
            url="https://api.anthropic.com/v1/messages",
            api_key_name="ANTHROPIC_API_KEY",
            output_mode="json_schema",
            output_schema=schema,
        )
        args = make_anthropic_request_args(opts, "Hello", "System")

        assert args.data["output_config"]["format"]["schema"] == schema


class TestMakeVertexaiAnthropicRequestArgs:
    @patch("subprocess.run")
    def test_basic_request_structure(self, mock_run, env_with_api_keys):
        mock_run.return_value = MagicMock(stdout="fake-gcloud-token\n")

        opts = LLMOptions(
            model="claude-3-5-sonnet@20240620",
            url="https://us-central1-aiplatform.googleapis.com/...",
            api_key_name=None,
        )
        args = make_vertexai_anthropic_request_args(opts, "Hello", "System")

        assert args.data["anthropic_version"] == "vertex-2023-10-16"
        assert args.data["system"] == "System"
        assert args.data["stream"] is True

    @patch("subprocess.run")
    def test_uses_gcloud_auth(self, mock_run, env_with_api_keys):
        mock_run.return_value = MagicMock(stdout="fake-gcloud-token\n")

        opts = LLMOptions(
            model="claude-3-5-sonnet@20240620",
            url="https://us-central1-aiplatform.googleapis.com/...",
        )
        args = make_vertexai_anthropic_request_args(opts, "Hello", "System")

        mock_run.assert_called_once()
        assert "gcloud" in mock_run.call_args[0][0]
        assert args.headers["Authorization"] == "Bearer fake-gcloud-token"

    @patch("subprocess.run")
    def test_thinking_mode(self, mock_run, env_with_api_keys):
        mock_run.return_value = MagicMock(stdout="token\n")

        opts = LLMOptions(
            model="claude-3-5-sonnet@20240620",
            url="https://us-central1-aiplatform.googleapis.com/...",
            think=True,
        )
        args = make_vertexai_anthropic_request_args(opts, "Hello", "System")

        assert args.data["max_tokens"] == 32000
        assert args.data["thinking"]["type"] == "enabled"

    @patch("subprocess.run")
    def test_structured_output_config(self, mock_run, env_with_api_keys):
        mock_run.return_value = MagicMock(stdout="token\n")
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

        opts = LLMOptions(
            model="claude-3-5-sonnet@20240620",
            url="https://us-central1-aiplatform.googleapis.com/...",
            output_mode="json_schema",
            output_schema=schema,
        )
        args = make_vertexai_anthropic_request_args(opts, "Hello", "System")

        assert args.data["output_config"]["format"]["schema"] == schema


class TestMakeOpenaiResponsesRequestArgs:
    def test_basic_request_structure(self, env_with_api_keys):
        opts = LLMOptions(
            model="o3",
            url="https://api.openai.com/v1/responses",
            api_key_name="OPENAI_API_KEY",
        )
        args = make_openai_responses_request_args(opts, "Hello", "Instructions")

        assert args.data["model"] == "o3"
        assert args.data["input"][0]["role"] == "user"
        assert args.data["input"][0]["content"][0]["type"] == "input_text"
        assert args.data["input"][0]["content"][0]["text"] == "Hello"
        assert args.data["instructions"] == "Instructions"
        assert args.data["stream"] is True
        assert args.data["store"] is False
        assert args.data["reasoning"]["effort"] == "none"

    def test_tools_are_included(self, env_with_api_keys):
        tools = [{"type": "function", "function": {"name": "test"}}]
        opts = LLMOptions(
            model="o3",
            url="https://api.openai.com/v1/responses",
            api_key_name="OPENAI_API_KEY",
            tools=tools,
        )

        args = make_openai_responses_request_args(opts, "Hello", "System")
        assert args.data["tools"][0]["name"] == "test"
        assert args.data["tool_choice"] == "auto"

    def test_reasoning_mode(self, env_with_api_keys):
        opts = LLMOptions(
            model="o3",
            url="https://api.openai.com/v1/responses",
            api_key_name="OPENAI_API_KEY",
            think=True,
        )

        args = make_openai_responses_request_args(opts, "Hello", "System")
        assert args.data["reasoning"]["effort"] == "high"
        assert args.data["reasoning"]["summary"] == "detailed"

    def test_max_output_tokens_and_temperature(self, env_with_api_keys):
        opts = LLMOptions(
            model="o3",
            url="https://api.openai.com/v1/responses",
            api_key_name="OPENAI_API_KEY",
            max_tokens=2048,
            temperature=0.3,
        )

        args = make_openai_responses_request_args(opts, "Hello", "System")
        assert args.data["max_output_tokens"] == 2048
        assert args.data["temperature"] == 0.3
