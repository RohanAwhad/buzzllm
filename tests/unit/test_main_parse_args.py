import pytest
import sys
from unittest.mock import patch


class TestParseArgs:
    def test_required_positional_args(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "gpt-4o-mini",
            "https://api.openai.com/v1/chat/completions",
            "Hello world",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "OPENAI_API_KEY",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.model == "gpt-4o-mini"
        assert args.url == "https://api.openai.com/v1/chat/completions"
        assert args.prompt == "Hello world"
        assert args.provider == "openai-chat"
        assert args.api_key_name == "OPENAI_API_KEY"

    def test_default_values(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.max_tokens == 8192
        assert args.temperature == 0.8
        assert args.think is False
        assert args.sse is False

    def test_custom_max_tokens(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "--max-tokens",
            "4096",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.max_tokens == 4096

    def test_custom_temperature(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "--temperature",
            "0.5",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.temperature == 0.5

    def test_think_flag(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "--think",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.think is True

    def test_sse_flag_short(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "-S",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.sse is True

    def test_sse_flag_long(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "--sse",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.sse is True

    def test_provider_choices(self):
        from buzzllm.main import parse_args

        for provider in ["openai-chat", "openai-responses", "anthropic", "vertexai-anthropic"]:
            test_args = [
                "buzzllm",
                "model",
                "http://localhost",
                "prompt",
                "--provider",
                provider,
                "--api-key-name",
                "KEY",
            ]

            with patch.object(sys, "argv", test_args):
                args = parse_args()

            assert args.provider == provider

    def test_invalid_provider_exits(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "invalid-provider",
            "--api-key-name",
            "KEY",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                parse_args()

    def test_missing_provider_exits(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--api-key-name",
            "KEY",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                parse_args()

    def test_missing_api_key_name_exits(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "openai-chat",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                parse_args()

    def test_system_prompt_custom_text(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "--system-prompt",
            "You are a helpful assistant",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.system_prompt == "You are a helpful assistant"

    def test_system_prompt_template_name(self):
        from buzzllm.main import parse_args

        test_args = [
            "buzzllm",
            "model",
            "http://localhost",
            "prompt",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "--system-prompt",
            "websearch",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.system_prompt == "websearch"
