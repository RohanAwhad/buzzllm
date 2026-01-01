import pytest
import subprocess
import os

from tests.conftest import skip_if_no_openai, skip_if_no_anthropic


@pytest.mark.e2e
class TestCLI:
    """End-to-end tests for the buzzllm CLI."""

    def test_help_command(self):
        """buzzllm --help should work"""
        result = subprocess.run(
            ["buzzllm", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "buzzllm" in result.stdout

    def test_help_shows_providers(self):
        """Help should list available providers"""
        result = subprocess.run(
            ["buzzllm", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "openai-chat" in result.stdout
        assert "anthropic" in result.stdout

    def test_missing_required_args_fails(self):
        """Should fail without required arguments"""
        result = subprocess.run(
            ["buzzllm"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0

    def test_missing_provider_fails(self):
        """Should fail without --provider"""
        result = subprocess.run(
            [
                "buzzllm",
                "model",
                "http://localhost",
                "prompt",
                "--api-key-name",
                "KEY",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "provider" in result.stderr.lower()

    def test_missing_api_key_name_fails(self):
        """Should fail without --api-key-name"""
        result = subprocess.run(
            [
                "buzzllm",
                "model",
                "http://localhost",
                "prompt",
                "--provider",
                "openai-chat",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "api-key-name" in result.stderr.lower()

    def test_invalid_provider_fails(self):
        """Should fail with invalid provider"""
        result = subprocess.run(
            [
                "buzzllm",
                "model",
                "http://localhost",
                "prompt",
                "--provider",
                "invalid",
                "--api-key-name",
                "KEY",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0


@pytest.mark.e2e
class TestCLIWithOpenAI:
    """E2E tests that require OpenAI API access."""

    @skip_if_no_openai
    def test_simple_openai_call(self):
        """Real CLI call to OpenAI with gpt-4.1-mini"""
        result = subprocess.run(
            [
                "buzzllm",
                "gpt-4.1-mini",
                "https://api.openai.com/v1/chat/completions",
                "Say exactly: test passed",
                "--provider",
                "openai-chat",
                "--api-key-name",
                "OPENAI_API_KEY",
                "--max-tokens",
                "20",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ},
        )
        assert result.returncode == 0
        assert "test" in result.stdout.lower() or "passed" in result.stdout.lower()

    @skip_if_no_openai
    def test_sse_output_format(self):
        """Test -S flag produces SSE format"""
        result = subprocess.run(
            [
                "buzzllm",
                "gpt-4.1-mini",
                "https://api.openai.com/v1/chat/completions",
                "Say hi",
                "--provider",
                "openai-chat",
                "--api-key-name",
                "OPENAI_API_KEY",
                "--max-tokens",
                "10",
                "-S",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ},
        )
        assert result.returncode == 0
        # SSE format should have "event:" and "data:" lines
        assert "event:" in result.stdout
        assert "data:" in result.stdout

    @skip_if_no_openai
    def test_websearch_template(self):
        """Test websearch system prompt template"""
        result = subprocess.run(
            [
                "buzzllm",
                "gpt-4.1-mini",
                "https://api.openai.com/v1/chat/completions",
                "What is 2+2? Answer directly without searching.",
                "--provider",
                "openai-chat",
                "--api-key-name",
                "OPENAI_API_KEY",
                "--system-prompt",
                "websearch",
                "--max-tokens",
                "50",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ},
        )
        # Should complete without error (tools are registered but may not be called)
        assert result.returncode == 0


@pytest.mark.e2e
class TestCLIWithAnthropic:
    """E2E tests that require Anthropic API access."""

    @skip_if_no_anthropic
    def test_simple_anthropic_call(self):
        """Real CLI call to Anthropic with Haiku 4.5"""
        result = subprocess.run(
            [
                "buzzllm",
                "claude-3-5-haiku-20241022",
                "https://api.anthropic.com/v1/messages",
                "Say exactly: test passed",
                "--provider",
                "anthropic",
                "--api-key-name",
                "ANTHROPIC_API_KEY",
                "--max-tokens",
                "20",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ},
        )
        assert result.returncode == 0
        assert "test" in result.stdout.lower() or "passed" in result.stdout.lower()

    @skip_if_no_anthropic
    def test_thinking_mode_flag(self):
        """Test --think flag with Anthropic (Sonnet supports thinking)"""
        result = subprocess.run(
            [
                "buzzllm",
                "claude-sonnet-4-20250514",
                "https://api.anthropic.com/v1/messages",
                "What is 5*5?",
                "--provider",
                "anthropic",
                "--api-key-name",
                "ANTHROPIC_API_KEY",
                "--think",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ},
        )
        assert result.returncode == 0
        assert "25" in result.stdout

    @skip_if_no_anthropic
    def test_codesearch_template(self):
        """Test codesearch system prompt template"""
        result = subprocess.run(
            [
                "buzzllm",
                "claude-3-5-haiku-20241022",
                "https://api.anthropic.com/v1/messages",
                "What is 1+1? Answer directly.",
                "--provider",
                "anthropic",
                "--api-key-name",
                "ANTHROPIC_API_KEY",
                "--system-prompt",
                "codesearch",
                "--max-tokens",
                "50",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ},
        )
        assert result.returncode == 0
