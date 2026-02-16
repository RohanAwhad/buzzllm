import os
import pytest


# === Skip conditions ===


def docker_available() -> bool:
    """Check if Docker daemon is running and accessible"""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def openai_api_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def anthropic_api_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


skip_if_no_docker = pytest.mark.skipif(
    not docker_available(), reason="Docker daemon not available"
)

skip_if_no_openai = pytest.mark.skipif(
    not openai_api_available(), reason="OPENAI_API_KEY not set"
)

skip_if_no_anthropic = pytest.mark.skipif(
    not anthropic_api_available(), reason="ANTHROPIC_API_KEY not set"
)


# === Reset global state ===


@pytest.fixture(autouse=True)
def reset_tool_state():
    """Reset global tool state between tests"""
    from buzzllm.tools import utils
    from buzzllm import llm

    utils.AVAILABLE_TOOLS.clear()
    llm.TOOL_CALLS.clear()
    llm.current_tool_call_id = ""
    llm.last_openai_response_id = ""

    yield

    utils.AVAILABLE_TOOLS.clear()
    llm.TOOL_CALLS.clear()
    llm.current_tool_call_id = ""
    llm.last_openai_response_id = ""


# === LLM fixtures ===


@pytest.fixture
def sample_llm_options():
    """Basic LLMOptions fixture"""
    from buzzllm.llm import LLMOptions

    return LLMOptions(
        model="gpt-4o-mini",
        url="https://api.openai.com/v1/chat/completions",
        api_key_name="OPENAI_API_KEY",
        max_tokens=1024,
        temperature=0.5,
    )


@pytest.fixture
def sample_tool_calls():
    """Sample ToolCall dict for testing message conversion"""
    from buzzllm.llm import ToolCall

    return {
        "call_1": ToolCall(
            id="call_1",
            name="search_web",
            arguments='{"query": "python tutorial"}',
            executed=True,
            result="Search results...",
        ),
    }


# === Tool schema fixtures ===


@pytest.fixture
def sample_function_with_docstring():
    """Sample function for schema conversion"""

    def my_func(name: str, count: int = 10) -> str:
        """This is a sample function

        :param name: The name to use
        :param count: Number of items
        """
        return f"{name}: {count}"

    return my_func


@pytest.fixture
def sample_function_without_docstring():
    """Function without docstring - should raise error"""

    def no_doc_func(x: int):
        pass

    return no_doc_func


# === Codesearch fixtures ===


@pytest.fixture
def temp_cwd(tmp_path, monkeypatch):
    """Set up a temp directory as CWD for codesearch tests"""
    (tmp_path / "test.py").write_text("def hello():\n    print('hello')\n")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.py").write_text("x = 1\n")

    import buzzllm.tools.codesearch as codesearch_module

    monkeypatch.setattr(codesearch_module, "CWD", tmp_path)
    return tmp_path


# === Websearch mock fixtures ===


@pytest.fixture
def mock_duckduckgo_html():
    """Mock HTML response from DuckDuckGo lite"""
    return """
    <html><body>
    <table>
        <tr><td>1. </td><td><a href="https://example.com">Example Title</a></td></tr>
        <tr><td></td><td>This is the description</td></tr>
        <tr><td></td><td>https://example.com</td></tr>
    </table>
    </body></html>
    """


@pytest.fixture
def mock_brave_json():
    """Mock JSON response from Brave Search API"""
    return {
        "web": {
            "results": [
                {
                    "title": "Example Title",
                    "url": "https://example.com",
                    "description": "Example description",
                }
            ]
        }
    }


# === Environment fixtures ===


@pytest.fixture
def env_with_api_keys(monkeypatch):
    """Set up environment with fake API keys for unit tests"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-xxx")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key-xxx")
    monkeypatch.setenv("BRAVE_SEARCH_AI_API_KEY", "test-brave-key-xxx")
