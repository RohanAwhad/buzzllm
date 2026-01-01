import pytest
import json
from unittest.mock import patch, MagicMock

from tests.conftest import skip_if_no_docker


class TestFindAvailablePort:
    def test_finds_port_in_range(self):
        from buzzllm.tools.pythonexec import _find_available_port

        with patch("socket.socket") as mock_socket:
            mock_sock_instance = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_sock_instance
            # First port fails, second succeeds
            mock_sock_instance.bind.side_effect = [OSError(), None]

            port = _find_available_port(start_port=3000, end_port=3010)

            assert port == 3001

    def test_returns_none_when_no_ports_available(self):
        from buzzllm.tools.pythonexec import _find_available_port

        with patch("socket.socket") as mock_socket:
            mock_sock_instance = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_sock_instance
            mock_sock_instance.bind.side_effect = OSError()

            port = _find_available_port(start_port=3000, end_port=3002)

            assert port is None


class TestPythonExecute:
    def test_empty_code_returns_error(self):
        from buzzllm.tools.pythonexec import python_execute

        result = python_execute("")
        assert result["stderr"] == "No code provided"
        assert result["stdout"] == ""

    def test_whitespace_only_returns_error(self):
        from buzzllm.tools.pythonexec import python_execute

        result = python_execute("   \n\t  ")
        assert result["stderr"] == "No code provided"

    @patch("buzzllm.tools.pythonexec._start_container")
    @patch("socket.create_connection")
    def test_sends_code_to_container(self, mock_conn, mock_start):
        from buzzllm.tools.pythonexec import python_execute

        # Mock socket response
        mock_sock = MagicMock()
        mock_conn.return_value.__enter__.return_value = mock_sock
        mock_sock.recv.side_effect = [
            json.dumps({"stdout": "hello\n", "stderr": "", "result": None}).encode(),
            b"",
        ]

        result = python_execute("print('hello')")

        assert result["stdout"] == "hello\n"
        assert result["stderr"] == ""
        mock_start.assert_called_once()

    @patch("buzzllm.tools.pythonexec._start_container")
    @patch("socket.create_connection")
    def test_truncates_large_output(self, mock_conn, mock_start):
        from buzzllm.tools.pythonexec import python_execute

        large_output = "x" * 20000
        response_json = json.dumps({"stdout": large_output, "stderr": "", "result": None}).encode()
        mock_sock = MagicMock()
        mock_conn.return_value.__enter__.return_value = mock_sock
        # Return the full response in one chunk, then empty to signal end
        mock_sock.recv.side_effect = [response_json, b""]

        result = python_execute("print('x' * 20000)")

        # Check truncation happened (10000 chars + "... (truncated)" suffix)
        assert len(result["stdout"]) < 20000
        assert "truncated" in result["stdout"]

    @patch("buzzllm.tools.pythonexec._start_container")
    def test_handles_container_start_failure(self, mock_start):
        from buzzllm.tools.pythonexec import python_execute

        mock_start.side_effect = RuntimeError("Docker not available")

        result = python_execute("print('test')")

        assert "Execution failed" in result["stderr"]

    @patch("buzzllm.tools.pythonexec._start_container")
    @patch("socket.create_connection")
    def test_handles_no_response(self, mock_conn, mock_start):
        from buzzllm.tools.pythonexec import python_execute

        mock_sock = MagicMock()
        mock_conn.return_value.__enter__.return_value = mock_sock
        mock_sock.recv.return_value = b""

        result = python_execute("print('test')")

        assert result["stderr"] == "No response from container"


class TestContainerManagement:
    @patch("buzzllm.tools.pythonexec._get_docker_client")
    @patch("buzzllm.tools.pythonexec._find_available_port")
    @patch("socket.create_connection")
    @patch("time.sleep")
    def test_start_container_creates_container(
        self, mock_sleep, mock_conn, mock_port, mock_client
    ):
        from buzzllm.tools import pythonexec

        # Reset global state
        pythonexec._container = None
        pythonexec._port = None

        mock_port.return_value = 5000
        mock_docker = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_docker.containers.run.return_value = mock_container
        mock_client.return_value = mock_docker

        pythonexec._start_container()

        mock_docker.containers.run.assert_called_once()
        assert pythonexec._port == 5000

        # Cleanup
        pythonexec._container = None
        pythonexec._port = None

    def test_kill_container_cleans_up(self):
        from buzzllm.tools import pythonexec

        mock_container = MagicMock()
        pythonexec._container = mock_container
        pythonexec._port = 5000

        pythonexec._kill_container()

        mock_container.kill.assert_called_once()
        assert pythonexec._container is None
        assert pythonexec._port is None


@pytest.mark.docker
@skip_if_no_docker
class TestPythonExecuteWithDocker:
    """Tests that require Docker to be running"""

    def test_executes_simple_code(self):
        from buzzllm.tools.pythonexec import python_execute, cleanup_python_exec

        try:
            result = python_execute("print(2 + 2)")
            assert "4" in result["stdout"]
            assert result["stderr"] == ""
        finally:
            cleanup_python_exec()

    def test_returns_computation_result(self):
        from buzzllm.tools.pythonexec import python_execute, cleanup_python_exec

        try:
            result = python_execute("x = 10 * 5\nx")
            # Result should contain 50
            assert result["result"] == 50 or "50" in str(result)
        finally:
            cleanup_python_exec()

    def test_captures_errors(self):
        from buzzllm.tools.pythonexec import python_execute, cleanup_python_exec

        try:
            result = python_execute("raise ValueError('test error')")
            assert "ValueError" in result["stderr"] or "error" in result["stderr"].lower()
        finally:
            cleanup_python_exec()
