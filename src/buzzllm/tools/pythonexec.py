import docker
import json
import socket
import atexit
import uuid
import time
import threading
from typing import Optional


_DOCKER_IMAGE = "buzz/python-exec:latest"
_client = None
_container = None
_port = None
_lock = threading.Lock()


def _get_docker_client():
    """Get or create Docker client"""
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def _find_available_port(start_port: int = 3000, end_port: int = 7990) -> Optional[int]:
    """Find the first available port in the given range"""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("localhost", port))
                return port
            except OSError:
                continue
    return None


def _start_container(mem: str = "512m", timeout: int = 600):
    """Start the Python execution container"""
    global _container, _port

    with _lock:
        if _container:
            return

        # Find available port
        _port = _find_available_port()
        if _port is None:
            raise RuntimeError("No available ports found in range 3000-7990")

        client = _get_docker_client()
        container_name = f"pyexec-{uuid.uuid4().hex[:8]}"

        try:
            _container = client.containers.run(
                _DOCKER_IMAGE,
                detach=True,
                remove=True,
                name=container_name,
                mem_limit=mem,
                network_mode="bridge",
                ports={"8787/tcp": _port},
                auto_remove=True,
            )

            # Wait for container to be ready
            max_wait = 10
            for _ in range(max_wait):
                try:
                    _container.reload()
                    if _container.status == "running":
                        # Test connection
                        with socket.create_connection(("localhost", _port), timeout=2):
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("Container failed to start or become ready")

            time.sleep(2)  # wait for internal service to start
            print(f"Python execution container started on port {_port}")

        except Exception as e:
            _container = None
            _port = None
            raise RuntimeError(f"Failed to start container: {str(e)}")


def _kill_container():
    """Kill and cleanup the container"""
    global _container, _port

    with _lock:
        if _container:
            try:
                _container.kill()
                print("Python execution container stopped")
            except Exception:
                pass
            finally:
                _container = None
                _port = None


# Register cleanup function
atexit.register(_kill_container)


def python_execute(code: str, mem: str = "512m", timeout: int = 30) -> dict:
    """
    Execute Python code in an isolated Docker container with persistent IPython kernel.

    Args:
        code: Python source code to execute
        mem: Memory limit for container (e.g. "512m", "1g")
        timeout: Socket timeout in seconds

    Returns:
        dict: Contains 'stdout', 'stderr', and 'result' keys with execution output
    """
    if not code or not code.strip():
        return {"stdout": "", "stderr": "No code provided", "result": None}

    try:
        # Ensure container is running
        _start_container(mem)

        # Send code to container
        with socket.create_connection(("localhost", _port), timeout=timeout) as sock:
            request = json.dumps({"code": code})
            sock.sendall(request.encode())

            # Receive response
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk

                # Try to parse JSON to see if we have complete response
                try:
                    json.loads(response_data.decode())
                    break
                except json.JSONDecodeError:
                    continue

            if not response_data:
                return {
                    "stdout": "",
                    "stderr": "No response from container",
                    "result": None,
                }

            response = json.loads(response_data.decode())

            # Truncate large outputs
            max_length = 10000
            if len(response.get("stdout", "")) > max_length:
                response["stdout"] = (
                    response["stdout"][:max_length] + "\n... (truncated)"
                )
            if len(response.get("stderr", "")) > max_length:
                response["stderr"] = (
                    response["stderr"][:max_length] + "\n... (truncated)"
                )

            return response

    except Exception as e:
        return {"stdout": "", "stderr": f"Execution failed: {str(e)}", "result": None}


def cleanup_python_exec():
    """Manually cleanup the container (called from main cleanup)"""
    _kill_container()
