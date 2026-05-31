# Phase 7: Python Execution Tool

## Goal

Port the `python_execute` tool to Rust. This manages a Docker container running a persistent IPython kernel, communicates via TCP socket, and handles container lifecycle (start, execute, cleanup).

## Source reference

- `src/buzzllm/tools/pythonexec.py:11-15` — module-level state (`_client`, `_container`, `_port`, `_lock`)
- `src/buzzllm/tools/pythonexec.py:18-23` — `_get_docker_client()`
- `src/buzzllm/tools/pythonexec.py:26-35` — `_find_available_port()`
- `src/buzzllm/tools/pythonexec.py:38-86` — `_start_container()`
- `src/buzzllm/tools/pythonexec.py:89-102` — `_kill_container()`
- `src/buzzllm/tools/pythonexec.py:109-171` — `python_execute()`
- `python_runtime_docker/` — Docker image (unchanged, reused as-is)

## Docker image

The existing Docker image `buzz/python-exec:latest` is unchanged. It runs a kernel server (`kernel_server.py`) that:
- Listens on port 8787 inside the container
- Accepts TCP connections
- Receives JSON: `{"code": "print('hello')"}`
- Returns JSON: `{"stdout": "hello\n", "stderr": "", "result": null}`
- Persistent IPython kernel — state carries across executions

The Rust tool talks to this same image. No changes to the Docker side.

## Deliverables

### File structure

```
rust/src/tools/
  pythonexec.rs
```

### Container state

Python uses module-level globals with a threading lock. In Rust, use a struct with `tokio::sync::Mutex`:

```rust
pub struct PythonExecState {
    container_id: Option<String>,
    port: Option<u16>,
}

pub struct PythonExecute {
    state: Arc<Mutex<PythonExecState>>,
    docker: bollard::Docker,
}
```

### Container lifecycle

#### `find_available_port()`
Scan ports 3000-7990, try to bind each, return first available.

```rust
fn find_available_port() -> Result<u16> {
    for port in 3000..=7990 {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
    }
    Err(anyhow!("no available ports in range 3000-7990"))
}
```

#### `start_container()`
Using `bollard` crate:

1. Find available port
2. Create container:
   - Image: `buzz/python-exec:latest`
   - Memory limit: 512MB (configurable)
   - Port mapping: `8787/tcp` → `{port}`
   - Network mode: bridge
   - Auto remove: true
3. Start container
4. Wait up to 10 seconds for container to become ready:
   - Check container status is "running"
   - Try TCP connection to `localhost:{port}`
5. Additional 2-second wait for internal service startup
6. On failure: clean up container, return error

#### `kill_container()`
1. Kill container by ID (ignore errors — may already be stopped)
2. Clear state (container_id = None, port = None)

#### Cleanup on exit
Register cleanup via `tokio::signal` or rely on the `invoke_llm` finally block calling `cleanup_python_exec()`.

### `python_execute()` — Tool execution

**Name**: `python_execute`

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `code` | string | (required) | Python source code to execute |
| `mem` | string | `"512m"` | Container memory limit |
| `timeout` | integer | `30` | Socket timeout in seconds |

**Returns**:
```json
{
  "stdout": "hello\n",
  "stderr": "",
  "result": null
}
```

**Behavior**:
1. Ensure container is running (start if needed)
2. Open TCP connection to `localhost:{port}` with timeout
3. Send JSON: `{"code": "{code}"}`
4. Read response until complete JSON is received:
   - Read chunks into buffer
   - Try parsing buffer as JSON after each chunk
   - Break when parse succeeds or connection closes
5. Truncate large outputs (stdout/stderr) to 10,000 characters with `"... (truncated)"` suffix
6. On any error: return `{"stdout": "", "stderr": "Execution failed: {error}", "result": null}`

### bollard API usage

```rust
use bollard::Docker;
use bollard::container::{Config, CreateContainerOptions, StartContainerOptions};
use bollard::models::HostConfig;

let docker = Docker::connect_with_local_defaults()?;

let config = Config {
    image: Some("buzz/python-exec:latest"),
    host_config: Some(HostConfig {
        memory: Some(512 * 1024 * 1024),  // 512MB
        auto_remove: Some(true),
        network_mode: Some("bridge".to_string()),
        port_bindings: Some(port_bindings),
        ..Default::default()
    }),
    exposed_ports: Some(exposed_ports),
    ..Default::default()
};

let container = docker.create_container(
    Some(CreateContainerOptions { name: &container_name, .. }),
    config,
).await?;

docker.start_container(&container.id, None::<StartContainerOptions<String>>).await?;
```

## Verification

1. `python_execute(code: "print('hello')")` returns `{"stdout": "hello\n", ...}`
2. State persists: execute `x = 42`, then `print(x)` → second call returns `42`
3. No code provided: returns `{"stderr": "No code provided"}`
4. Container starts on first call, reuses on subsequent calls
5. Large output truncation works at 10,000 chars
6. Container cleanup: after tool cleanup, container is no longer running
7. No Docker available: returns clear error message
8. End-to-end: `cargo run -- "gpt-4o-mini" ... --system-prompt pythonexec` with a math question
