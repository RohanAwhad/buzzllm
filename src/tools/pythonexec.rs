use async_trait::async_trait;
use bollard::container::{Config, CreateContainerOptions, KillContainerOptions};
use bollard::models::{HostConfig, PortBinding};
use bollard::Docker;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

const DOCKER_IMAGE: &str = "buzz/python-exec:latest";
const MAX_OUTPUT_LENGTH: usize = 10000;

#[derive(Debug, Default)]
struct PythonExecState {
    container_id: Option<String>,
    port: Option<u16>,
}

pub struct PythonExecute {
    state: Arc<Mutex<PythonExecState>>,
}

impl PythonExecute {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for PythonExecute {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(PythonExecState::default())),
        }
    }
}

fn find_available_port() -> anyhow::Result<u16> {
    for port in 3000..=7990u16 {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
    }
    Err(anyhow::anyhow!(
        "No available ports found in range 3000-7990"
    ))
}

async fn start_container(state: &mut PythonExecState, mem: &str) -> anyhow::Result<()> {
    if state.container_id.is_some() {
        return Ok(());
    }

    let port = find_available_port()?;
    let docker = Docker::connect_with_local_defaults()
        .map_err(|e| anyhow::anyhow!("Failed to connect to Docker: {}", e))?;

    let container_name = format!("pyexec-{}", &uuid::Uuid::new_v4().to_string()[..8]);

    let mut port_bindings = HashMap::new();
    port_bindings.insert(
        "8787/tcp".to_string(),
        Some(vec![PortBinding {
            host_ip: Some("0.0.0.0".to_string()),
            host_port: Some(port.to_string()),
        }]),
    );

    let mut exposed_ports = HashMap::new();
    exposed_ports.insert("8787/tcp".to_string(), HashMap::new());

    // Parse memory limit
    let memory = if mem.ends_with('m') || mem.ends_with('M') {
        mem.trim_end_matches(['m', 'M'])
            .parse::<i64>()
            .unwrap_or(512)
            * 1024
            * 1024
    } else if mem.ends_with('g') || mem.ends_with('G') {
        mem.trim_end_matches(['g', 'G']).parse::<i64>().unwrap_or(1) * 1024 * 1024 * 1024
    } else {
        512 * 1024 * 1024
    };

    let config = Config {
        image: Some(DOCKER_IMAGE.to_string()),
        exposed_ports: Some(exposed_ports),
        host_config: Some(HostConfig {
            memory: Some(memory),
            auto_remove: Some(true),
            network_mode: Some("bridge".to_string()),
            port_bindings: Some(port_bindings),
            ..Default::default()
        }),
        ..Default::default()
    };

    let container = docker
        .create_container(
            Some(CreateContainerOptions::<String> {
                name: container_name,
                platform: None,
            }),
            config,
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create container: {}", e))?;

    docker
        .start_container::<String>(&container.id, None)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start container: {}", e))?;

    // Wait for container to be ready
    let mut ready = false;
    for _ in 0..10 {
        if TcpStream::connect(format!("127.0.0.1:{}", port))
            .await
            .is_ok()
        {
            ready = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    if !ready {
        let _ = docker
            .kill_container(
                &container.id,
                Some(KillContainerOptions { signal: "SIGKILL" }),
            )
            .await;
        return Err(anyhow::anyhow!("Container failed to start or become ready"));
    }

    // Extra wait for internal service startup
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    eprintln!("Python execution container started on port {}", port);

    state.container_id = Some(container.id);
    state.port = Some(port);

    Ok(())
}

fn truncate_output(s: &str) -> String {
    if s.len() > MAX_OUTPUT_LENGTH {
        format!("{}\n... (truncated)", &s[..MAX_OUTPUT_LENGTH])
    } else {
        s.to_string()
    }
}

#[async_trait]
impl super::Tool for PythonExecute {
    fn name(&self) -> &str {
        "python_execute"
    }

    fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "python_execute",
                "description": "Execute Python code in an isolated Docker container with persistent IPython kernel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source code to execute"},
                        "mem": {"type": "string", "description": "Container memory limit (e.g. '512m', '1g')"},
                        "timeout": {"type": "integer", "description": "Socket timeout in seconds"},
                    },
                    "required": ["code"],
                }
            }
        })
    }

    fn anthropic_schema(&self) -> Value {
        json!({
            "name": "python_execute",
            "description": "Execute Python code in an isolated Docker container with persistent IPython kernel.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to execute"},
                    "mem": {"type": "string", "description": "Container memory limit (e.g. '512m', '1g')"},
                    "timeout": {"type": "integer", "description": "Socket timeout in seconds"},
                },
                "required": ["code"],
            }
        })
    }

    async fn execute(&self, args: Value) -> Value {
        let code = args.get("code").and_then(|v| v.as_str()).unwrap_or("");
        let mem = args.get("mem").and_then(|v| v.as_str()).unwrap_or("512m");
        let timeout_secs = args.get("timeout").and_then(|v| v.as_u64()).unwrap_or(30);

        if code.is_empty() || code.trim().is_empty() {
            return json!({"stdout": "", "stderr": "No code provided", "result": null});
        }

        let mut state = self.state.lock().await;

        // Ensure container is running
        if let Err(e) = start_container(&mut state, mem).await {
            return json!({"stdout": "", "stderr": format!("Execution failed: {}", e), "result": null});
        }

        let port = match state.port {
            Some(p) => p,
            None => {
                return json!({"stdout": "", "stderr": "Execution failed: no port", "result": null})
            }
        };

        // Drop the lock before doing I/O
        drop(state);

        // Connect and send code
        let result = tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), async {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).await?;
            let request = serde_json::to_string(&json!({"code": code}))?;
            stream.write_all(request.as_bytes()).await?;

            let mut response_data = Vec::new();
            let mut buf = [0u8; 4096];

            loop {
                let n = stream.read(&mut buf).await?;
                if n == 0 {
                    break;
                }
                response_data.extend_from_slice(&buf[..n]);

                // Try parsing as JSON
                if serde_json::from_slice::<Value>(&response_data).is_ok() {
                    break;
                }
            }

            if response_data.is_empty() {
                return Ok::<Value, anyhow::Error>(
                    json!({"stdout": "", "stderr": "No response from container", "result": null}),
                );
            }

            let mut response: Value = serde_json::from_slice(&response_data)?;

            // Truncate large outputs
            if let Some(stdout) = response.get("stdout").and_then(|v| v.as_str()) {
                if stdout.len() > MAX_OUTPUT_LENGTH {
                    response["stdout"] = json!(truncate_output(stdout));
                }
            }
            if let Some(stderr) = response.get("stderr").and_then(|v| v.as_str()) {
                if stderr.len() > MAX_OUTPUT_LENGTH {
                    response["stderr"] = json!(truncate_output(stderr));
                }
            }

            Ok(response)
        })
        .await;

        match result {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                json!({"stdout": "", "stderr": format!("Execution failed: {}", e), "result": null})
            }
            Err(_) => json!({"stdout": "", "stderr": "Execution timed out", "result": null}),
        }
    }

    async fn cleanup(&self) {
        let mut state = self.state.lock().await;
        if let Some(ref container_id) = state.container_id {
            if let Ok(docker) = Docker::connect_with_local_defaults() {
                let _ = docker
                    .kill_container(
                        container_id,
                        Some(KillContainerOptions { signal: "SIGKILL" }),
                    )
                    .await;
                tracing::info!("Python execution container stopped");
            }
        }
        state.container_id = None;
        state.port = None;
    }
}
