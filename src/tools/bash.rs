use async_trait::async_trait;
use serde_json::{json, Value};

const BASH_DESC: &str = r#"Executes a shell command via `/bin/sh -c <cmd>`.
Use for running tests, lint, build, git commands, and invoking other CLI agents.
Returns stdout, stderr, exit code, and whether the command timed out."#;

pub struct Bash;

#[async_trait]
impl super::Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }

    fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": BASH_DESC,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string", "description": "The shell command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 120)"},
                    },
                    "required": ["cmd"],
                }
            }
        })
    }

    fn anthropic_schema(&self) -> Value {
        json!({
            "name": "bash",
            "description": BASH_DESC,
            "input_schema": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 120)"},
                },
                "required": ["cmd"],
            }
        })
    }

    async fn execute(&self, args: Value) -> Value {
        let cmd = match args.get("cmd").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return json!({"error": "cmd is required"}),
        };

        let timeout_secs = args
            .get("timeout")
            .and_then(|v| v.as_i64())
            .unwrap_or(30)
            .clamp(1, 120) as u64;

        if cmd.trim().is_empty() {
            return json!({"error": "cmd is empty"});
        }

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(cmd)
                .output(),
        )
        .await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                json!({
                    "stdout": stdout.trim(),
                    "stderr": stderr.trim(),
                    "exit_code": output.status.code().unwrap_or(-1),
                    "timed_out": false,
                })
            }
            Ok(Err(e)) => {
                json!({"error": format!("Command failed: {}", e)})
            }
            Err(_) => {
                json!({
                    "stdout": "",
                    "stderr": format!("Command timed out after {}s", timeout_secs),
                    "exit_code": -1,
                    "timed_out": true,
                })
            }
        }
    }
}
