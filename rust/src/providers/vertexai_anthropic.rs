use std::collections::HashMap;
use serde_json::json;
use crate::types::{LlmOptions, RequestArgs};

pub fn make_request_args(opts: &LlmOptions, prompt: &str, system_prompt: &str) -> anyhow::Result<RequestArgs> {
    let mut data = json!({
        "anthropic_version": "vertex-2023-10-16",
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
    });

    if opts.think {
        data["max_tokens"] = json!(32000);
        data["thinking"] = json!({"type": "enabled", "budget_tokens": 24000});
    } else {
        data["max_tokens"] = json!(opts.max_tokens.unwrap_or(8192));
    }

    if let Some(ref tools) = opts.tools {
        data["tools"] = json!(tools);
    }

    let mut headers = HashMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());

    // Get access token from gcloud CLI
    let output = std::process::Command::new("gcloud")
        .args(["auth", "print-access-token"])
        .output()
        .map_err(|e| anyhow::anyhow!("failed to run gcloud: {}", e))?;

    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    headers.insert("Authorization".to_string(), format!("Bearer {}", token));

    Ok(RequestArgs { data, headers })
}
