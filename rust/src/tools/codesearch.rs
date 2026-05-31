use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tokio::process::Command;

pub(crate) fn validate_path(path_str: &str, cwd: &Path) -> anyhow::Result<PathBuf> {
    let resolved = cwd
        .join(path_str)
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("invalid path '{}': {}", path_str, e))?;
    if !resolved.starts_with(cwd) {
        return Err(anyhow::anyhow!(
            "Path outside CWD not allowed: {}",
            resolved.display()
        ));
    }
    Ok(resolved)
}

fn paginate_results(results: &[String], limit: i64, offset: i64) -> Value {
    let total = results.len() as i64;
    let offset = offset.max(0);

    let (paginated, has_more) = if limit <= 0 {
        let p: Vec<&String> = results.iter().skip(offset as usize).collect();
        (p, false)
    } else {
        let end = offset + limit;
        let p: Vec<&String> = results
            .iter()
            .skip(offset as usize)
            .take(limit as usize)
            .collect();
        (p, end < total)
    };

    let returned = paginated.len();
    json!({
        "results": paginated,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "returned": returned,
    })
}

fn get_cwd() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

const BASH_FIND_DESC: &str = r#"This function uses `rg --files` to list files efficiently:

**Core command**: `rg --files [path]`
- Lists all files recursively
- Respects `.gitignore` automatically
- Much faster than `find` for file listing

**Key usage patterns**:
- Basic file listing: `rg --files`
- With glob filter: `rg --files --glob "*.py"`
- In specific directory: `rg --files /path/to/dir`

Falls back to `find` for directories (`type_filter="d"`)"#;

const BASH_RIPGREP_DESC: &str = r#"This function uses `rg` for text searching within files:

**Core command**: `rg "pattern" [path]`
- Searches for regex patterns in file contents
- Fast, recursive, respects `.gitignore`
- Shows filename:line:content by default

**Common extra_args**: `--ignore-case`, `--word-regexp`, `--context 3`, `--type py`"#;

// --- BashFind ---

pub struct BashFind;

#[async_trait]
impl super::Tool for BashFind {
    fn name(&self) -> &str {
        "bash_find"
    }

    fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "bash_find",
                "description": BASH_FIND_DESC,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path to search in, defaults to current directory"},
                        "name": {"type": "string", "description": "Glob pattern to filter filenames"},
                        "type_filter": {"type": "string", "description": "Filter by type ('d' for directories)"},
                        "extra_args": {"type": "string", "description": "Additional command line arguments"},
                        "limit": {"type": "integer", "description": "Maximum number of results (0 = all)"},
                        "offset": {"type": "integer", "description": "Number of results to skip"},
                    },
                    "required": [],
                }
            }
        })
    }

    fn anthropic_schema(&self) -> Value {
        json!({
            "name": "bash_find",
            "description": BASH_FIND_DESC,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to search in, defaults to current directory"},
                    "name": {"type": "string", "description": "Glob pattern to filter filenames"},
                    "type_filter": {"type": "string", "description": "Filter by type ('d' for directories)"},
                    "extra_args": {"type": "string", "description": "Additional command line arguments"},
                    "limit": {"type": "integer", "description": "Maximum number of results (0 = all)"},
                    "offset": {"type": "integer", "description": "Number of results to skip"},
                },
                "required": [],
            }
        })
    }

    async fn execute(&self, args: Value) -> Value {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let type_filter = args
            .get("type_filter")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let extra_args = args
            .get("extra_args")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(20);
        let offset = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0);

        let cwd = get_cwd();

        let validated_path = match validate_path(path, &cwd) {
            Ok(p) => p,
            Err(e) => return json!({"error": e.to_string()}),
        };

        let mut cmd_parts: Vec<String>;

        if type_filter == "d" {
            cmd_parts = vec![
                "find".to_string(),
                validated_path.to_string_lossy().to_string(),
                "-type".to_string(),
                "d".to_string(),
            ];
            if !name.is_empty() {
                cmd_parts.push("-name".to_string());
                cmd_parts.push(name.to_string());
            }
        } else {
            cmd_parts = vec!["rg".to_string(), "--files".to_string()];
            if !name.is_empty() {
                cmd_parts.push("--glob".to_string());
                cmd_parts.push(name.to_string());
            }
            if path != "." {
                cmd_parts.push(validated_path.to_string_lossy().to_string());
            }
        }

        if !extra_args.is_empty() {
            cmd_parts.extend(extra_args.split_whitespace().map(String::from));
        }

        let result = match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            Command::new(&cmd_parts[0])
                .args(&cmd_parts[1..])
                .current_dir(&cwd)
                .output(),
        )
        .await
        {
            Ok(Ok(output)) => {
                if !output.status.success() {
                    return json!({"error": format!("Command failed: {}", String::from_utf8_lossy(&output.stderr))});
                }
                let stdout = String::from_utf8_lossy(&output.stdout);
                let trimmed = stdout.trim();
                if trimmed.is_empty() {
                    Vec::new()
                } else {
                    trimmed.split('\n').map(String::from).collect()
                }
            }
            Ok(Err(e)) => return json!({"error": format!("Command failed: {}", e)}),
            Err(_) => return json!({"error": "Command timed out after 30 seconds"}),
        };

        paginate_results(&result, limit, offset)
    }
}

// --- BashRipgrep ---

pub struct BashRipgrep;

#[async_trait]
impl super::Tool for BashRipgrep {
    fn name(&self) -> &str {
        "bash_ripgrep"
    }

    fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "bash_ripgrep",
                "description": BASH_RIPGREP_DESC,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "files": {"type": "string", "description": "Path to search in"},
                        "extra_args": {"type": "string", "description": "Additional CLI args"},
                        "limit": {"type": "integer", "description": "Maximum results (0 = all)"},
                        "offset": {"type": "integer", "description": "Results to skip"},
                    },
                    "required": ["pattern"],
                }
            }
        })
    }

    fn anthropic_schema(&self) -> Value {
        json!({
            "name": "bash_ripgrep",
            "description": BASH_RIPGREP_DESC,
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "files": {"type": "string", "description": "Path to search in"},
                    "extra_args": {"type": "string", "description": "Additional CLI args"},
                    "limit": {"type": "integer", "description": "Maximum results (0 = all)"},
                    "offset": {"type": "integer", "description": "Results to skip"},
                },
                "required": ["pattern"],
            }
        })
    }

    async fn execute(&self, args: Value) -> Value {
        let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return json!({"error": "pattern is required"}),
        };
        let files = args.get("files").and_then(|v| v.as_str()).unwrap_or(".");
        let extra_args = args
            .get("extra_args")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(20);
        let offset = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0);

        let cwd = get_cwd();

        let validated_files = match validate_path(files, &cwd) {
            Ok(p) => p,
            Err(e) => return json!({"error": e.to_string()}),
        };

        let mut cmd = Command::new("rg");
        cmd.arg(pattern);
        if files != "." {
            cmd.arg(validated_files.to_string_lossy().as_ref());
        }
        if !extra_args.is_empty() {
            for arg in extra_args.split_whitespace() {
                cmd.arg(arg);
            }
        }
        cmd.current_dir(&cwd);

        let result = match tokio::time::timeout(std::time::Duration::from_secs(30), cmd.output())
            .await
        {
            Ok(Ok(output)) => {
                if !output.status.success() {
                    return json!({"error": format!("No matches found or command failed: {}", String::from_utf8_lossy(&output.stderr))});
                }
                let stdout = String::from_utf8_lossy(&output.stdout);
                let trimmed = stdout.trim();
                if trimmed.is_empty() {
                    Vec::new()
                } else {
                    trimmed.split('\n').map(String::from).collect()
                }
            }
            Ok(Err(e)) => return json!({"error": format!("Command failed: {}", e)}),
            Err(_) => return json!({"error": "Command timed out after 30 seconds"}),
        };

        paginate_results(&result, limit, offset)
    }
}

// --- BashRead ---

pub struct BashRead;

#[async_trait]
impl super::Tool for BashRead {
    fn name(&self) -> &str {
        "bash_read"
    }

    fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "bash_read",
                "description": "Read file contents with pagination, restricted to current working directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path to the file to read"},
                        "limit": {"type": "integer", "description": "Maximum number of lines (0 = all)"},
                        "offset": {"type": "integer", "description": "Lines to skip"},
                    },
                    "required": ["filepath"],
                }
            }
        })
    }

    fn anthropic_schema(&self) -> Value {
        json!({
            "name": "bash_read",
            "description": "Read file contents with pagination, restricted to current working directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to read"},
                    "limit": {"type": "integer", "description": "Maximum number of lines (0 = all)"},
                    "offset": {"type": "integer", "description": "Lines to skip"},
                },
                "required": ["filepath"],
            }
        })
    }

    async fn execute(&self, args: Value) -> Value {
        let filepath = match args.get("filepath").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return json!({"error": "filepath is required"}),
        };
        let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(0);
        let offset = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0);

        let cwd = get_cwd();

        let validated_path = match validate_path(filepath, &cwd) {
            Ok(p) => p,
            Err(e) => return json!({"error": e.to_string()}),
        };

        let content = match std::fs::read_to_string(&validated_path) {
            Ok(c) => c,
            Err(e) => return json!({"error": format!("Failed to read file: {}", e)}),
        };

        let lines: Vec<String> = content.trim().split('\n').map(String::from).collect();
        let mut paginated = paginate_results(&lines, limit, offset);

        // Add content field
        if let Some(results) = paginated.get("results").and_then(|v| v.as_array()) {
            let content_str: String = results
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            paginated["content"] = json!(content_str);
        }

        paginated
    }
}
