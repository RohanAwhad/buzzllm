use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::PathBuf;

fn get_cwd() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn resolve_path(filepath: &str, cwd: &PathBuf) -> Result<PathBuf, String> {
    let path = std::path::Path::new(filepath);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(filepath)
    };
    let is_absolute = path.is_absolute();

    let resolved_canonical = if resolved.exists() {
        resolved
            .canonicalize()
            .map_err(|e| format!("invalid path '{}': {}", filepath, e))?
    } else {
        let parent = resolved
            .parent()
            .ok_or_else(|| format!("no parent directory for '{}'", filepath))?;
        if !parent.exists() {
            return Err(format!(
                "parent directory does not exist: {}",
                parent.display()
            ));
        }
        let parent_canonical = parent
            .canonicalize()
            .map_err(|e| format!("invalid path '{}': {}", filepath, e))?;
        parent_canonical.join(resolved.file_name().unwrap())
    };

    if is_absolute {
        return Ok(resolved_canonical);
    }

    if !resolved_canonical.starts_with(cwd) {
        return Err(format!(
            "Path outside CWD not allowed: {}",
            resolved_canonical.display()
        ));
    }

    Ok(resolved_canonical)
}

const WRITE_FILE_DESC: &str = r#"Performs exact string replacements in a file.
Replaces old_string with new_string. Fails if old_string is not found or found multiple times.
Use when editing or modifying files."#;

pub struct WriteFile;

#[async_trait]
impl super::Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }

    fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": WRITE_FILE_DESC,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path to the file to edit, relative to current working directory"},
                        "old_string": {"type": "string", "description": "The exact text to find and replace"},
                        "new_string": {"type": "string", "description": "The replacement text"},
                    },
                    "required": ["filepath", "old_string", "new_string"],
                }
            }
        })
    }

    fn anthropic_schema(&self) -> Value {
        json!({
            "name": "write_file",
            "description": WRITE_FILE_DESC,
            "input_schema": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to edit, relative to current working directory"},
                    "old_string": {"type": "string", "description": "The exact text to find and replace"},
                    "new_string": {"type": "string", "description": "The replacement text"},
                },
                "required": ["filepath", "old_string", "new_string"],
            }
        })
    }

    async fn execute(&self, args: Value) -> Value {
        let filepath = match args.get("filepath").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return json!({"error": "filepath is required"}),
        };
        let old_string = args
            .get("old_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let new_string = args
            .get("new_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let cwd = get_cwd();
        let validated_path = match resolve_path(filepath, &cwd) {
            Ok(p) => p,
            Err(e) => return json!({"error": e}),
        };

        let file_exists = validated_path.exists();

        if !file_exists {
            if old_string.is_empty() {
                if let Some(parent) = validated_path.parent() {
                    if !parent.exists() {
                        return json!({"error": format!("parent directory does not exist: {}", parent.display())});
                    }
                }
                match std::fs::write(&validated_path, new_string) {
                    Ok(_) => {
                        return json!({
                            "filepath": filepath,
                            "new_length": new_string.len(),
                            "created": true,
                            "success": true,
                        })
                    }
                    Err(e) => return json!({"error": format!("Failed to write file: {}", e)}),
                }
            } else {
                return json!({"error": format!("File not found: {}", filepath)});
            }
        }

        let content = match std::fs::read_to_string(&validated_path) {
            Ok(c) => c,
            Err(e) => return json!({"error": format!("Failed to read file: {}", e)}),
        };

        let count = content.matches(old_string).count();

        if count == 0 {
            return json!({"error": format!(
                "old_string not found in file. File: {}, old_string length: {}",
                filepath, old_string.len()
            )});
        }

        if count > 1 {
            return json!({"error": format!(
                "old_string found {} times in file — be more specific by including more surrounding context",
                count
            )});
        }

        let new_content = content.replacen(old_string, new_string, 1);

        if let Err(e) = std::fs::write(&validated_path, &new_content) {
            return json!({"error": format!("Failed to write file: {}", e)});
        }

        json!({
            "filepath": filepath,
            "old_length": old_string.len(),
            "new_length": new_string.len(),
            "success": true,
        })
    }
}
