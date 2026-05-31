use buzzllm::tools::bash::Bash;
use buzzllm::tools::codesearch::{BashFind, BashRead, BashRipgrep};
use buzzllm::tools::write_file::WriteFile;
use buzzllm::tools::{Tool, ToolRegistry};
use serde_json::json;
use std::fs;
use tempfile::TempDir;

// ============================================================
// ToolRegistry
// ============================================================

#[test]
fn test_register_tool() {
    let mut reg = ToolRegistry::new();
    reg.register(Box::new(WriteFile));
    assert!(reg.get("write_file").is_some());
}

#[test]
fn test_register_overwrites() {
    let mut reg = ToolRegistry::new();
    reg.register(Box::new(WriteFile));
    reg.register(Box::new(WriteFile));
    assert!(reg.get("write_file").is_some());
}

#[test]
fn test_unknown_tool_returns_none() {
    let reg = ToolRegistry::new();
    assert!(reg.get("nonexistent").is_none());
}

#[test]
fn test_openai_schema_structure() {
    let _reg = ToolRegistry::new();
    let tool: &dyn Tool = &WriteFile;
    let schema = tool.openai_schema();
    assert_eq!(schema["type"], "function");
    assert!(schema["function"]["name"]
        .as_str()
        .unwrap()
        .contains("write_file"));
}

#[test]
fn test_anthropic_schema_structure() {
    let tool: &dyn Tool = &WriteFile;
    let schema = tool.anthropic_schema();
    assert!(schema["name"].as_str().unwrap().contains("write_file"));
    assert!(schema.get("input_schema").is_some());
}

// ============================================================
// WriteFile
// ============================================================

#[test]
fn test_write_file_name() {
    assert_eq!(WriteFile.name(), "write_file");
}

#[tokio::test]
async fn test_write_file_new_file() {
    let dir = TempDir::new().unwrap();
    let filepath = dir.path().join("new.txt");
    let _guard = std::env::set_current_dir(dir.path());

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": "new.txt",
            "old_string": "",
            "new_string": "hello world"
        }))
        .await;

    assert_eq!(result["success"], true);
    assert_eq!(result["created"], true);
    assert!(filepath.exists());
    assert_eq!(fs::read_to_string(&filepath).unwrap(), "hello world");
}

#[tokio::test]
async fn test_write_file_edit_existing() {
    let dir = TempDir::new().unwrap();
    let filepath = dir.path().join("edit.txt");
    fs::write(&filepath, "hello world").unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": "edit.txt",
            "old_string": "hello",
            "new_string": "goodbye"
        }))
        .await;

    assert_eq!(result["success"], true);
    assert_eq!(fs::read_to_string(&filepath).unwrap(), "goodbye world");
}

#[tokio::test]
async fn test_write_file_not_found() {
    let dir = TempDir::new().unwrap();
    let filepath = dir.path().join("edit.txt");
    fs::write(&filepath, "hello world").unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": "edit.txt",
            "old_string": "nonexistent",
            "new_string": "x"
        }))
        .await;

    assert!(result["error"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_write_file_multiple_matches() {
    let dir = TempDir::new().unwrap();
    let filepath = dir.path().join("edit.txt");
    fs::write(&filepath, "hello hello world").unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": "edit.txt",
            "old_string": "hello",
            "new_string": "x"
        }))
        .await;

    assert!(result["error"].as_str().unwrap().contains("2 times"));
}

#[tokio::test]
async fn test_write_file_missing_filepath() {
    let tool = WriteFile;
    let result = tool.execute(json!({})).await;
    assert!(result["error"].as_str().unwrap().contains("filepath"));
}

#[tokio::test]
async fn test_write_file_parent_not_exists() {
    let dir = TempDir::new().unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": "subdir/new.txt",
            "old_string": "",
            "new_string": "hello"
        }))
        .await;

    assert!(result["error"].as_str().unwrap().contains("parent"));
}

#[tokio::test]
async fn test_write_file_missing_file_with_old_string() {
    let dir = TempDir::new().unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": "nonexistent.txt",
            "old_string": "hello",
            "new_string": "x"
        }))
        .await;

    assert!(result["error"].as_str().unwrap().contains("File not found"));
}

// ============================================================
// Bash
// ============================================================

#[test]
fn test_bash_name() {
    assert_eq!(Bash.name(), "bash");
}

#[tokio::test]
async fn test_bash_simple_command() {
    let tool = Bash;
    let result = tool.execute(json!({"cmd": "echo hello"})).await;

    assert_eq!(result["exit_code"], 0);
    assert!(result["stdout"].as_str().unwrap().contains("hello"));
    assert!(!result["timed_out"].as_bool().unwrap());
}

#[tokio::test]
async fn test_bash_empty_cmd() {
    let tool = Bash;
    let result = tool.execute(json!({"cmd": "   "})).await;
    assert!(result["error"].as_str().unwrap().contains("empty"));
}

#[tokio::test]
async fn test_bash_missing_cmd() {
    let tool = Bash;
    let result = tool.execute(json!({})).await;
    assert!(result["error"].as_str().unwrap().contains("cmd"));
}

#[tokio::test]
async fn test_bash_failing_command() {
    let tool = Bash;
    let result = tool.execute(json!({"cmd": "exit 42"})).await;

    assert_eq!(result["exit_code"], 42);
}

#[tokio::test]
async fn test_bash_short_timeout() {
    let tool = Bash;
    let result = tool.execute(json!({"cmd": "sleep 10", "timeout": 1})).await;

    assert!(result["timed_out"].as_bool().unwrap());
    assert_eq!(result["exit_code"], -1);
}

#[tokio::test]
async fn test_bash_default_timeout_30() {
    let tool = Bash;
    let result = tool.execute(json!({"cmd": "echo done"})).await;
    assert_eq!(result["exit_code"], 0);
}

// ============================================================
// Codesearch — BashRead
// ============================================================

#[test]
fn test_bash_read_name() {
    assert_eq!(BashRead.name(), "bash_read");
}

#[tokio::test]
async fn test_bash_read_content() {
    let dir = TempDir::new().unwrap();
    let filepath = dir.path().join("readme.txt");
    fs::write(&filepath, "line1\nline2\nline3").unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let result = BashRead.execute(json!({"filepath": "readme.txt"})).await;
    assert!(result["content"].as_str().unwrap().contains("line1"));
    assert_eq!(result["total"], 3);
}

#[tokio::test]
async fn test_bash_read_pagination() {
    let dir = TempDir::new().unwrap();
    let filepath = dir.path().join("readme.txt");
    fs::write(&filepath, "a\nb\nc\nd\ne").unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let result = BashRead
        .execute(json!({"filepath": "readme.txt", "limit": 2, "offset": 1}))
        .await;
    assert_eq!(result["returned"], 2);
    assert_eq!(result["total"], 5);
    assert_eq!(result["results"][0], "b");
    assert_eq!(result["results"][1], "c");
    assert!(result["has_more"].as_bool().unwrap());
}

#[tokio::test]
async fn test_bash_read_not_found() {
    let dir = TempDir::new().unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let result = BashRead
        .execute(json!({"filepath": "nonexistent.txt"}))
        .await;
    assert!(result["error"].as_str().unwrap().contains("invalid path"));
}

// ============================================================
// Codesearch — BashFind
// ============================================================

#[tokio::test]
async fn test_bash_find_lists_files() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.rs"), "content").unwrap();
    fs::write(dir.path().join("b.py"), "content").unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let result = BashFind.execute(json!({"path": ".", "limit": 10})).await;
    let results = result["results"].as_array().unwrap();
    let files: Vec<&str> = results.iter().filter_map(|v| v.as_str()).collect();
    assert!(files.iter().any(|f| f.contains("a.rs")));
    assert!(files.iter().any(|f| f.contains("b.py")));
}

#[tokio::test]
async fn test_bash_find_glob_filter() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.rs"), "").unwrap();
    fs::write(dir.path().join("b.py"), "").unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let result = BashFind
        .execute(json!({"path": ".", "name": "*.rs", "limit": 10}))
        .await;
    let results = result["results"].as_array().unwrap();
    for r in results {
        assert!(r.as_str().unwrap().ends_with(".rs"));
    }
}

// ============================================================
// Codesearch — BashRipgrep
// ============================================================

#[tokio::test]
async fn test_bash_ripgrep_finds_pattern() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("test.txt"), "hello world\nfoo bar").unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let result = BashRipgrep
        .execute(json!({"pattern": "hello", "files": "."}))
        .await;
    let results = result["results"].as_array().unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_bash_ripgrep_no_match() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("test.txt"), "hello world").unwrap();
    let _guard = std::env::set_current_dir(dir.path());

    let result = BashRipgrep
        .execute(json!({"pattern": "ZXYZZYNOTFOUND", "files": "."}))
        .await;
    assert!(result.get("error").is_some());
}
