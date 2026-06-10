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
    let path_str = filepath.to_string_lossy().to_string();

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": path_str,
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
    let path_str = filepath.to_string_lossy().to_string();
    fs::write(&filepath, "hello world").unwrap();

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": path_str,
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
    let path_str = filepath.to_string_lossy().to_string();
    fs::write(&filepath, "hello world").unwrap();

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": path_str,
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
    let path_str = filepath.to_string_lossy().to_string();
    fs::write(&filepath, "hello hello world").unwrap();

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": path_str,
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
    let path_str = dir.path().join("subdir/new.txt").to_string_lossy().to_string();

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": path_str,
            "old_string": "",
            "new_string": "hello"
        }))
        .await;

    assert!(result["error"].as_str().unwrap().contains("parent"));
}

#[tokio::test]
async fn test_write_file_missing_file_with_old_string() {
    let dir = TempDir::new().unwrap();
    let path_str = dir.path().join("nonexistent.txt").to_string_lossy().to_string();

    let tool = WriteFile;
    let result = tool
        .execute(json!({
            "filepath": path_str,
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
    let path_str = filepath.to_string_lossy().to_string();
    fs::write(&filepath, "line1\nline2\nline3").unwrap();

    let result = BashRead.execute(json!({"filepath": path_str})).await;
    assert!(result["content"].as_str().unwrap().contains("line1"));
    assert_eq!(result["total"], 3);
}

#[tokio::test]
async fn test_bash_read_pagination() {
    let dir = TempDir::new().unwrap();
    let filepath = dir.path().join("readme.txt");
    let path_str = filepath.to_string_lossy().to_string();
    fs::write(&filepath, "a\nb\nc\nd\ne").unwrap();

    let result = BashRead
        .execute(json!({"filepath": path_str, "limit": 2, "offset": 1}))
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
    let path = dir.path().to_string_lossy().to_string();
    fs::write(dir.path().join("a.rs"), "content").unwrap();
    fs::write(dir.path().join("b.py"), "content").unwrap();

    let result = BashFind.execute(json!({"path": path, "limit": 10})).await;
    let results = result["results"]
        .as_array()
        .unwrap_or_else(|| panic!("no results key: {:?}", result));
    let files: Vec<&str> = results.iter().filter_map(|v| v.as_str()).collect();
    assert!(files.iter().any(|f| f.contains("a.rs")));
    assert!(files.iter().any(|f| f.contains("b.py")));
}

#[tokio::test]
async fn test_bash_find_glob_filter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_string_lossy().to_string();
    fs::write(dir.path().join("a.rs"), "").unwrap();
    fs::write(dir.path().join("b.py"), "").unwrap();

    let result = BashFind
        .execute(json!({"path": path, "name": "*.rs", "limit": 10}))
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
    let path = dir.path().to_string_lossy().to_string();

    let result = BashRipgrep
        .execute(json!({"pattern": "hello", "files": path}))
        .await;
    let results = result["results"].as_array().unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_bash_ripgrep_no_match() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("test.txt"), "hello world").unwrap();
    let path = dir.path().to_string_lossy().to_string();

    let result = BashRipgrep
        .execute(json!({"pattern": "ZXYZZYNOTFOUND", "files": path}))
        .await;
    assert!(result.get("error").is_some());
}

// ============================================================
// Codesearch — Path Validation
// ============================================================

#[tokio::test]
async fn test_bash_find_invalid_path() {
    let result = BashFind
        .execute(json!({"path": "/nonexistent/path/xyzzy", "limit": 10}))
        .await;
    assert!(result.get("error").is_some());
}

#[tokio::test]
async fn test_bash_find_path_outside_cwd() {
    let result = BashFind
        .execute(json!({"path": "../outside", "limit": 10}))
        .await;
    assert!(result.get("error").is_some());
}

#[tokio::test]
async fn test_bash_ripgrep_invalid_path() {
    let result = BashRipgrep
        .execute(json!({"pattern": "hello", "files": "/nonexistent/xyzzy"}))
        .await;
    assert!(result.get("error").is_some());
}

#[tokio::test]
async fn test_bash_read_invalid_path() {
    let result = BashRead
        .execute(json!({"filepath": "/nonexistent/file.txt"}))
        .await;
    assert!(result.get("error").is_some());
}

// ============================================================
// Codesearch — Pagination
// ============================================================

#[tokio::test]
async fn test_bash_find_pagination_limit() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_string_lossy().to_string();
    fs::write(dir.path().join("a.rs"), "").unwrap();
    fs::write(dir.path().join("b.py"), "").unwrap();
    fs::write(dir.path().join("c.md"), "").unwrap();

    let result = BashFind.execute(json!({"path": path, "limit": 2})).await;
    let results = result["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_bash_find_pagination_offset() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_string_lossy().to_string();
    fs::write(dir.path().join("a.rs"), "").unwrap();
    fs::write(dir.path().join("b.py"), "").unwrap();
    fs::write(dir.path().join("c.md"), "").unwrap();

    let all = BashFind.execute(json!({"path": path, "limit": 0})).await;
    let offset = BashFind
        .execute(json!({"path": path.clone(), "limit": 0, "offset": 1}))
        .await;
    let all_results = all["results"]
        .as_array()
        .unwrap_or_else(|| panic!("all has no results key: {:?}", all));
    let offset_results = offset["results"]
        .as_array()
        .unwrap_or_else(|| panic!("offset has no results key: {:?}", offset));
    assert_eq!(offset_results.len(), all_results.len() - 1);
}

#[tokio::test]
async fn test_bash_ripgrep_pagination_limit() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_string_lossy().to_string();
    fs::write(dir.path().join("test.txt"), "line1\nline2\nline3\nline4").unwrap();

    let result = BashRipgrep
        .execute(json!({"pattern": "line", "files": path, "limit": 2}))
        .await;
    let results = result["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_bash_read_pagination_limit() {
    let dir = TempDir::new().unwrap();
    let content = "line1\nline2\nline3\nline4\nline5";
    fs::write(dir.path().join("test.txt"), content).unwrap();
    let path_str = dir.path().join("test.txt").to_string_lossy().to_string();

    let result = BashRead
        .execute(json!({"filepath": path_str, "limit": 2}))
        .await;
    if let Some(text) = result.get("content").and_then(|v| v.as_str()) {
        assert!(text.contains("line1"));
        assert!(!text.contains("line5"));
        assert_eq!(text.split('\n').count(), 2);
    } else {
        panic!("no content key: {:?}", result);
    }
}

// ============================================================
// Codesearch — Directory Filter (type_filter = "d")
// ============================================================

#[tokio::test]
async fn test_bash_find_directory_filter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_string_lossy().to_string();
    let subdir = dir.path().join("subdir");
    fs::create_dir(&subdir).unwrap();
    fs::write(dir.path().join("file.txt"), "").unwrap();

    let result = BashFind
        .execute(json!({"path": path, "type_filter": "d", "limit": 10}))
        .await;
    let results = result["results"]
        .as_array()
        .unwrap_or_else(|| panic!("no results key: {:?}", result));
    let dirs: Vec<&str> = results.iter().filter_map(|v| v.as_str()).collect();
    assert!(dirs.iter().any(|d| d.contains("subdir")));
    assert!(!dirs.iter().any(|d| d.contains("file.txt")));
}
