use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_help_exits_zero() {
    Command::cargo_bin("buzzllm")
        .unwrap()
        .arg("--help")
        .assert()
        .success();
}

#[test]
fn test_help_shows_providers() {
    Command::cargo_bin("buzzllm")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("openai-chat"))
        .stdout(predicate::str::contains("anthropic"))
        .stdout(predicate::str::contains("vertexai-anthropic"));
}

#[test]
fn test_help_shows_prompts() {
    Command::cargo_bin("buzzllm")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("coding"))
        .stdout(predicate::str::contains("websearch"))
        .stdout(predicate::str::contains("codesearch"));
}

#[test]
fn test_missing_provider_fails() {
    Command::cargo_bin("buzzllm")
        .unwrap()
        .arg("gpt-4.1-mini")
        .arg("")
        .arg("hello")
        .arg("--api-key-name")
        .arg("KEY")
        .assert()
        .failure();
}

#[test]
fn test_invalid_provider_fails() {
    Command::cargo_bin("buzzllm")
        .unwrap()
        .arg("gpt-4.1-mini")
        .arg("")
        .arg("hello")
        .arg("--provider")
        .arg("invalid")
        .arg("--api-key-name")
        .arg("KEY")
        .assert()
        .failure();
}

#[test]
fn test_system_prompt_coding_accepted() {
    Command::cargo_bin("buzzllm")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Available prompts:"))
        .stdout(predicate::str::contains("coding"));
}

#[test]
fn test_openai_simple_call() {
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping: OPENAI_API_KEY not set");
        return;
    }
    Command::cargo_bin("buzzllm")
        .unwrap()
        .arg("gpt-4.1-mini")
        .arg("")
        .arg("Say hello")
        .arg("--provider")
        .arg("openai-chat")
        .arg("--api-key-name")
        .arg("OPENAI_API_KEY")
        .arg("--max-tokens")
        .arg("50")
        .assert()
        .success()
        .stdout(predicate::str::contains("=== [ DONE ] ==="));
}

#[test]
fn test_sse_output_format() {
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping: OPENAI_API_KEY not set");
        return;
    }
    Command::cargo_bin("buzzllm")
        .unwrap()
        .arg("gpt-4.1-mini")
        .arg("")
        .arg("Hi")
        .arg("--provider")
        .arg("openai-chat")
        .arg("--api-key-name")
        .arg("OPENAI_API_KEY")
        .arg("-S")
        .arg("--max-tokens")
        .arg("50")
        .assert()
        .success()
        .stdout(predicate::str::contains("event:"))
        .stdout(predicate::str::contains("response_end"));
}
