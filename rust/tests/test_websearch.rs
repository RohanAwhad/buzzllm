use buzzllm::tools::websearch::{html_to_markdown, ScrapeWebpage, SearchWeb};
use buzzllm::tools::Tool;
use serde_json::json;

// ============================================================
// SearchWeb — Schema Tests
// ============================================================

#[test]
fn test_search_web_name() {
    assert_eq!(SearchWeb.name(), "search_web");
}

#[test]
fn test_search_web_openai_schema() {
    let schema = SearchWeb.openai_schema();
    assert_eq!(schema["type"], "function");
    assert_eq!(schema["function"]["name"], "search_web");
    let params = &schema["function"]["parameters"];
    assert!(params["properties"]["query"]["type"] == "string");
    assert!(params["required"].as_array().unwrap().contains(&json!("query")));
}

#[test]
fn test_search_web_anthropic_schema() {
    let schema = SearchWeb.anthropic_schema();
    assert_eq!(schema["name"], "search_web");
    let params = &schema["input_schema"];
    assert!(params["properties"]["query"]["type"] == "string");
}

// ============================================================
// ScrapeWebpage — Schema Tests
// ============================================================

#[test]
fn test_scrape_webpage_name() {
    assert_eq!(ScrapeWebpage.name(), "scrape_webpage");
}

#[test]
fn test_scrape_webpage_openai_schema() {
    let schema = ScrapeWebpage.openai_schema();
    assert_eq!(schema["type"], "function");
    assert_eq!(schema["function"]["name"], "scrape_webpage");
    let params = &schema["function"]["parameters"];
    assert!(params["properties"]["url"]["type"] == "string");
    assert!(params["required"].as_array().unwrap().contains(&json!("url")));
}

#[test]
fn test_scrape_webpage_anthropic_schema() {
    let schema = ScrapeWebpage.anthropic_schema();
    assert_eq!(schema["name"], "scrape_webpage");
    let params = &schema["input_schema"];
    assert!(params["properties"]["url"]["type"] == "string");
}

// ============================================================
// html_to_markdown — Parsing Tests
// ============================================================

#[test]
fn test_html_to_markdown_headings() {
    let html = "<h1>Title</h1><h2>Sub</h2><h3>Subsub</h3>";
    let md = html_to_markdown(html);
    assert!(md.contains("# Title"));
    assert!(md.contains("## Sub"));
    assert!(md.contains("### Subsub"));
}

#[test]
fn test_html_to_markdown_paragraphs() {
    let html = "<p>First paragraph</p><p>Second paragraph.</p>";
    let md = html_to_markdown(html);
    assert!(md.contains("First paragraph"));
    assert!(md.contains("Second paragraph."));
}

#[test]
fn test_html_to_markdown_lists() {
    let html = "<ul><li>Item 1</li><li>Item 2</li></ul>";
    let md = html_to_markdown(html);
    assert!(md.contains("- Item 1"));
    assert!(md.contains("- Item 2"));
}

#[test]
fn test_html_to_markdown_code_block() {
    let html = "<pre><code>fn main() {}</code></pre>";
    let md = html_to_markdown(html);
    assert!(md.contains("```"));
    assert!(md.contains("fn main() {}"));
}

#[test]
fn test_html_to_markdown_skips_nav() {
    let html = "<body><nav>menu</nav><p>Content</p></body>";
    let md = html_to_markdown(html);
    assert!(!md.contains("menu"));
    assert!(md.contains("Content"));
}

#[test]
fn test_html_to_markdown_skips_script() {
    let html = "<body><script>alert(1)</script><p>Text</p></body>";
    let md = html_to_markdown(html);
    assert!(!md.contains("alert"));
    assert!(md.contains("Text"));
}

#[test]
fn test_html_to_markdown_empty() {
    let md = html_to_markdown("");
    assert_eq!(md, "");
}

#[test]
fn test_html_to_markdown_text_in_paragraph() {
    let md = html_to_markdown("<body><p>Hello World</p></body>");
    assert!(md.contains("Hello World"));
}

// ============================================================
// SearchWeb.execute() — Edge Cases (network dependent, skipped in CI)
// ============================================================

#[tokio::test]
async fn test_search_web_missing_query_returns_array() {
    let result = SearchWeb.execute(json!({})).await;
    assert!(result.as_array().is_some());
}
