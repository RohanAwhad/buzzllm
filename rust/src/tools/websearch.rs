use async_trait::async_trait;
use regex::Regex;
use scraper::{Html, Selector};
use serde_json::{json, Value};

#[derive(Debug)]
struct DuckDuckGoBotDetected;

impl std::fmt::Display for DuckDuckGoBotDetected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DuckDuckGo bot detection triggered")
    }
}

impl std::error::Error for DuckDuckGoBotDetected {}

async fn search_duckduckgo(
    query: &str,
    count: usize,
) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>> {
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let client = reqwest::Client::new();
    let response = client
        .get("https://lite.duckduckgo.com/lite/")
        .query(&[("q", query)])
        .header(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        )
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        )
        .header("Accept-Language", "en-US,en;q=0.5")
        .header("Accept-Encoding", "gzip, deflate")
        .header("Connection", "keep-alive")
        .header("Upgrade-Insecure-Requests", "1")
        .header("Sec-Fetch-Dest", "document")
        .header("Sec-Fetch-Mode", "navigate")
        .header("Sec-Fetch-Site", "none")
        .header("Cache-Control", "max-age=0")
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await?;

    if response.status().as_u16() == 202 {
        tracing::warn!("DuckDuckGo bot detection triggered");
        return Err(Box::new(DuckDuckGoBotDetected));
    }

    let text = response.text().await?;

    if text.contains("anomaly.js") {
        tracing::warn!("DuckDuckGo bot detection triggered");
        return Err(Box::new(DuckDuckGoBotDetected));
    }

    let document = Html::parse_document(&text);
    let table_sel = Selector::parse("table").unwrap();
    let tr_sel = Selector::parse("tr").unwrap();
    let td_sel = Selector::parse("td").unwrap();
    let a_sel = Selector::parse("a[href]").unwrap();

    let tables: Vec<_> = document.select(&table_sel).collect();
    let results_table = match tables.last() {
        Some(t) => t,
        None => return Ok(Vec::new()),
    };

    let rows: Vec<_> = results_table.select(&tr_sel).collect();
    let num_pattern = Regex::new(r"^\d+\.\s*$").unwrap();
    let ws_re = Regex::new(r"\s+").unwrap();

    let mut results = Vec::new();
    let mut i = 0;

    while i < rows.len() && results.len() < count {
        let cells: Vec<_> = rows[i].select(&td_sel).collect();

        if cells.len() >= 2 {
            let first_text = cells[0].text().collect::<String>().trim().to_string();

            if num_pattern.is_match(&first_text) {
                if let Some(link) = cells[1].select(&a_sel).next() {
                    let title = link.text().collect::<String>().trim().to_string();
                    let url = link.value().attr("href").unwrap_or("").to_string();

                    let mut description = String::new();
                    if i + 1 < rows.len() {
                        let desc_cells: Vec<_> = rows[i + 1].select(&td_sel).collect();
                        if desc_cells.len() >= 2 {
                            let desc_text =
                                desc_cells[1].text().collect::<String>().trim().to_string();
                            if !desc_text.starts_with("http://")
                                && !desc_text.starts_with("https://")
                            {
                                description = desc_text;
                            }
                        }
                    }

                    // Collapse whitespace
                    description = ws_re.replace_all(&description, " ").trim().to_string();

                    if !title.is_empty() && !url.is_empty() {
                        results.push(json!({
                            "title": title,
                            "url": url,
                            "description": description,
                        }));
                    }

                    i += 3;
                    continue;
                }
            }
        }
        i += 1;
    }

    Ok(results)
}

async fn search_brave(
    query: &str,
    count: usize,
) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>> {
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let api_key = std::env::var("BRAVE_SEARCH_AI_API_KEY").map_err(
        |_| -> Box<dyn std::error::Error + Send + Sync> {
            tracing::error!("Missing Brave Search API key");
            "Missing Brave Search API key".into()
        },
    )?;

    if api_key.is_empty() {
        tracing::error!("Missing Brave Search API key");
        return Err("Missing Brave Search API key".into());
    }

    let client = reqwest::Client::new();
    let response = client
        .get("https://api.search.brave.com/res/v1/web/search")
        .query(&[("q", query), ("count", &count.to_string())])
        .header("Accept", "application/json")
        .header("X-Subscription-Token", &api_key)
        .send()
        .await?;

    response.error_for_status_ref()?;
    let results_json: Value = response.json().await?;

    let mut results = Vec::new();
    if let Some(items) = results_json
        .get("web")
        .and_then(|w| w.get("results"))
        .and_then(|r| r.as_array())
    {
        for item in items {
            results.push(json!({
                "title": item.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                "url": item.get("url").and_then(|v| v.as_str()).unwrap_or(""),
                "description": item.get("description").and_then(|v| v.as_str()).unwrap_or(""),
            }));
        }
    }

    Ok(results)
}

async fn search_web_internal(query: &str) -> Vec<Value> {
    // Try DDG with retry (up to 3 attempts)
    for attempt in 0..3u32 {
        match search_duckduckgo(query, 10).await {
            Ok(results) => return results,
            Err(e) => {
                if e.downcast_ref::<DuckDuckGoBotDetected>().is_some() {
                    tracing::warn!(
                        "DuckDuckGo bot detection triggered, falling back to Brave Search"
                    );
                    break;
                }
                if attempt < 2 {
                    let delay = std::time::Duration::from_secs(2u64.pow(attempt + 1));
                    tokio::time::sleep(delay).await;
                    continue;
                }
                tracing::warn!("DuckDuckGo search failed, falling back to Brave Search");
                break;
            }
        }
    }

    // Brave fallback with retry (up to 3 attempts)
    for attempt in 0..3u32 {
        match search_brave(query, 10).await {
            Ok(results) => return results,
            Err(_) => {
                if attempt < 2 {
                    let delay = std::time::Duration::from_secs(2u64.pow(attempt + 1));
                    tokio::time::sleep(delay).await;
                    continue;
                }
                tracing::error!("Both DuckDuckGo and Brave Search failed");
            }
        }
    }

    Vec::new()
}

async fn fetch_markdown(url: &str) -> String {
    // Use reqwest to fetch the page, then convert HTML to basic markdown
    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .timeout(std::time::Duration::from_secs(30))
        .build();

    let client = match client {
        Ok(c) => c,
        Err(e) => return format!("Error scraping {}: {}", url, e),
    };

    let response = match client.get(url).send().await {
        Ok(r) => r,
        Err(e) => return format!("Error scraping {}: {}", url, e),
    };

    let html = match response.text().await {
        Ok(t) => t,
        Err(e) => return format!("Error scraping {}: {}", url, e),
    };

    html_to_markdown(&html)
}

fn html_to_markdown(html: &str) -> String {
    let document = Html::parse_document(html);
    let mut output = String::new();

    // Remove nav, header, footer by collecting text from body excluding those
    let body_sel = Selector::parse("body").unwrap();
    let body = match document.select(&body_sel).next() {
        Some(b) => b,
        None => return document.root_element().text().collect::<String>(),
    };

    // Simple extraction: iterate elements and convert
    extract_text_recursive(&body, &mut output, 0);

    // Collapse multiple blank lines
    let re = Regex::new(r"\n{3,}").unwrap();
    re.replace_all(&output, "\n\n").trim().to_string()
}

fn extract_text_recursive(element: &scraper::ElementRef, output: &mut String, depth: usize) {
    let tag = element.value().name();

    // Skip nav, header, footer, script, style
    if matches!(
        tag,
        "nav" | "header" | "footer" | "script" | "style" | "noscript"
    ) {
        return;
    }

    match tag {
        "h1" => {
            output.push_str("\n# ");
            let text: String = element.text().collect();
            output.push_str(text.trim());
            output.push_str("\n\n");
            return;
        }
        "h2" => {
            output.push_str("\n## ");
            let text: String = element.text().collect();
            output.push_str(text.trim());
            output.push_str("\n\n");
            return;
        }
        "h3" => {
            output.push_str("\n### ");
            let text: String = element.text().collect();
            output.push_str(text.trim());
            output.push_str("\n\n");
            return;
        }
        "h4" | "h5" | "h6" => {
            output.push_str("\n#### ");
            let text: String = element.text().collect();
            output.push_str(text.trim());
            output.push_str("\n\n");
            return;
        }
        "p" => {
            let text: String = element.text().collect();
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                output.push_str(trimmed);
                output.push_str("\n\n");
            }
            return;
        }
        "pre" | "code" => {
            output.push_str("\n```\n");
            let text: String = element.text().collect();
            output.push_str(&text);
            output.push_str("\n```\n\n");
            return;
        }
        "li" => {
            output.push_str("- ");
            let text: String = element.text().collect();
            output.push_str(text.trim());
            output.push('\n');
            return;
        }
        "br" => {
            output.push('\n');
            return;
        }
        _ => {}
    }

    // Recurse into children
    for child in element.children() {
        if let Some(child_element) = scraper::ElementRef::wrap(child) {
            extract_text_recursive(&child_element, output, depth + 1);
        } else if let Some(text) = child.value().as_text() {
            let t = text.trim();
            if !t.is_empty() && depth > 0 {
                output.push_str(t);
                output.push(' ');
            }
        }
    }
}

// --- SearchWeb Tool ---

pub struct SearchWeb;

#[async_trait]
impl super::Tool for SearchWeb {
    fn name(&self) -> &str {
        "search_web"
    }

    fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information using DuckDuckGo with Brave fallback",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query string"},
                    },
                    "required": ["query"],
                }
            }
        })
    }

    fn anthropic_schema(&self) -> Value {
        json!({
            "name": "search_web",
            "description": "Search the web for information using DuckDuckGo with Brave fallback",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string"},
                },
                "required": ["query"],
            }
        })
    }

    async fn execute(&self, args: Value) -> Value {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let results = search_web_internal(query).await;
        json!(results)
    }
}

// --- ScrapeWebpage Tool ---

pub struct ScrapeWebpage;

#[async_trait]
impl super::Tool for ScrapeWebpage {
    fn name(&self) -> &str {
        "scrape_webpage"
    }

    fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "scrape_webpage",
                "description": "Scrape and extract content from a webpage as markdown",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL of the webpage to scrape"},
                    },
                    "required": ["url"],
                }
            }
        })
    }

    fn anthropic_schema(&self) -> Value {
        json!({
            "name": "scrape_webpage",
            "description": "Scrape and extract content from a webpage as markdown",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the webpage to scrape"},
                },
                "required": ["url"],
            }
        })
    }

    async fn execute(&self, args: Value) -> Value {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
        if url.is_empty() {
            return json!("Error: url is required");
        }

        // Retry up to 5 attempts with exponential backoff
        for attempt in 0..5u32 {
            let result = fetch_markdown(url).await;
            if !result.starts_with("Error scraping") || attempt >= 4 {
                return json!(result);
            }
            let delay =
                std::time::Duration::from_secs((4u64).min(2u64.pow(attempt + 1)).clamp(4, 10));
            tokio::time::sleep(delay).await;
        }

        json!(format!("Error scraping {}: all retries exhausted", url))
    }
}
