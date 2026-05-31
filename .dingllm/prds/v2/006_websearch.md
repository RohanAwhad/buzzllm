# Phase 6: Websearch Tools

## Goal

Port the two websearch tools (`search_web`, `scrape_webpage`) to Rust. `search_web` uses DuckDuckGo HTML scraping with Brave API fallback. `scrape_webpage` uses headless Chrome (chromiumoxide) to render pages and extract markdown.

## Source reference

- `src/buzzllm/tools/websearch.py:42-49` — `WebSearchResults` model, `DuckDuckGoBotDetectedError`
- `src/buzzllm/tools/websearch.py:52-135` — `_search_duckduckgo()`
- `src/buzzllm/tools/websearch.py:138-170` — `_search_brave()`
- `src/buzzllm/tools/websearch.py:173-197` — `search_web()` (fallback logic)
- `src/buzzllm/tools/websearch.py:200-235` — `scrape_webpage()` + `_fetch_markdown()`

## Tools to implement

### Tool 1: `SearchWeb`

**Name**: `search_web`

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | (required) | Search query string |

**Returns**: JSON array of `{title, url, description}` objects.

**Behavior — DuckDuckGo primary**:
1. GET `https://lite.duckduckgo.com/lite/` with `q={query}`
2. Headers: mimic Firefox browser (User-Agent, Accept, etc. — copy from Python source)
3. Bot detection: if status 202 or response contains `"anomaly.js"` → throw `DuckDuckGoBotDetected` error (do NOT retry DDG, fall through to Brave)
4. Parse HTML with `scraper` crate:
   - Find last `<table>` in page
   - Iterate rows: look for cells matching `^\d+\.\s*$` pattern
   - Extract title + URL from link in second cell
   - Description from next row's second cell (skip if it looks like a URL)
   - Collect up to 10 results
5. Retry: up to 3 attempts, exponential backoff (2s, 4s, 8s) — but NOT on bot detection

**Behavior — Brave fallback**:
1. Triggered when DDG fails (bot detection or retry exhaustion)
2. GET `https://api.search.brave.com/res/v1/web/search` with `q={query}&count=10`
3. Header: `X-Subscription-Token` from `BRAVE_SEARCH_AI_API_KEY` env var
4. Parse JSON response: `response.web.results[]` → extract `title`, `url`, `description`
5. Retry: up to 3 attempts, exponential backoff

**Fallback chain**: DDG → Brave → empty array

### Tool 2: `ScrapeWebpage`

**Name**: `scrape_webpage`

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | string | (required) | URL to scrape |

**Returns**: String — markdown content of the page.

**Behavior**:
1. Launch headless Chromium via `chromiumoxide`
2. Navigate to URL, wait for page load
3. Extract page content as text/HTML
4. Convert to markdown (strip nav, header, footer elements)
5. Retry: up to 5 attempts, exponential backoff (4s min, 10s max)
6. On failure: return `"Error scraping {url}: {error}"`

**chromiumoxide implementation plan**:

```rust
use chromiumoxide::browser::{Browser, BrowserConfig};

async fn fetch_markdown(url: &str) -> Result<String> {
    let (browser, mut handler) = Browser::launch(
        BrowserConfig::builder()
            .no_sandbox()
            .build()?
    ).await?;

    // Spawn browser event handler
    tokio::spawn(async move { while let Some(_) = handler.next().await {} });

    let page = browser.new_page(url).await?;
    page.wait_for_navigation().await?;

    // Get page content
    let html = page.content().await?;

    // Parse with scraper, strip nav/header/footer, convert to simple markdown
    let markdown = html_to_markdown(&html);

    browser.close().await?;
    Ok(markdown)
}
```

**HTML to markdown conversion**:
The Python version uses `crawl4ai`'s `PruningContentFilter` and `DefaultMarkdownGenerator`. For parity, implement a basic converter:
- Parse HTML with `scraper`
- Remove `<nav>`, `<header>`, `<footer>` elements
- Convert headings (`<h1>`-`<h6>`) to `#` prefixes
- Convert `<p>` to text blocks with blank line separators
- Convert `<a>` to text (ignore links, matching Python's `ignore_links: True`)
- Convert `<ul>/<ol>/<li>` to markdown lists
- Convert `<code>`/`<pre>` to fenced code blocks
- Strip remaining HTML tags, collapse whitespace

This won't be as sophisticated as crawl4ai's pruning filter, but covers the common case. Can iterate on quality later.

### Chromium dependency

`chromiumoxide` requires a Chromium binary. Behavior:
- Check if Chrome/Chromium is available in PATH or standard install locations
- If not found, print an error message directing user to install it
- The Python version auto-installs via Playwright — we skip that complexity for now and document the requirement

### Retry implementation

Use `tokio-retry` or a manual loop:

```rust
use tokio_retry::strategy::ExponentialBackoff;
use tokio_retry::Retry;

let strategy = ExponentialBackoff::from_millis(1000).factor(2).take(3);
let result = Retry::spawn(strategy, || async { _search_duckduckgo(&query).await }).await;
```

For DDG bot detection, use a custom retry condition that does NOT retry on that specific error.

## Verification

1. `search_web("rust programming")` returns non-empty results array with title/url/description
2. DDG bot detection triggers Brave fallback (may need to mock or force)
3. Brave search works with valid `BRAVE_SEARCH_AI_API_KEY`
4. Both fail gracefully → returns empty array
5. `scrape_webpage("https://example.com")` returns markdown string
6. `scrape_webpage` with invalid URL returns error string (not panic)
7. Retry logic: inject failures, verify retries happen with backoff
8. End-to-end: `cargo run -- "gpt-4o-mini" ... --system-prompt websearch` with a search question
