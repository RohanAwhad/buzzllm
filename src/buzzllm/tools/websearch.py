import os
import httpx
import subprocess
import sys
from functools import lru_cache

from duckduckgo_search import DDGS
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError


@lru_cache(maxsize=1)
def _ensure_chromium():
    """Ensure Chromium is installed for Playwright"""
    # Check if already installed
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
        capture_output=True,
        text=True,
    )

    if "is already installed" not in result.stdout:
        print("Installing Chromium for Playwright...")
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"], check=True
        )
        print("Chromium installed successfully.")


class WebSearchResults(BaseModel):
    title: str
    url: str
    description: str | list[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def _search_duckduckgo(query: str, count: int = 10) -> list[dict]:
    """Search using DuckDuckGo"""
    if not query:
        return []

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=count))

    return [
        WebSearchResults(
            title=result["title"], url=result["href"], description=result["body"]
        ).model_dump()
        for result in results
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def _search_brave(query: str, count: int = 10) -> list[dict]:
    """Search using Brave Search API"""
    if not query:
        return []

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": os.environ.get("BRAVE_SEARCH_AI_API_KEY", ""),
    }

    if not headers["X-Subscription-Token"]:
        logger.error("Missing Brave Search API key")
        raise Exception("Missing Brave Search API key")

    params = {"q": query, "count": count}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        results_json = response.json()

    results = []
    for item in results_json.get("web", {}).get("results", []):
        result = WebSearchResults(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            description=str(item.get("description", "")),
        ).model_dump()
        results.append(result)

    return results


async def _search_web_async(query: str) -> list[dict]:
    """Internal async search function with fallback"""
    try:
        return await _search_duckduckgo(query)
    except RetryError:
        logger.warning("DuckDuckGo search failed, falling back to Brave Search")
        try:
            return await _search_brave(query)
        except RetryError:
            logger.error("Both DuckDuckGo and Brave Search failed")
            return []


async def search_web(query: str) -> list[dict]:
    """Search the web for information using DuckDuckGo with Brave fallback

    :param query: The search query string to find relevant web pages
    :returns: List of json objects containing search results with title, URL, and description.
    """
    return await _search_web_async(query)


async def scrape_webpage(url: str) -> str:
    """Scrape and extract content from a webpage as markdown

    :param url: The URL of the webpage to scrape
    :returns: Markdown content of the webpage, or error message if scraping fails
    """
    _ensure_chromium()
    return await _fetch_markdown(url)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def _fetch_markdown(url: str) -> str:
    try:
        assert url is not None, "url is None"
        assert len(url) > 0, "len(url) is 0"

        md_generator = DefaultMarkdownGenerator(
            options={"ignore_links": True, "escape_html": True, "body_width": 80},
            content_filter=PruningContentFilter(threshold=0.5, min_word_threshold=50),
        )

        browser_config = BrowserConfig(verbose=False)
        crawler_config = CrawlerRunConfig(
            markdown_generator=md_generator,
            word_count_threshold=10,
            excluded_tags=["nav", "header", "footer"],
            exclude_external_links=True,
            verbose=False,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url, config=crawler_config)
            return str(result.markdown)
    except Exception as e:
        logger.exception(f"Failed to fetch markdown for url: {url}")
        return f"Error scraping {url}: {str(e)}"
