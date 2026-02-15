import pytest
import responses
import respx
import httpx
from unittest.mock import patch, MagicMock, AsyncMock


class TestSearchDuckDuckGo:
    @responses.activate
    @pytest.mark.asyncio
    async def test_parses_html_results(self, mock_duckduckgo_html):
        from buzzllm.tools.websearch import _search_duckduckgo

        responses.add(
            responses.GET,
            "https://lite.duckduckgo.com/lite/",
            body=mock_duckduckgo_html,
            status=200,
        )

        results = await _search_duckduckgo("test query")

        assert len(results) >= 1
        assert results[0]["title"] == "Example Title"
        assert results[0]["url"] == "https://example.com"

    @responses.activate
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        from buzzllm.tools.websearch import _search_duckduckgo

        results = await _search_duckduckgo("")
        assert results == []

    @responses.activate
    @pytest.mark.asyncio
    async def test_bot_detection_raises(self):
        from buzzllm.tools.websearch import (
            _search_duckduckgo,
            DuckDuckGoBotDetectedError,
        )

        responses.add(
            responses.GET,
            "https://lite.duckduckgo.com/lite/",
            body="<script>anomaly.js</script>",
            status=200,
        )

        with pytest.raises(DuckDuckGoBotDetectedError):
            await _search_duckduckgo("test")

    @responses.activate
    @pytest.mark.asyncio
    async def test_202_status_raises(self):
        from buzzllm.tools.websearch import (
            _search_duckduckgo,
            DuckDuckGoBotDetectedError,
        )

        responses.add(
            responses.GET,
            "https://lite.duckduckgo.com/lite/",
            body="",
            status=202,
        )

        with pytest.raises(DuckDuckGoBotDetectedError):
            await _search_duckduckgo("test")


class TestSearchBrave:
    @respx.mock
    @pytest.mark.asyncio
    async def test_parses_json_results(self, mock_brave_json, env_with_api_keys):
        from buzzllm.tools.websearch import _search_brave

        respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(200, json=mock_brave_json)
        )

        results = await _search_brave("test query")

        assert len(results) == 1
        assert results[0]["title"] == "Example Title"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["description"] == "Example description"

    @respx.mock
    @pytest.mark.asyncio
    async def test_retries_on_http_error(self, mock_brave_json, env_with_api_keys):
        from buzzllm.tools.websearch import _search_brave

        call_count = {"count": 0}

        def handler(request):
            call_count["count"] += 1
            if call_count["count"] == 1:
                return httpx.Response(500)
            return httpx.Response(200, json=mock_brave_json)

        respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            side_effect=handler
        )

        results = await _search_brave("test query")

        assert len(results) == 1
        assert call_count["count"] == 2

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, env_with_api_keys):
        from buzzllm.tools.websearch import _search_brave

        results = await _search_brave("")
        assert results == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        from buzzllm.tools.websearch import _search_brave
        from tenacity import RetryError

        monkeypatch.delenv("BRAVE_SEARCH_AI_API_KEY", raising=False)

        with pytest.raises(RetryError):
            await _search_brave("test")


class TestSearchWeb:
    @responses.activate
    @pytest.mark.asyncio
    async def test_uses_duckduckgo_by_default(self, mock_duckduckgo_html):
        from buzzllm.tools.websearch import search_web

        responses.add(
            responses.GET,
            "https://lite.duckduckgo.com/lite/",
            body=mock_duckduckgo_html,
            status=200,
        )

        results = await search_web("test")
        assert len(results) >= 1

    @responses.activate
    @respx.mock
    @pytest.mark.asyncio
    async def test_falls_back_to_brave_on_duckduckgo_failure(
        self, mock_brave_json, env_with_api_keys
    ):
        from buzzllm.tools.websearch import search_web

        # DuckDuckGo fails all retries
        for _ in range(5):
            responses.add(
                responses.GET,
                "https://lite.duckduckgo.com/lite/",
                body="error",
                status=500,
            )

        # Brave succeeds
        respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(200, json=mock_brave_json)
        )

        results = await search_web("test")
        assert len(results) >= 1

    @responses.activate
    @respx.mock
    @pytest.mark.asyncio
    async def test_falls_back_to_brave_on_bot_detection(
        self, mock_brave_json, env_with_api_keys
    ):
        from buzzllm.tools.websearch import search_web

        responses.add(
            responses.GET,
            "https://lite.duckduckgo.com/lite/",
            body="<script>anomaly.js</script>",
            status=200,
        )

        respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(200, json=mock_brave_json)
        )

        results = await search_web("test")
        assert len(results) == 1


class TestScrapeWebpage:
    @pytest.mark.asyncio
    async def test_returns_markdown(self):
        from buzzllm.tools.websearch import scrape_webpage

        # Mock _ensure_chromium and AsyncWebCrawler
        with patch("buzzllm.tools.websearch._ensure_chromium"):
            with patch("buzzllm.tools.websearch._fetch_markdown") as mock_fetch:
                mock_fetch.return_value = "# Hello World\n\nThis is content."

                result = await scrape_webpage("https://example.com")

                assert "Hello World" in result
                mock_fetch.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_error_handling(self):
        from buzzllm.tools.websearch import scrape_webpage

        with patch("buzzllm.tools.websearch._ensure_chromium"):
            with patch("buzzllm.tools.websearch._fetch_markdown") as mock_fetch:
                mock_fetch.return_value = "Error scraping https://bad.com: timeout"

                result = await scrape_webpage("https://bad.com")

                assert "Error" in result


class TestEnsureChromium:
    def test_checks_installation(self):
        from buzzllm.tools.websearch import _ensure_chromium

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="is already installed")

            # Clear the lru_cache
            _ensure_chromium.cache_clear()
            _ensure_chromium()

            # Should check dry-run
            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert "playwright" in call_args
            assert "--dry-run" in call_args
