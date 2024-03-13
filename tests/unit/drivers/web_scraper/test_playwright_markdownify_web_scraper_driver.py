import pytest

from griptape.drivers.web_scraper.playwright_markdownify_web_scraper_driver import PlaywrightMarkdownifyWebScraperDriver


class TestPlaywrightMarkdownifyWebScraperDriver:
    @pytest.fixture(autouse=True)
    def mock_playwright(self, mocker):
        fake_response = '<html><a href="foobar.com">foobar</a></html>'

        playwright = mocker.MagicMock()
        playwright.__enter__.return_value.chromium.launch.return_value.__enter__.return_value.new_page.return_value.content.return_value = fake_response
        mocker.patch("playwright.sync_api.sync_playwright", return_value=playwright)

    @pytest.fixture
    def web_scraper(self):
        return PlaywrightMarkdownifyWebScraperDriver()

    def test_scrape_url(self, web_scraper):
        text = web_scraper.scrape_url("https://example.com/")
        assert '[foobar](foobar.com)' == text

    # TODO: Fix this test
    # def test_scrape_url_exclude_links(self):
    #     web_scraper = PlaywrightMarkdownifyWebScraperDriver(include_links=False)
    #     text = web_scraper.scrape_url("https://example.com/")
    #     assert '[foobar](foobar.com)' not in text
    #     assert "foobar" == text

    def test_scrape_url_raises_on_empty_string_from_playwright(self, web_scraper, mocker):
        playwright = mocker.MagicMock()
        playwright.__enter__.return_value.chromium.launch.return_value.__enter__.return_value.new_page.return_value.content.return_value = ""
        mocker.patch("playwright.sync_api.sync_playwright", return_value=playwright)

        with pytest.raises(Exception, match="can't access URL"):
            web_scraper.scrape_url("https://example.com/")

    def test_scrape_url_raises_on_none_from_playwright(self, web_scraper, mocker):
        playwright = mocker.MagicMock()
        playwright.__enter__.return_value.chromium.launch.return_value.__enter__.return_value.new_page.return_value.content.return_value = None
        mocker.patch("playwright.sync_api.sync_playwright", return_value=playwright)

        with pytest.raises(Exception, match="can't access URL"):
            web_scraper.scrape_url("https://example.com/")

