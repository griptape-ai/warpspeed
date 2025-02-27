---
search:
  boost: 2
---

## Overview

Web Scraper Drivers can be used to scrape text from the web. They are used by [WebLoader](../../reference/griptape/loaders/web_loader.md) to provide its functionality. All Web Scraper Drivers implement the following methods:

- `scrape_url()` scrapes text from a website and returns a [TextArtifact](../../reference/griptape/artifacts/text_artifact.md). The format of the scrapped text is determined by the Driver.

## Web Scraper Drivers

### Proxy

The [ProxyWebScraperDriver](../../reference/griptape/drivers/web_scraper/proxy_web_scraper_driver.md) uses the `requests` library with a provided set of proxies to do web scraping. Paid webscraping services like [ZenRows](https://www.zenrows.com/) or [ScraperAPI](https://www.scraperapi.com/) offer a way to use their API via a set of proxies passed to `requests.get()`

Example using `ProxyWebScraperDriver` directly:

```python
--8<-- "docs/griptape-framework/drivers/src/web_scraper_drivers_1.py"
```

### Markdownify

!!! info

    This driver requires the `drivers-web-scraper-markdownify` [extra](../index.md#extras) and the
    playwright browsers to be installed.

    To install the playwright browsers, run `playwright install` in your terminal. If you are using
    uv, run `uv run playwright install` instead. The `playwright` command should already be
    installed as a dependency of the `drivers-web-scraper-markdownify` extra. For more details about
    playwright, see [the playwright docs](https://playwright.dev/python/docs/library).

    Note that if you skip installing the playwright browsers, you will see the following error when
    you run your code:

    ```
    playwright._impl._errors.Error: Executable doesn't exist at ...
    ╔════════════════════════════════════════════════════════════╗
    ║ Looks like Playwright was just installed or updated.       ║
    ║ Please run the following command to download new browsers: ║
    ║                                                            ║
    ║     playwright install                                     ║
    ║                                                            ║
    ║ <3 Playwright Team                                         ║
    ╚════════════════════════════════════════════════════════════╝
    ```

The [MarkdownifyWebScraperDriver](../../reference/griptape/drivers/web_scraper/markdownify_web_scraper_driver.md) outputs the scraped text in markdown format. It uses [playwright](https://pypi.org/project/playwright/) to render web pages along with dynamically loaded content, and a combination of [beautifulsoup4](https://pypi.org/project/beautifulsoup4/) and [markdownify](https://pypi.org/project/markdownify/) to produce a markdown representation of a webpage. It makes a best effort to produce a markdown representation of a webpage that is concise yet human (and LLM) readable.

Example using `MarkdownifyWebScraperDriver` directly:

```python
--8<-- "docs/griptape-framework/drivers/src/web_scraper_drivers_2.py"
```

Example of using `MarkdownifyWebScraperDriver` with an agent:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/web_scraper_drivers_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/web_scraper_drivers_3.txt"
    ```


### Trafilatura

!!! info

    This driver requires the `drivers-web-scraper-trafilatura` [extra](../index.md#extras).

The [TrafilaturaWebScraperDriver](../../reference/griptape/drivers/web_scraper/trafilatura_web_scraper_driver.md) scrapes text from a webpage using the [Trafilatura](https://trafilatura.readthedocs.io) library.

Example of using `TrafilaturaWebScraperDriver` directly:

```python
--8<-- "docs/griptape-framework/drivers/src/web_scraper_drivers_4.py"
```
