---
search:
  boost: 2
---

## Overview

Web Search Drivers can be used to search for links from a search query. They are used by [WebSearch](../../reference/griptape/tools/web_search/tool.md) to provide its functionality. All Web Search Drivers implement the following methods:

- `search()` searches the web and returns a [ListArtifact](../../reference/griptape/artifacts/list_artifact.md) that contains JSON-serializable [TextArtifact](../../reference/griptape/artifacts/text_artifact.md)s with the search results.

You can use Web Search Drivers with [Structures](../structures/agents.md):

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/web_search_drivers_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/web_search_drivers_5.txt"
    ```


```
PromptTask 45a53f1024494baab41a1f10a67017b1
    Output: Here are some websites with information about AI
    frameworks:

      1. [The Top 16 AI Frameworks and Libraries: A Beginner's Guide -
      DataCamp](https://www.datacamp.com/blog/top-ai-frameworks-and-lib
      raries)
      2. [AI Frameworks: Top Types To Adopt in 2024 -
      Splunk](https://www.splunk.com/en_us/blog/learn/ai-frameworks.htm
      l)
      3. [Top AI Frameworks in 2024: A Review -
      BairesDev](https://www.bairesdev.com/blog/ai-frameworks/)
      4. [The Top 16 AI Frameworks and Libraries - AI
      Slackers](https://aislackers.com/the-top-16-ai-frameworks-and-lib
      raries/)
      5. [Top AI Frameworks in 2024: Artificial Intelligence Frameworks
      Comparison - Clockwise
      Software](https://clockwise.software/blog/artificial-intelligence
      -framework/)
```

Or use them independently:

```python
--8<-- "docs/griptape-framework/drivers/src/web_search_drivers_3.py"
```

```
{"title": "The Top 16 AI Frameworks and Libraries: A Beginner's Guide", "url": "https://www.datacamp.com/blog/top-ai-frameworks-and-libraries", "description": "PyTorch. Torch is an open-source machine learning library known for its dynamic computational graph and is favored by researchers. The framework is excellent for prototyping and experimentation. Moreover, it's empowered by growing community support, with tools like PyTorch being built on the library."}

{"title": "Top 11 AI Frameworks and Tools in 2024 | Fively | 5ly.co", "url": "https://5ly.co/blog/best-ai-frameworks/", "description": "Discover the top 11 modern artificial intelligence tools and frameworks to build robust architectures for your AI-powered apps. ... - Some advanced use cases may need further fine-tuning. Caffe 2. Now we move on to deep learning tools and frameworks. The first one is Caffe 2: an open-source deep learning framework with modularity and speed in ..."}

{"title": "The Top 16 AI Frameworks and Libraries | AI Slackers", "url": "https://aislackers.com/the-top-16-ai-frameworks-and-libraries/", "description": "Experiment with different frameworks to find the one that aligns with your needs and goals as a data practitioner. Embrace the world of AI frameworks, and embark on a journey of building smarter software with confidence. Discover the top AI frameworks and libraries like PyTorch, Scikit-Learn, TensorFlow, Keras, LangChain, and more."}
```

## Web Search Drivers

### Google

The [GoogleWebSearchDriver](../../reference/griptape/drivers/web_search/google_web_search_driver.md) uses the [Google Custom Search JSON API](https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list) for web searching.

Example using `GoogleWebSearchDriver` directly:

```python
--8<-- "docs/griptape-framework/drivers/src/web_search_drivers_1.py"
```

### DuckDuckGo

!!! info

    This driver requires the `drivers-web-search-duckduckgo` [extra](../index.md#extras).

The [DuckDuckGoWebSearchDriver](../../reference/griptape/drivers/web_search/duck_duck_go_web_search_driver.md) uses the [duckduckgo_search](https://github.com/deedy5/duckduckgo_search) SDK for web searching.

Example of using `DuckDuckGoWebSearchDriver` directly:

```python
--8<-- "docs/griptape-framework/drivers/src/web_search_drivers_3.py"
```

### Tavily

!!! info

    This driver requires the `drivers-web-search-tavily` [extra](../index.md#extras), and a Tavily [api key](https://app.tavily.com).

Example of using `TavilyWebSearchDriver` directly:

```python
--8<-- "docs/griptape-framework/drivers/src/web_search_drivers_4.py"
```

### Exa

!!! info

    This driver requires the `drivers-web-search-exa` [extra](../index.md#extras),
    and an Exa [api key](https://dashboard.exa.ai/api-keys)

Example of using `ExaWebSearchDriver` directly:

```python
--8<-- "docs/griptape-framework/drivers/src/web_search_drivers_6.py"
```
