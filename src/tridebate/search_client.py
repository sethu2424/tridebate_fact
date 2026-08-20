"""Live web search via the Tavily API."""
import logging
from tavily import TavilyClient
from tridebate.config import Settings
from tridebate.models import SearchResult
logger = logging.getLogger(__name__)
class SearchClient:
    """Wrapper around Tavily for retrieving live web evidence."""
    def __init__(self, settings: Settings) -> None:
        self._client = TavilyClient(api_key=settings.tavily_api_key)
        self._max_results = settings.max_search_results
    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """Retrieve live web search results for a given query."""
        logger.info("Searching for: %s", query)
        response = self._client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results or self._max_results,
        )
        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
            )
            for item in response.get("results", [])
        ]
        logger.info("Retrieved %d results", len(results))
        return results