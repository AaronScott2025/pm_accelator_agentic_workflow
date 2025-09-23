from __future__ import annotations

from typing import List

from src.ai_interviewer_pm.settings import settings
from langchain_community.tools.tavily_search import TavilySearchResults


def internet_search(query: str, *, k: int = 5) -> List[dict]:
    """Search the internet using Tavily and return structured results.

    Args:
        query: The search query.
        k: Number of results.

    Returns:
        List of result dicts with title, url, and content snippet.
    """
    tool = TavilySearchResults(max_results=k)
    return tool.invoke({"query": query, "api_key": settings.tavily_api_key})
