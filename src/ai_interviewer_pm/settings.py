from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables.

    Attributes:
        openai_api_key: API key for OpenAI models.
        cohere_api_key: API key for Cohere embeddings and models.
        tavily_api_key: API key for Tavily internet search.
        langsmith_api_key: API key for LangSmith tracing.
        langsmith_project: Project name for LangSmith traces grouping.
        qdrant_url: URL of Qdrant server.
        qdrant_api_key: API key for Qdrant (optional if local).
        qdrant_collection: Default collection name for chunks.
        data_dir: Local data directory path for ingestion.
    """

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    cohere_api_key: str | None = os.getenv("COHERE_API_KEY")
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")

    langsmith_api_key: str | None = os.getenv("LANGCHAIN_API_KEY")
    langsmith_project: str = os.getenv("LANGCHAIN_PROJECT", "ai-interviewer-pm")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "ai_interviewer_chunks")

    data_dir: str = os.getenv("DATA_DIR", "data")


settings = Settings()
