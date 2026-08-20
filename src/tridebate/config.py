"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    tavily_api_key: str
    llm_base_url: str
    llm_model_name: str
    max_search_results: int = 10
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512


def load_settings() -> Settings:
    """Load and validate application settings from environment variables."""
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise EnvironmentError("TAVILY_API_KEY is not set in the environment.")

    return Settings(
        tavily_api_key=tavily_api_key,
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        llm_model_name=os.environ.get(
            "LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"
        ),
    )
