"""Client for the locally hosted LLM, served via vLLM through an
OpenAI-compatible API."""

from openai import OpenAI

from tridebate.config import Settings


class LLMClient:
    """Wrapper around the OpenAI-compatible client, pointed at a local vLLM server."""

    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(base_url=settings.llm_base_url, api_key="not-needed")
        self._model = settings.llm_model_name
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return its raw text response."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""
