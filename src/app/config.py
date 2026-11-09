import os
from dataclasses import dataclass

from app.exceptions import LLMConfigError

DEFAULT_LLM_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class LLMSettings:
    api_url: str
    api_key: str
    model: str
    timeout_seconds: float


def get_llm_settings() -> LLMSettings:
    api_url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not (api_url and api_key and model):
        raise LLMConfigError("LLM provider is not configured")

    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS))
    except ValueError:
        timeout = 0
    if timeout <= 0:
        raise LLMConfigError("LLM timeout is misconfigured")

    return LLMSettings(api_url=api_url, api_key=api_key, model=model, timeout_seconds=timeout)
