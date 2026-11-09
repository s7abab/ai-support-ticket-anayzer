import asyncio
import json

import httpx
from pydantic import ValidationError

from app.config import get_llm_settings
from app.exceptions import (
    LLMInvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.schemas import AnalyzeResponse

SYSTEM_PROMPT = """You are a support ticket analyzer.
Read the customer's message and respond with ONLY a JSON object with these keys:
- "category": one of "billing", "technical", "account", "general"
- "priority": one of "low", "medium", "high"
- "sentiment": one of "positive", "neutral", "negative"
- "summary": a one-sentence summary of the ticket, at most 300 characters
Do not include any other text."""

MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
# Rate limiting and server-side errors are worth retrying; other 4xx are not.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


async def _post_with_retries(
    client: httpx.AsyncClient, url: str, headers: dict, payload: dict
) -> httpx.Response:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        is_last = attempt == MAX_ATTEMPTS
        try:
            res = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            if is_last:
                raise ProviderTimeoutError("LLM provider timed out") from exc
        except httpx.TransportError as exc:
            if is_last:
                raise ProviderUnavailableError("LLM provider unavailable") from exc
        else:
            if res.is_success:
                return res
            if res.status_code not in RETRYABLE_STATUS_CODES or is_last:
                raise ProviderUnavailableError("LLM provider unavailable")

        # Exponential backoff: 1s, then 2s.
        await asyncio.sleep(RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1))

    raise AssertionError("unreachable")


async def analyze_ticket(message: str) -> AnalyzeResponse:
    settings = get_llm_settings()

    headers = {"Authorization": f"Bearer {settings.api_key}"}
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        res = await _post_with_retries(client, settings.api_url, headers, payload)

    try:
        content = res.json()["choices"][0]["message"]["content"]
        return AnalyzeResponse.model_validate(json.loads(content))
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
        raise LLMInvalidResponseError("LLM provider returned invalid output") from exc
