import json

import httpx
from pydantic import ValidationError

from app.config import get_llm_settings
from app.exceptions import (
    LLMInvalidResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.schemas import AnalyzeResponse

SYSTEM_PROMPT = """You are a support ticket analyzer.
Read the customer's message and respond with ONLY a JSON object with these keys:
- "category": one of "billing", "technical", "account", "general"
- "priority": one of "low", "medium", "high"
- "sentiment": one of "positive", "neutral", "negative"
- "summary": a one-sentence summary of the ticket, at most 300 characters
Do not include any other text."""


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

    try:
        async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
            res = await client.post(settings.api_url, headers=headers, json=payload)
            res.raise_for_status()
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError("LLM provider timed out") from exc
    except httpx.HTTPError as exc:
        raise LLMUnavailableError("LLM provider unavailable") from exc

    try:
        content = res.json()["choices"][0]["message"]["content"]
        return AnalyzeResponse.model_validate(json.loads(content))
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
        raise LLMInvalidResponseError("LLM provider returned invalid output") from exc
