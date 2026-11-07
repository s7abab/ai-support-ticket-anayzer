import json
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Support Ticket Analyzer API!"}


SYSTEM_PROMPT = """You are a support ticket analyzer.
Read the customer's message and respond with ONLY a JSON object with these keys:
- "category": one of "billing", "technical", "account", "general"
- "priority": one of "low", "medium", "high"
- "sentiment": one of "positive", "neutral", "negative"
- "summary": a one-sentence summary of the ticket, at most 300 characters
Do not include any other text."""


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_ticket(request: AnalyzeRequest) -> AnalyzeResponse:
    api_url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not (api_url and api_key and model):
        raise HTTPException(status_code=503, detail="LLM provider is not configured")

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(api_url, headers=headers, json=payload)
            res.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM provider timed out")
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="LLM provider unavailable")

    try:
        content = res.json()["choices"][0]["message"]["content"]
        return AnalyzeResponse.model_validate(json.loads(content))
    except (KeyError, IndexError, TypeError, ValueError, ValidationError):
        raise HTTPException(status_code=502, detail="LLM provider returned invalid output")
