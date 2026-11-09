from fastapi import FastAPI, HTTPException

from app.exceptions import (
    LLMConfigError,
    LLMInvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services import llm_service

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Support Ticket Analyzer API!"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_ticket(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return await llm_service.analyze_ticket(request.message)
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ProviderTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except LLMInvalidResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
