from fastapi import FastAPI

from app.error_handlers import ERROR_RESPONSES, register_error_handlers
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services import llm_service

app = FastAPI()
register_error_handlers(app)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Support Ticket Analyzer API!"}


@app.post("/analyze", response_model=AnalyzeResponse, responses=ERROR_RESPONSES)
async def analyze_ticket(request: AnalyzeRequest) -> AnalyzeResponse:
    return await llm_service.analyze_ticket(request.message)
