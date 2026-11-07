
from pydantic import BaseModel, Field
from typing import Literal


class AnalyzeRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=5000
    )


class AnalyzeResponse(BaseModel):
    category: Literal[
        "billing",
        "technical",
        "account",
        "general"
    ]

    priority: Literal[
        "low",
        "medium",
        "high"
    ]

    sentiment: Literal[
        "positive",
        "neutral",
        "negative"
    ]

    summary: str = Field(
        min_length=1,
        max_length=300
    )
