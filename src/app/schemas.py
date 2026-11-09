
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class AnalyzeRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=5000
    )


class AnalyzeResponse(BaseModel):
    # Provider output is untrusted: reject keys outside the contract instead of
    # silently dropping them, and reject wrong types instead of coercing them
    # (e.g. a numeric or boolean value where a string is expected). Surrounding
    # whitespace is stripped before length checks, so "   " is not a summary.
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

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



class FieldError(BaseModel):
    field: str
    message: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    fields: list[FieldError] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
