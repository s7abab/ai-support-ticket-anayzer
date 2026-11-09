import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.exceptions import (
    LLMConfigError,
    LLMInvalidResponseError,
    LLMServiceError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.schemas import ErrorResponse

logger = logging.getLogger(__name__)

# Fixed, client-safe responses for each service failure. Exception messages
# and causes are only logged, never returned, so provider URLs, keys, or raw
# provider output cannot leak to clients.
SERVICE_ERRORS: dict[type[LLMServiceError], tuple[int, str, str]] = {
    LLMConfigError: (503, "service_misconfigured", "The analysis service is not configured."),
    ProviderTimeoutError: (504, "provider_timeout", "The analysis provider timed out."),
    ProviderUnavailableError: (503, "provider_unavailable", "The analysis provider is unavailable."),
    LLMInvalidResponseError: (502, "provider_invalid_response", "The analysis provider returned an invalid response."),
}
DEFAULT_SERVICE_ERROR = (503, "provider_unavailable", "The analysis provider is unavailable.")

ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in {422, 500, *(s for s, _, _ in SERVICE_ERRORS.values())}
}


def error_response(status: int, code: str, message: str, fields: list | None = None) -> JSONResponse:
    error = {"code": code, "message": message}
    if fields is not None:
        error["fields"] = fields
    return JSONResponse(status_code=status, content={"error": error})


async def handle_service_error(request: Request, exc: LLMServiceError) -> JSONResponse:
    status, code, message = SERVICE_ERRORS.get(type(exc), DEFAULT_SERVICE_ERROR)
    # Log only the error code and cause type: cause messages can contain
    # provider response bodies, which are not safe to persist in logs.
    cause = type(exc.__cause__).__name__ if exc.__cause__ else None
    logger.warning("Analysis failed: code=%s cause=%s", code, cause)
    return error_response(status, code, message)


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Report which fields failed and why, but never echo the submitted input.
    fields = [
        {
            # A malformed JSON body reports a character offset, not a field.
            "field": "body"
            if err["type"] == "json_invalid"
            else ".".join(str(part) for part in err["loc"] if part != "body") or "body",
            "message": err["msg"],
        }
        for err in exc.errors()
    ]
    return error_response(422, "invalid_request", "The request is invalid.", fields)


async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    return error_response(exc.status_code, code, str(exc.detail))


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return error_response(500, "internal_error", "An unexpected error occurred.")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(LLMServiceError, handle_service_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
