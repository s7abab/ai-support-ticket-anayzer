# Support Ticket Analyzer

FastAPI service that classifies support messages through an OpenAI-compatible
chat-completions provider.

## Setup

Copy `.env.example` to `.env`, then replace the placeholder values with the
endpoint, API key, and model supplied by your OpenAI-compatible provider.
Changing providers later only requires updating these values, not the
application code.

```sh
cp .env.example .env
# Edit .env and replace its placeholder values.
uv sync
uv run uvicorn app.main:app --app-dir src --env-file .env --reload
```

## Analyze a ticket

```sh
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"message":"My payment failed twice. Please help urgently."}'
```

The endpoint returns a category (`billing`, `technical`, `account`, or `general`),
priority (`low`, `medium`, or `high`), sentiment (`positive`, `neutral`, or
`negative`), and a summary of at most 300 characters. Request validation errors
return 422; provider timeouts, availability problems, and invalid output return
504, 503, and 502 respectively.

Every error uses the same shape, with a stable `code` and a client-safe
`message`. Validation errors also list the failing `fields`:

```json
{"error": {"code": "provider_timeout", "message": "The analysis provider timed out."}}
```

| Status | Code |
|---|---|
| 422 | `invalid_request` |
| 502 | `provider_invalid_response` |
| 503 | `provider_unavailable`, `service_misconfigured` |
| 504 | `provider_timeout` |
| 500 | `internal_error` |

## Tests

```sh
uv run pytest
```
