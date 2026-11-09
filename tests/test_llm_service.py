"""End-to-end tests for POST /analyze with the LLM provider mocked out.

No real provider is ever contacted: every httpx.AsyncClient the service
creates is given an httpx.MockTransport that replays scripted responses and
records each outgoing request. Retry backoff sleeps are recorded instead of
awaited, so the retry tests run instantly.
"""

import functools
import json
from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import llm_service

API_URL = "https://llm.test/v1/chat/completions"
API_KEY = "test-secret-key"
MODEL = "test-model"

VALID_ANALYSIS = {
    "category": "billing",
    "priority": "high",
    "sentiment": "negative",
    "summary": "Customer was charged twice for the same subscription.",
}

Step = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


def completion(content: str) -> httpx.Response:
    """An OpenAI-style chat completion whose message content is `content`."""
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


class FakeProvider:
    """Replays one scripted step per request and records what was sent."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.requests: list[httpx.Request] = []

    def script(self, *steps: Step) -> None:
        self.steps = list(steps)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.steps:
            raise AssertionError(f"unexpected provider call #{len(self.requests)}")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, httpx.Response):
            return step(request)
        return step

    @property
    def attempts(self) -> int:
        return len(self.requests)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    fake = FakeProvider()
    transport = httpx.MockTransport(fake.handler)
    monkeypatch.setattr(
        llm_service.httpx,
        "AsyncClient",
        functools.partial(httpx.AsyncClient, transport=transport),
    )
    return fake


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Records retry backoff delays instead of waiting them out."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(llm_service.asyncio, "sleep", fake_sleep)
    return delays


@pytest.fixture(autouse=True)
def llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_URL", API_URL)
    monkeypatch.setenv("LLM_API_KEY", API_KEY)
    monkeypatch.setenv("LLM_MODEL", MODEL)
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "5")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def analyze(client: TestClient, message: str = "I was billed twice this month!") -> httpx.Response:
    return client.post("/analyze", json={"message": message})


# --- Valid provider response -------------------------------------------------


def test_valid_provider_response_returns_analysis(client, provider, sleeps):
    provider.script(completion(json.dumps(VALID_ANALYSIS)))

    res = analyze(client, "I was billed twice this month!")

    assert res.status_code == 200
    assert res.json() == VALID_ANALYSIS
    assert provider.attempts == 1
    assert sleeps == []


def test_valid_request_is_forwarded_to_provider_correctly(client, provider, sleeps):
    provider.script(completion(json.dumps(VALID_ANALYSIS)))

    analyze(client, "I was billed twice this month!")

    [request] = provider.requests
    assert request.method == "POST"
    assert str(request.url) == API_URL
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    body = json.loads(request.content)
    assert body["model"] == MODEL
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][-1] == {"role": "user", "content": "I was billed twice this month!"}


# --- Invalid request body ----------------------------------------------------


@pytest.mark.parametrize(
    ("request_kwargs", "expected_field"),
    [
        pytest.param({"json": {}}, "message", id="missing-message"),
        pytest.param({"json": {"message": ""}}, "message", id="empty-message"),
        pytest.param({"json": {"message": "x" * 5001}}, "message", id="message-too-long"),
        pytest.param({"json": {"message": 123}}, "message", id="wrong-type"),
        pytest.param(
            {"content": b"{not json", "headers": {"Content-Type": "application/json"}},
            "body",
            id="malformed-json",
        ),
    ],
)
def test_invalid_request_body_returns_422_without_calling_provider(
    client, provider, request_kwargs, expected_field
):
    res = client.post("/analyze", **request_kwargs)

    assert res.status_code == 422
    error = res.json()["error"]
    assert error["code"] == "invalid_request"
    assert [f["field"] for f in error["fields"]] == [expected_field]
    assert provider.attempts == 0


def test_invalid_request_does_not_echo_submitted_input(client, provider):
    secret_input = "x" * 5001

    res = client.post("/analyze", json={"message": secret_input})

    assert res.status_code == 422
    assert secret_input not in res.text


# --- Provider timeout --------------------------------------------------------


def test_provider_timeout_returns_controlled_504_after_retries(client, provider, sleeps):
    provider.script(*[httpx.ReadTimeout("timed out") for _ in range(llm_service.MAX_ATTEMPTS)])

    res = analyze(client)

    assert res.status_code == 504
    assert res.json() == {
        "error": {"code": "provider_timeout", "message": "The analysis provider timed out."}
    }
    assert provider.attempts == llm_service.MAX_ATTEMPTS
    assert sleeps == [1.0, 2.0]


def test_provider_timeout_then_success_recovers(client, provider, sleeps):
    provider.script(httpx.ConnectTimeout("timed out"), completion(json.dumps(VALID_ANALYSIS)))

    res = analyze(client)

    assert res.status_code == 200
    assert res.json() == VALID_ANALYSIS
    assert provider.attempts == 2
    assert sleeps == [1.0]


# --- Provider returns 503 temporarily ---------------------------------------


def test_temporary_503_is_retried_then_succeeds(client, provider, sleeps):
    provider.script(httpx.Response(503), completion(json.dumps(VALID_ANALYSIS)))

    res = analyze(client)

    assert res.status_code == 200
    assert res.json() == VALID_ANALYSIS
    assert provider.attempts == 2
    assert sleeps == [1.0]


def test_two_503s_then_success_uses_exponential_backoff(client, provider, sleeps):
    provider.script(
        httpx.Response(503), httpx.Response(503), completion(json.dumps(VALID_ANALYSIS))
    )

    res = analyze(client)

    assert res.status_code == 200
    assert provider.attempts == 3
    assert sleeps == [1.0, 2.0]


def test_persistent_503_stops_after_max_attempts(client, provider, sleeps):
    provider.script(*[httpx.Response(503) for _ in range(llm_service.MAX_ATTEMPTS)])

    res = analyze(client)

    assert res.status_code == 503
    assert res.json() == {
        "error": {"code": "provider_unavailable", "message": "The analysis provider is unavailable."}
    }
    # The fake raises on any call beyond the scripted ones, so this also
    # proves no fourth attempt was made.
    assert provider.attempts == llm_service.MAX_ATTEMPTS
    assert sleeps == [1.0, 2.0]


# --- Provider returns 401 ----------------------------------------------------


def test_401_is_not_retried(client, provider, sleeps):
    provider.script(httpx.Response(401, json={"error": {"message": f"bad key {API_KEY}"}}))

    res = analyze(client)

    assert res.status_code == 503
    assert res.json() == {
        "error": {"code": "provider_unavailable", "message": "The analysis provider is unavailable."}
    }
    assert provider.attempts == 1
    assert sleeps == []
    # Provider error bodies (which may echo credentials) never reach the client.
    assert API_KEY not in res.text


# --- Provider returns malformed JSON ----------------------------------------


@pytest.mark.parametrize(
    "provider_response",
    [
        pytest.param(httpx.Response(200, text="<html>oops</html>"), id="body-not-json"),
        pytest.param(httpx.Response(200, json={"unexpected": "shape"}), id="missing-choices"),
        pytest.param(httpx.Response(200, json={"choices": []}), id="empty-choices"),
        pytest.param(completion("Sure! Here is the analysis: {"), id="content-not-json"),
        pytest.param(
            completion(json.dumps({**VALID_ANALYSIS, "category": "shipping"})),
            id="content-fails-schema",
        ),
    ],
)
def test_malformed_provider_output_returns_controlled_502(
    client, provider, sleeps, provider_response
):
    provider.script(provider_response)

    res = analyze(client)

    assert res.status_code == 502
    assert res.json() == {
        "error": {
            "code": "provider_invalid_response",
            "message": "The analysis provider returned an invalid response.",
        }
    }
    # A successful HTTP response with bad content is not retried.
    assert provider.attempts == 1
    assert sleeps == []
    assert "oops" not in res.text


# --- Provider output fails the response contract ----------------------------


def _without(key: str) -> dict:
    return {k: v for k, v in VALID_ANALYSIS.items() if k != key}


@pytest.mark.parametrize(
    "content",
    [
        # Required fields must be present.
        *[pytest.param(_without(f), id=f"missing-{f}") for f in VALID_ANALYSIS],
        # Categories, priorities and sentiments come from fixed allowed values.
        pytest.param({**VALID_ANALYSIS, "priority": "urgent"}, id="priority-not-allowed"),
        pytest.param({**VALID_ANALYSIS, "sentiment": "angry"}, id="sentiment-not-allowed"),
        pytest.param({**VALID_ANALYSIS, "category": "Billing"}, id="category-wrong-case"),
        # Summaries have sensible length limits.
        pytest.param({**VALID_ANALYSIS, "summary": ""}, id="summary-empty"),
        pytest.param({**VALID_ANALYSIS, "summary": "   "}, id="summary-whitespace-only"),
        pytest.param({**VALID_ANALYSIS, "summary": "x" * 301}, id="summary-too-long"),
        # Unexpected fields are rejected, not silently dropped.
        pytest.param({**VALID_ANALYSIS, "confidence": 0.9}, id="extra-field"),
        # Wrong types are rejected rather than coerced.
        pytest.param({**VALID_ANALYSIS, "summary": 123}, id="summary-is-number"),
        pytest.param({**VALID_ANALYSIS, "priority": 1}, id="priority-is-number"),
        pytest.param({**VALID_ANALYSIS, "category": None}, id="category-is-null"),
        pytest.param({**VALID_ANALYSIS, "summary": ["a", "b"]}, id="summary-is-list"),
        # The content must be a JSON object at all.
        pytest.param([VALID_ANALYSIS], id="content-is-array"),
        pytest.param("billing", id="content-is-string"),
    ],
)
def test_provider_output_violating_contract_returns_controlled_502(
    client, provider, sleeps, content
):
    provider.script(completion(json.dumps(content)))

    res = analyze(client)

    assert res.status_code == 502
    assert res.json() == {
        "error": {
            "code": "provider_invalid_response",
            "message": "The analysis provider returned an invalid response.",
        }
    }
    assert provider.attempts == 1


@pytest.mark.parametrize(
    "message",
    [
        pytest.param({"content": None}, id="content-null"),
        pytest.param({"content": VALID_ANALYSIS}, id="content-already-object"),
        pytest.param({}, id="content-missing"),
    ],
)
def test_provider_message_without_string_content_returns_502(client, provider, sleeps, message):
    provider.script(httpx.Response(200, json={"choices": [{"message": message}]}))

    res = analyze(client)

    assert res.status_code == 502
    assert res.json()["error"]["code"] == "provider_invalid_response"


def test_summary_at_limit_is_accepted_and_whitespace_trimmed(client, provider, sleeps):
    summary = "x" * 300
    provider.script(completion(json.dumps({**VALID_ANALYSIS, "summary": f"  {summary}\n"})))

    res = analyze(client)

    assert res.status_code == 200
    assert res.json()["summary"] == summary


def test_provider_is_sent_strict_schema_matching_response_contract(client, provider, sleeps):
    provider.script(completion(json.dumps(VALID_ANALYSIS)))

    analyze(client)

    json_schema = json.loads(provider.requests[0].content)["response_format"]["json_schema"]
    assert json_schema["name"] == "ticket_analysis"
    assert json_schema["strict"] is True
    schema = json_schema["schema"]
    # Strict mode requires every field and no extra keys.
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(VALID_ANALYSIS)
    assert schema["properties"]["category"]["enum"] == ["billing", "technical", "account", "general"]
    assert schema["properties"]["priority"]["enum"] == ["low", "medium", "high"]
    assert schema["properties"]["sentiment"]["enum"] == ["positive", "neutral", "negative"]
    # Length keywords are removed for strict mode; validation still enforces them.
    assert "maxLength" not in schema["properties"]["summary"]
    assert "minLength" not in schema["properties"]["summary"]


def test_strict_schema_does_not_mutate_public_response_schema():
    from app.schemas import AnalyzeResponse

    summary = AnalyzeResponse.model_json_schema()["properties"]["summary"]
    assert summary["maxLength"] == 300
    assert summary["minLength"] == 1
