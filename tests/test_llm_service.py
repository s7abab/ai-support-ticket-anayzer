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
    assert body["response_format"] == {"type": "json_object"}
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
