# tests/test_api.py
#
# pytest is Python's standard testing framework. It discovers test files
# (any file named test_*.py or *_test.py), discovers test functions
# (any function named test_*), runs them, and reports which passed and failed.
#
# Testing a FastAPI app without a real server requires two tools:
#   - httpx.AsyncClient: an async HTTP client that can send real HTTP requests
#   - ASGITransport: tells httpx to send requests directly to the ASGI app
#     object in memory, bypassing the network entirely. This means tests run
#     in milliseconds and need no open ports.

import pytest
from httpx import AsyncClient, ASGITransport
from clv_uplift.api.main import create_app

# create_app() gives us a fresh app instance for testing.
# We call the factory rather than importing the module-level `app` object
# because the factory ensures each test gets a clean, unconfigured app
# without any state left over from previous tests.
app = create_app()


# @pytest.mark.asyncio tells pytest that this test function is a coroutine
# and should be run inside an async event loop. Without this decorator,
# pytest would not know how to execute an async function — it would just
# see a coroutine object and never actually run it.
@pytest.mark.asyncio
async def test_health():
    """
    The simplest possible test: confirm the health endpoint is alive
    and returns the expected status field.
    """
    # AsyncClient is used as a context manager (async with) so that the
    # client connection is properly opened before the request and closed
    # after, even if the test raises an exception.
    #
    # ASGITransport(app=app) is the critical line. It tells httpx:
    # "instead of opening a real TCP connection to a server, call this
    # ASGI app object directly." The request goes through the full FastAPI
    # machinery — routing, dependency injection, Pydantic validation —
    # but never touches the network.
    #
    # base_url="http://test" is a required placeholder. httpx needs a base
    # URL to construct full URLs from relative paths, but since we are using
    # ASGITransport, the actual host is irrelevant — no DNS lookup happens.
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        resp = await client.get("/health")

    # assert statements are how pytest checks correctness.
    # If an assertion fails, pytest marks the test as failed and shows
    # you exactly what the actual value was versus what you expected.
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_score_valid():
    """
    Test the happy path: a completely valid request should return 200
    and a response containing the expected fields within valid ranges.
    """
    # This payload satisfies every constraint in CustomerFeatures:
    # customer_id is a string, recency_days is between 0 and 3650,
    # frequency is >= 1, monetary_value is >= 0, clv_segment is valid.
    payload = {
        "customer_id": "99",
        "recency_days": 7,
        "frequency": 10,
        "monetary_value": 500.0,
        "clv_segment": "Loyal",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/score", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    # Confirm the response has the field we care about
    assert "composite_score" in data

    # Confirm the score is within the declared range [0, 1].
    # This is testing a business invariant: no matter what input you send,
    # the composite score should always be a valid probability-like value.
    assert 0.0 <= data["composite_score"] <= 1.0


@pytest.mark.asyncio
async def test_score_invalid_segment():
    """
    Test that an invalid clv_segment value is rejected with a 422.
    This confirms that Pydantic's Literal type constraint is working.
    """
    payload = {
        "customer_id": "99",
        "recency_days": 7,
        "frequency": 10,
        "monetary_value": 500.0,
        "clv_segment": "NotASegment",  # not in the Literal type
    }
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/score", json=payload)

    # 422 Unprocessable Entity is FastAPI's standard response for
    # Pydantic validation failures. We are asserting that the validation
    # layer is doing its job — bad input never reaches the model.
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_score_missing_frequency():
    """
    Test that a missing required field produces a 422.
    This is the scenario from your question: frequency is simply absent
    from the JSON. Pydantic should detect the missing required field
    and FastAPI should return 422 before the endpoint function runs.
    """
    payload = {
        "customer_id": "99",
        "recency_days": 7,
        # frequency is intentionally missing
        "monetary_value": 500.0,
        "clv_segment": "Loyal",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/score", json=payload)

    assert resp.status_code == 422

    # We can also inspect the error detail to confirm Pydantic correctly
    # identified WHICH field is missing, not just that something is wrong.
    detail = resp.json()["detail"]

    # detail is a list of error objects. We check that at least one of them
    # points to "frequency" as the location of the problem.
    missing_fields = [err["loc"] for err in detail]
    assert any("frequency" in loc for loc in missing_fields)


@pytest.mark.asyncio
async def test_score_negative_recency_rejected():
    """
    Test that a negative recency_days value is rejected.
    This confirms the ge=0 constraint in Field() is enforced.
    """
    payload = {
        "customer_id": "99",
        "recency_days": -5,   # violates ge=0
        "frequency": 10,
        "monetary_value": 500.0,
        "clv_segment": "Loyal",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/score", json=payload)

    assert resp.status_code == 422

# --- Phase 3 additions: /predict and /explain degraded-contract tests ------------------
# These assert the CONTRACT regardless of whether a trained bundle exists yet, so the
# suite stays green before training is run (503) and after (200 with the documented body).

@pytest.mark.asyncio
async def test_predict_contract():
    """
    /predict returns 503 (no bundle yet) or 200 with a CATE estimate and an explicit
    targeting_certified flag. Payload is the raw-RFM CATEFeatures shape.
    """
    payload = {
        "customer_id": "99",
        "recency_days": 7,
        "frequency": 10,
        "monetary_value": 500.0,
        "cancel_rate": 0.05,
        "clv_segment": "Loyal",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/predict", json=payload)

    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "cate_estimate" in data
        assert "targeting_certified" in data
        assert data["confidence"] in ("high", "medium", "low", "not_certified")


@pytest.mark.asyncio
async def test_explain_contract():
    """
    /explain returns 503 (no bundle yet) or 200 with explanation_available True/False.
    A withheld explanation is HTTP 200, not an error.
    """
    payload = {
        "customer_id": "99",
        "recency_days": 7,
        "frequency": 10,
        "monetary_value": 500.0,
        "cancel_rate": 0.05,
        "clv_segment": "Loyal",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/explain", json=payload)

    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "explanation_available" in data
        assert "surrogate_r2" in data
        if not data["explanation_available"]:
            assert data["explanation_unavailable_reason"] is not None
            assert data["top_features"] == []