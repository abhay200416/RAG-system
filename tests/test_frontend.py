from types import SimpleNamespace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fronend import streamlit as frontend


def test_request_api_sends_bearer_token_and_json(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return SimpleNamespace(
            headers={"x-request-id": "request-123"},
            ok=True,
            json=lambda: {"status": "ok"},
        )

    monkeypatch.setattr(frontend.requests, "request", fake_request)
    result = frontend.request_api(
        "POST",
        "/query",
        base_url="http://localhost:8000/",
        api_key=" secret ",
        json_body={"query": "What is in the docs?"},
    )

    assert result == {"status": "ok"}
    assert captured["url"] == "http://localhost:8000/query"
    assert captured["headers"] == {
        "Accept": "application/json",
        "Authorization": "Bearer secret",
    }
    assert captured["json"] == {"query": "What is in the docs?"}


def test_request_api_surfaces_http_detail_and_request_id(monkeypatch) -> None:
    monkeypatch.setattr(
        frontend.requests,
        "request",
        lambda *_args, **_kwargs: SimpleNamespace(
            headers={},
            ok=False,
            status_code=403,
            reason="Forbidden",
            json=lambda: {
                "detail": "URL host is not allow-listed",
                "request_id": "request-456",
            },
        ),
    )

    with pytest.raises(frontend.APIError, match="URL host is not allow-listed") as error:
        frontend.request_api(
            "POST",
            "/ingest/url",
            base_url="http://localhost:8000",
            api_key="secret",
            json_body={"url": "https://example.com"},
        )

    assert error.value.request_id == "request-456"


def test_request_api_rejects_urls_with_embedded_credentials() -> None:
    with pytest.raises(frontend.APIError, match="Do not put credentials"):
        frontend.request_api(
            "GET",
            "/health",
            base_url="https://user:password@example.com",
        )


def test_request_api_gives_startup_guidance_when_backend_is_offline(
    monkeypatch,
) -> None:
    def fail_to_connect(*_args, **_kwargs):
        raise frontend.requests.ConnectionError("connection refused")

    monkeypatch.setattr(frontend.requests, "request", fail_to_connect)

    with pytest.raises(frontend.APIError, match="uv run rag-api"):
        frontend.request_api(
            "POST",
            "/ingest/file",
            base_url="http://127.0.0.1:8000",
            api_key="secret",
        )


def test_request_api_explains_missing_backend_auth_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        frontend.requests,
        "request",
        lambda *_args, **_kwargs: SimpleNamespace(
            headers={"x-request-id": "request-503"},
            ok=False,
            status_code=503,
            reason="Service Unavailable",
            json=lambda: {"detail": "API authentication is not configured"},
        ),
    )

    with pytest.raises(frontend.APIError, match="restart `uv run rag-api`") as error:
        frontend.request_api(
            "POST",
            "/ingest/file",
            base_url="http://127.0.0.1:8000",
            api_key="secret",
        )

    assert error.value.request_id == "request-503"


def test_frontend_does_not_render_api_key_input() -> None:
    app = AppTest.from_file(str(Path(frontend.__file__))).run(timeout=30)

    assert not any(
        widget.label == "Bearer API key"
        for widget in app.text_input
    )
