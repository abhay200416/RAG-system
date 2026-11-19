import asyncio

from redis.exceptions import ConnectionError as RedisConnectionError

from rag.api.middleware import AuthRateLimitRequestIDMiddleware
from rag.util.logging import get_correlation_id


async def _app(scope, _receive, send):
    assert get_correlation_id() == scope["state"]["request_id"]
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"x-request-id", b"untrusted"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})


def _request(middleware, *, path="/query", headers=None, client_ip="127.0.0.1"):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": headers or [],
        "client": (client_ip, 12345),
    }
    asyncio.run(middleware(scope, receive, send))
    return messages, scope


def _status(messages):
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


def _response_headers(messages):
    start = next(
        message
        for message in messages
        if message["type"] == "http.response.start"
    )
    return dict(start["headers"])


def test_authentication_and_server_generated_request_id():
    middleware = AuthRateLimitRequestIDMiddleware(
        _app,
        api_key="test-key",
        redis_url="",
    )
    messages, scope = _request(
        middleware,
        headers=[(b"authorization", b"Bearer test-key")],
    )

    headers = _response_headers(messages)
    assert _status(messages) == 200
    assert headers[b"x-request-id"] == scope["state"]["request_id"].encode()
    assert headers[b"x-request-id"] != b"untrusted"
    assert headers[b"x-ratelimit-remaining"] == b"59"


def test_invalid_bearer_token_is_rejected_with_request_id():
    middleware = AuthRateLimitRequestIDMiddleware(
        _app,
        api_key="test-key",
        redis_url="",
    )

    messages, _scope = _request(
        middleware,
        headers=[(b"authorization", b"Bearer wrong-key")],
    )

    assert _status(messages) == 401
    assert b"x-request-id" in _response_headers(messages)


def test_rate_limit_returns_429_and_retry_after():
    middleware = AuthRateLimitRequestIDMiddleware(
        _app,
        api_key="test-key",
        redis_url="",
        rate_limit_requests=1,
        rate_limit_window_seconds=60,
    )
    headers = [(b"authorization", b"Bearer test-key")]

    first, _scope = _request(middleware, headers=headers)
    second, _scope = _request(middleware, headers=headers)

    assert _status(first) == 200
    assert _status(second) == 429
    assert b"retry-after" in _response_headers(second)


def test_public_path_does_not_require_auth():
    middleware = AuthRateLimitRequestIDMiddleware(
        _app,
        api_key="test-key",
        redis_url="",
    )

    messages, _scope = _request(middleware, path="/health")

    assert _status(messages) == 200


def test_missing_api_key_fails_closed():
    middleware = AuthRateLimitRequestIDMiddleware(
        _app,
        api_key="",
        redis_url="",
    )

    messages, _scope = _request(middleware)

    assert _status(messages) == 503


def test_redis_rate_limiter_failure_fails_closed():
    class FailedRedis:
        async def eval(self, *_args):
            raise RedisConnectionError("offline")

    middleware = AuthRateLimitRequestIDMiddleware(
        _app,
        api_key="test-key",
        redis_url="",
        redis_client=FailedRedis(),
    )

    messages, _scope = _request(middleware)

    assert _status(messages) == 503
