from __future__ import annotations

import asyncio
import hashlib
import json
import math
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from rag.util.config import get_settings
from rag.util.logging import LoggingContext, get_logger

logger = get_logger(__name__)

ASGIMessage = dict[str, Any]
ASGIScope = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]

_REDIS_RATE_LIMIT_SCRIPT = """
local clock = redis.call('TIME')
local now = tonumber(clock[1])
local window = tonumber(ARGV[1])
local bucket = math.floor(now / window)
local key = KEYS[1] .. ':' .. bucket
local count = redis.call('INCR', key)
if count == 1 then
    redis.call('EXPIRE', key, window + 1)
end
local reset_after = ((bucket + 1) * window) - now
return {count, reset_after}
"""

_MAX_LOCAL_CLIENTS = 10_000


class _LocalRateLimitCapacityError(RuntimeError):
    pass


class _LocalRateLimiter:
    """Bounded fixed-window limiter for single-process development deployments."""

    def __init__(self, max_clients: int = _MAX_LOCAL_CLIENTS) -> None:
        self._windows: dict[str, tuple[int, int]] = {}
        self._lock = asyncio.Lock()
        self._max_clients = max_clients

    async def consume(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        now = time.monotonic()
        bucket = int(now // window_seconds)
        reset_after = max(
            1,
            math.ceil((bucket + 1) * window_seconds - now),
        )

        async with self._lock:
            if len(self._windows) >= self._max_clients and key not in self._windows:
                self._windows = {
                    client: entry
                    for client, entry in self._windows.items()
                    if entry[0] == bucket
                }
                if len(self._windows) >= self._max_clients:
                    raise _LocalRateLimitCapacityError(
                        "Local rate-limit client capacity reached"
                    )

            old_bucket, count = self._windows.get(key, (bucket, 0))
            if old_bucket != bucket:
                count = 0
            count += 1
            self._windows[key] = (bucket, count)
            remaining = max(0, limit - count)
            return count <= limit, remaining, reset_after


class AuthRateLimitRequestIDMiddleware:
    """ASGI middleware for bearer-key auth, per-IP rate limits, and request IDs.

    Configure ``RAG_API_AUTH_KEY`` to protect all paths except ``/health``.
    ``RAG_REDIS_URL`` enables a shared atomic Redis limiter; without it, limits
    are process-local and intended only for development or a single worker.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: str | None = None,
        redis_url: str | None = None,
        rate_limit_requests: int | None = None,
        rate_limit_window_seconds: int | None = None,
        public_paths: tuple[str, ...] = ("/health",),
        redis_client: Redis | None = None,
    ) -> None:
        settings = get_settings()
        self.app = app
        self.api_key = settings.api_auth_key if api_key is None else api_key
        self.redis_url = settings.redis_url if redis_url is None else redis_url
        self.rate_limit_requests = (
            settings.api_rate_limit_requests
            if rate_limit_requests is None
            else rate_limit_requests
        )
        self.rate_limit_window_seconds = (
            settings.api_rate_limit_window_seconds
            if rate_limit_window_seconds is None
            else rate_limit_window_seconds
        )
        self.rate_limit_prefix = settings.api_rate_limit_prefix
        self.public_paths = frozenset(public_paths)
        self._local_limiter = _LocalRateLimiter()
        self._redis = redis_client

        if (
            isinstance(self.rate_limit_requests, bool)
            or not isinstance(self.rate_limit_requests, int)
            or self.rate_limit_requests < 1
        ):
            raise ValueError("rate_limit_requests must be positive")
        if (
            isinstance(self.rate_limit_window_seconds, bool)
            or not isinstance(self.rate_limit_window_seconds, int)
            or self.rate_limit_window_seconds < 1
        ):
            raise ValueError("rate_limit_window_seconds must be positive")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string")
        if not self.rate_limit_prefix.strip():
            raise ValueError("rate_limit_prefix cannot be empty")
        if self._redis is None and self.redis_url:
            self._redis = Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )

    async def __call__(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if scope.get("type") == "lifespan":
            try:
                await self.app(scope, receive, send)
            finally:
                await self.aclose()
            return
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        state = scope.setdefault("state", {})
        if not isinstance(state, dict):
            state = {}
            scope["state"] = state
        state["request_id"] = request_id

        headers = self._headers_dict(scope)
        client_ip = self._client_ip(scope)
        rate_key = self._rate_limit_key(self.rate_limit_prefix, client_ip)
        with LoggingContext(request_id):
            try:
                allowed, remaining, reset_after = await self._consume_rate_limit(
                    rate_key
                )
            except RedisError:
                logger.exception("Redis rate limiter is unavailable")
                await self._send_error(
                    send,
                    status=503,
                    detail="Rate limiting is temporarily unavailable",
                    request_id=request_id,
                )
                return
            except _LocalRateLimitCapacityError:
                logger.exception("Local rate limiter is unavailable")
                await self._send_error(
                    send,
                    status=503,
                    detail="Rate limiting is temporarily unavailable",
                    request_id=request_id,
                )
                return

            limit_headers = [
                (b"x-ratelimit-limit", str(self.rate_limit_requests).encode("ascii")),
                (b"x-ratelimit-remaining", str(remaining).encode("ascii")),
                (b"x-ratelimit-reset", str(reset_after).encode("ascii")),
            ]
            if not allowed:
                await self._send_error(
                    send,
                    status=429,
                    detail="Rate limit exceeded",
                    request_id=request_id,
                    headers=limit_headers
                    + [(b"retry-after", str(reset_after).encode("ascii"))],
                )
                return

            if scope.get("path") not in self.public_paths:
                if not self.api_key or not self.api_key.strip():
                    logger.error("API authentication is not configured")
                    await self._send_error(
                        send,
                        status=503,
                        detail="API authentication is not configured",
                        request_id=request_id,
                        headers=limit_headers,
                    )
                    return
                if not self._authorized(headers):
                    await self._send_error(
                        send,
                        status=401,
                        detail="A valid bearer token is required",
                        request_id=request_id,
                        headers=limit_headers
                        + [(b"www-authenticate", b"Bearer")],
                    )
                    return

            async def send_with_request_headers(message: ASGIMessage) -> None:
                if message.get("type") == "http.response.start":
                    response_headers = [
                        (key, value)
                        for key, value in message.get("headers", [])
                        if key.lower() != b"x-request-id"
                        and not key.lower().startswith(b"x-ratelimit-")
                    ]
                    response_headers.extend(
                        limit_headers
                        + [(b"x-request-id", request_id.encode("ascii"))]
                    )
                    message = {**message, "headers": response_headers}
                await send(message)

            await self.app(scope, receive, send_with_request_headers)

    async def aclose(self) -> None:
        """Close the Redis connection pool, if one is configured."""
        if self._redis is not None:
            await self._redis.aclose()

    async def _consume_rate_limit(self, key: str) -> tuple[bool, int, int]:
        if self._redis is None:
            return await self._local_limiter.consume(
                key,
                self.rate_limit_requests,
                self.rate_limit_window_seconds,
            )

        result = await self._redis.eval(
            _REDIS_RATE_LIMIT_SCRIPT,
            1,
            key,
            self.rate_limit_window_seconds,
        )
        count = int(result[0])
        reset_after = max(1, int(result[1]))
        remaining = max(0, self.rate_limit_requests - count)
        return count <= self.rate_limit_requests, remaining, reset_after

    def _authorized(self, headers: dict[bytes, bytes]) -> bool:
        if not isinstance(self.api_key, str) or not self.api_key:
            return False
        authorization_values = [
            value
            for name, value in headers.items()
            if name == b"authorization"
        ]
        if len(authorization_values) != 1:
            return False

        scheme, separator, token = authorization_values[0].partition(b" ")
        if not separator or scheme.lower() != b"bearer" or not token.strip():
            return False
        expected_token = self.api_key.encode("utf-8")
        return secrets.compare_digest(token.strip(), expected_token)

    @staticmethod
    def _headers_dict(scope: ASGIScope) -> dict[bytes, bytes]:
        headers: dict[bytes, bytes] = {}
        for name, value in scope.get("headers", []):
            normalized_name = name.lower()
            if normalized_name == b"authorization" and normalized_name in headers:
                headers[normalized_name] += b"," + value
            else:
                headers[normalized_name] = value
        return headers

    @staticmethod
    def _client_ip(scope: ASGIScope) -> str:
        client = scope.get("client")
        if isinstance(client, (tuple, list)) and client:
            return str(client[0])
        return "unknown"

    @staticmethod
    def _rate_limit_key(prefix: str, client_ip: str) -> str:
        client_digest = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()
        return f"{prefix}:{client_digest}"

    @staticmethod
    async def _send_error(
        send: ASGISend,
        *,
        status: int,
        detail: str,
        request_id: str,
        headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        body = json.dumps(
            {"detail": detail, "request_id": request_id}
        ).encode("utf-8")
        response_headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"x-request-id", request_id.encode("ascii")),
        ]
        response_headers.extend(headers or [])
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": response_headers,
            }
        )
        await send({"type": "http.response.body", "body": body})