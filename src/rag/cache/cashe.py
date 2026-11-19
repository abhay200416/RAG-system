import hashlib
import json

from redis import Redis


class RedisCache:
    """Small synchronous Redis cache for JSON-keyed string values."""

    def __init__(
        self,
        url: str,
        *,
        key_prefix: str = "rag",
        default_ttl_seconds: int = 3600,
        client: Redis | None = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Redis URL cannot be empty")
        if not isinstance(key_prefix, str) or not key_prefix.strip():
            raise ValueError("Redis key prefix cannot be empty")
        self._validate_ttl(default_ttl_seconds)

        self.default_ttl_seconds = default_ttl_seconds
        self.key_prefix = key_prefix.strip()
        self._client = client or Redis.from_url(
            url.strip(),
            decode_responses=True,
        )

    def get(self, namespace: str, payload: object) -> str | None:
        """Return a cached string, or None when the key is absent."""
        key = self._make_key(namespace, payload)
        value = self._client.get(key)
        if value is not None and not isinstance(value, str):
            raise TypeError("Redis client must return decoded string values")
        return value

    def set(
        self,
        namespace: str,
        payload: object,
        value: str,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store a string value with an expiration time."""
        if not isinstance(value, str):
            raise TypeError("Redis cache values must be strings")
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        self._validate_ttl(ttl)
        self._client.set(
            self._make_key(namespace, payload),
            value,
            ex=ttl,
        )

    def delete(self, namespace: str, payload: object) -> bool:
        """Delete a cached value and report whether a key existed."""
        return bool(self._client.delete(self._make_key(namespace, payload)))

    def close(self) -> None:
        """Close Redis connections owned by this client."""
        self._client.close()

    def _make_key(self, namespace: str, payload: object) -> str:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("Cache namespace cannot be empty")
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:{namespace.strip()}:{digest}"

    @staticmethod
    def _validate_ttl(ttl_seconds: int) -> None:
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or ttl_seconds < 1
        ):
            raise ValueError("Cache TTL must be a positive integer")