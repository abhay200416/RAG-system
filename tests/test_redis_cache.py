import pytest

from rag.cache.cashe import RedisCache


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.last_key = None
        self.last_ttl = None

    def get(self, key):
        self.last_key = key
        return self.values.get(key)

    def set(self, key, value, *, ex):
        self.last_key = key
        self.last_ttl = ex
        self.values[key] = value
        return True

    def delete(self, key):
        self.last_key = key
        return int(self.values.pop(key, None) is not None)

    def close(self):
        pass


def test_redis_cache_hashes_payload_and_uses_ttl():
    client = FakeRedis()
    cache = RedisCache(
        "redis://localhost:6379/0",
        key_prefix="test",
        default_ttl_seconds=90,
        client=client,
    )
    payload = {"query": "private query", "sources": ["document"]}

    assert cache.get("answer", payload) is None
    cache.set("answer", payload, "cached answer")

    assert cache.get("answer", payload) == "cached answer"
    assert client.last_ttl == 90
    assert "private query" not in client.last_key
    assert cache.delete("answer", payload)
    assert cache.get("answer", payload) is None


@pytest.mark.parametrize("ttl", [0, -1, True])
def test_redis_cache_rejects_invalid_ttl(ttl):
    with pytest.raises(ValueError, match="positive integer"):
        RedisCache("redis://localhost:6379/0", default_ttl_seconds=ttl)
