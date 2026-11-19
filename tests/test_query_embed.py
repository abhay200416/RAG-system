import pytest

from rag.generation import query_embed


def test_query_embed_uses_shared_embedding_model_and_strips_query(monkeypatch):
    received_queries = []

    class FakeEmbeddings:
        def embed_query(self, query):
            received_queries.append(query)
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(query_embed, "embeddings", FakeEmbeddings())

    vector = query_embed.QUERY_EMBED("  search text  ")

    assert vector == [0.1, 0.2, 0.3]
    assert received_queries == ["search text"]


@pytest.mark.parametrize("query", ["", "  ", None])
def test_query_embed_rejects_empty_or_non_string_query(monkeypatch, query):
    class FakeEmbeddings:
        def embed_query(self, _query):
            raise AssertionError("Embedding should not run for invalid queries")

    monkeypatch.setattr(query_embed, "embeddings", FakeEmbeddings())

    with pytest.raises(ValueError, match="cannot be empty"):
        query_embed.QUERY_EMBED(query)


def test_query_embed_rejects_non_finite_model_output(monkeypatch):
    class FakeEmbeddings:
        def embed_query(self, _query):
            return [0.1, float("nan")]

    monkeypatch.setattr(query_embed, "embeddings", FakeEmbeddings())

    with pytest.raises(RuntimeError, match="non-finite"):
        query_embed.QUERY_EMBED("search text")
