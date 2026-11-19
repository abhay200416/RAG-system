from types import SimpleNamespace

from rag.ingestion.embedding.embedd import RAGGoogleGenerativeAIEmbeddings


class _FakeModelsClient:
    def __init__(self) -> None:
        self.calls = []

    def embed_content(self, *, model, contents, config):
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float(len(self.calls) - 1), 0.5])
            ]
        )


def _embeddings() -> tuple[RAGGoogleGenerativeAIEmbeddings, _FakeModelsClient]:
    embeddings = RAGGoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key="not-a-real-key",
    )
    models_client = _FakeModelsClient()
    object.__setattr__(
        embeddings,
        "client",
        SimpleNamespace(models=models_client),
    )
    return embeddings, models_client


def test_embedding_2_documents_omit_unsupported_task_type() -> None:
    embeddings, client = _embeddings()

    vectors = embeddings.embed_documents(["first", "second"])

    assert vectors == [[0.0, 0.5], [1.0, 0.5]]
    assert len(client.calls) == 2
    assert [
        call["contents"]
        for call in client.calls
    ] == [
        "title: none | text: first",
        "title: none | text: second",
    ]
    assert all(call["model"] == "gemini-embedding-2" for call in client.calls)
    assert all(call["config"].task_type is None for call in client.calls)


def test_embedding_2_query_uses_question_answering_prefix() -> None:
    embeddings, client = _embeddings()

    vector = embeddings.embed_query("What does the document say?")

    assert vector == [0.0, 0.5]
    call = client.calls[0]
    assert call["contents"] == (
        "task: question answering | query: What does the document say?"
    )
    assert call["config"].task_type is None
