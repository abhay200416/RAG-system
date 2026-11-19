import pytest
from langchain_core.documents import Document

from rag.retrieval import reranking


def test_rerank_results_orders_documents_and_applies_limit(monkeypatch):
    documents = [
        Document(page_content="lower relevance"),
        Document(page_content="highest relevance"),
        Document(page_content="middle relevance"),
    ]
    captured_pairs = []

    class FakeReranker:
        def predict(self, pairs):
            captured_pairs.extend(pairs)
            return [0.1, 0.9, 0.5]

    monkeypatch.setattr(reranking, "_load_reranker", lambda _model: FakeReranker())

    results = reranking.RERANK_RESULTS(" question ", documents, top_k=2)

    assert results == [documents[1], documents[2]]
    assert captured_pairs == [
        ("question", "lower relevance"),
        ("question", "highest relevance"),
        ("question", "middle relevance"),
    ]


def test_rerank_results_empty_input_does_not_load_model(monkeypatch):
    def fail_if_loaded(_model):
        raise AssertionError("The reranker should not load for empty input")

    monkeypatch.setattr(reranking, "_load_reranker", fail_if_loaded)

    assert reranking.RERANK_RESULTS("question", []) == []


@pytest.mark.parametrize(
    ("query", "documents", "top_k", "error"),
    [
        ("  ", [], None, ValueError),
        ("question", [Document(page_content="doc")], 0, ValueError),
        ("question", ["not a document"], None, TypeError),
    ],
)
def test_rerank_results_validates_inputs(query, documents, top_k, error):
    with pytest.raises(error):
        reranking.RERANK_RESULTS(query, documents, top_k=top_k)
