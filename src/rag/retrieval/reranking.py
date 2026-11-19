from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from rag.util.logging import LoggingContext, get_logger

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


logger = get_logger(__name__)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _load_reranker(model_name: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def RERANK_RESULTS(
    query: str,
    retrieved_docs: Iterable[Document],
    top_k: int | None = None,
    *,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[Document]:
    
    """Reorder retrieved documents by CrossEncoder query-document relevance."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Reranking query cannot be empty")
    if not isinstance(retrieved_docs, Iterable) or isinstance(
        retrieved_docs, (str, bytes)
    ):
        raise TypeError("retrieved_docs must be an iterable of Document objects")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name cannot be empty")

    documents = list(retrieved_docs)
    if any(not isinstance(document, Document) for document in documents):
        raise TypeError("retrieved_docs must contain only Document objects")
    if not documents:
        return []

    result_limit = len(documents) if top_k is None else top_k
    pairs = [(query.strip(), document.page_content) for document in documents]

    with LoggingContext("reranking retrieved documents"):
        scores = _load_reranker(model_name).predict(pairs)
        if len(scores) != len(documents):
            raise RuntimeError(
                "Reranker returned a different number of scores than input documents"
            )

        numeric_scores = [float(score) for score in scores]
        if not all(math.isfinite(score) for score in numeric_scores):
            raise ValueError("Reranker returned a non-finite relevance score")

        ranked_indexes = sorted(
            range(len(documents)),
            key=lambda index: (-numeric_scores[index], index),
        )
        reranked_docs = [documents[index] for index in ranked_indexes[:result_limit]]
        logger.info("Reranked %d documents", len(reranked_docs))
        return reranked_docs