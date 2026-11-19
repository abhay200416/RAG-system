import math

from rag.ingestion.embedding.embedd import embeddings
from rag.util.logging import LoggingContext, get_logger

logger = get_logger(__name__)


def QUERY_EMBED(query: str) -> list[float]:
    """Convert a non-empty search query into the configured embedding vector."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query content cannot be empty")

    with LoggingContext("embedding search query"):
        vector = embeddings.embed_query(query.strip())
        if not vector:
            raise RuntimeError("Embedding provider returned an empty query vector")
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError("Embedding provider returned a non-finite query vector")

        logger.info("Embedded query into a vector with %d dimensions", len(vector))
        return vector