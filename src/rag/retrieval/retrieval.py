import json
import re

from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from rag.ingestion.embedding.embedd import (
    VECTOR_STORE_COLLECTION,
    VECTOR_STORE_CONNECTION,
    embeddings,
)
from rag.util.config import get_settings
from rag.util.logging import LoggingContext, get_logger

logger = get_logger(__name__)
settings = get_settings()

# 1. Reuse a single SQLAlchemy engine with connection pooling
engine = create_engine(VECTOR_STORE_CONNECTION, pool_pre_ping=True)

# 2. Include collection_id in the ParadeDB index so collection filtering happens inside the BM25 scan
_CREATE_BM25_INDEX = text(
    """
    CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_bm25
    ON langchain_pg_embedding
    USING paradedb (id, collection_id, document)
    """
)

_SEARCH_CAPABILITIES = text(
    """
    SELECT
      EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'pg_search'
      )
      AND to_regclass('ix_langchain_pg_embedding_bm25') IS NOT NULL
      AS has_bm25
    """
)

# 3. Return embedding.id so RRF can deduplicate by primary key instead of raw text/metadata
_BM25_SEARCH = text(
    """
    SELECT embedding.id, embedding.document, embedding.cmetadata
    FROM langchain_pg_embedding AS embedding
    JOIN langchain_pg_collection AS collection
      ON collection.uuid = embedding.collection_id
    WHERE collection.name = :collection_name
      AND embedding.document IS NOT NULL
      AND embedding.document ||| CAST(:query AS text)
    ORDER BY pdb.score(embedding.id) DESC, embedding.id
    LIMIT :limit
    """
)

_POSTGRES_FTS_SEARCH = text(
    """
    WITH query AS (
      SELECT plainto_tsquery('simple', :query) AS terms
    )
    SELECT embedding.id, embedding.document, embedding.cmetadata
    FROM langchain_pg_embedding AS embedding
    CROSS JOIN query
    WHERE embedding.document IS NOT NULL
      AND to_tsvector('simple', embedding.document) @@ query.terms
    ORDER BY ts_rank_cd(
      to_tsvector('simple', embedding.document),
      query.terms
    ) DESC, embedding.id
    LIMIT :limit
    """
)


def _sanitize_bm25_query(query: str) -> str:
    """Strip ParadeDB/Tantivy special syntax characters so user input cannot crash the parser."""
    cleaned = re.sub(r'[+\-=&|><!(){}\[\]^"~*?:\\/]', " ", query)
    return " ".join(cleaned.split())


def _document_key(document: Document) -> str:
    """Return the document's primary key ID if available, falling back to content+metadata."""
    if document.id:
        return str(document.id)
    metadata = json.dumps(document.metadata, sort_keys=True, default=str)
    return f"{document.page_content}:{metadata}"


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    limit: int,
    rank_constant: int,
) -> list[Document]:
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranked_documents in ranked_lists:
        for rank, document in enumerate(ranked_documents, start=1):
            key = _document_key(document)
            documents.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)

    ordered_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [documents[key] for key in ordered_keys[:limit]]


def _get_vector_store() -> PGVector:
    """Instantiate PGVector using the shared connection pool."""
    return PGVector(
        embeddings=embeddings,
        collection_name=VECTOR_STORE_COLLECTION,
        connection=engine,
        use_jsonb=True,
    )


def CREATE_BM25_INDEX() -> None:
    """Create the ParadeDB BM25 index; call once after pg_search is installed."""
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search"))

    # Ensure PGVector tables exist before creating the index
    _get_vector_store()

    with engine.begin() as connection:
        connection.execute(_CREATE_BM25_INDEX)
    logger.info("Created BM25 index for the PGVector embedding table")


def HYBRID_SEARCH(
    query: str,
    top_k: int | None = None,
    *,
    rank_constant: int = 60,
) -> list[Document]:
    """Retrieve documents using vector similarity and ParadeDB BM25 search."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty")

    result_limit = settings.top_k if top_k is None else top_k
    if (
        isinstance(result_limit, bool)
        or not isinstance(result_limit, int)
        or result_limit < 1
    ):
        raise ValueError("top_k must be a positive integer")
    if (
        isinstance(rank_constant, bool)
        or not isinstance(rank_constant, int)
        or rank_constant < 1
    ):
        raise ValueError("rank_constant must be a positive integer")

    search_query = query.strip()
    bm25_query = _sanitize_bm25_query(search_query)
    candidate_limit = result_limit * 2

    with LoggingContext("hybrid document search"):
        vector_store = _get_vector_store()
        vector_results = vector_store.similarity_search_with_score(
            search_query,
            k=candidate_limit,
        )

        lexical_documents: list[Document] = []
        if bm25_query:
            with engine.connect() as connection:
                has_bm25 = bool(
                    connection.execute(_SEARCH_CAPABILITIES).scalar_one()
                )
                if has_bm25:
                    lexical_rows = (
                        connection.execute(
                            _BM25_SEARCH,
                            {
                                "query": bm25_query,
                                "collection_name": VECTOR_STORE_COLLECTION,
                                "limit": candidate_limit,
                            },
                        )
                        .mappings()
                        .all()
                    )
                else:
                    logger.warning(
                        "ParadeDB BM25 is unavailable; using PostgreSQL full-text "
                        "search for this hybrid query"
                    )
                    lexical_rows = (
                        connection.execute(
                            _POSTGRES_FTS_SEARCH,
                            {
                                "query": search_query,
                                "limit": candidate_limit,
                            },
                        )
                        .mappings()
                        .all()
                    )

            lexical_documents = [
                Document(
                    id=str(row["id"]),
                    page_content=row["document"],
                    metadata=row["cmetadata"] or {},
                )
                for row in lexical_rows
            ]

        results = _reciprocal_rank_fusion(
            [
                [document for document, _score in vector_results],
                lexical_documents,
            ],
            result_limit,
            rank_constant,
        )
        logger.info(
            "Hybrid search returned %d documents for query",
            len(results),
        )
        return results
