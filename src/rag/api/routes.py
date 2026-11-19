from __future__ import annotations

import ipaddress
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from requests import RequestException, Timeout
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from langchain_core.documents import Document

from rag.api.middleware import AuthRateLimitRequestIDMiddleware
from rag.generation.agent import GENERATE_ANSWER
from rag.guardrails.guards import (
    MAX_QUERY_LENGTH,
    greeting_response,
    guardrail_response,
    validate_query,
)
from rag.ingestion.chunking.chunker import CHUNKER
from rag.ingestion.embedding.embedd import VECTORDB
from rag.ingestion.loader.smart_processor import SMART_PARSER
from rag.retrieval.reranking import RERANK_RESULTS
from rag.retrieval.retrieval import HYBRID_SEARCH, engine
from rag.util.config import get_settings
from rag.util.logging import LoggingContext, get_logger
from rag.util.setting import CHUNKING

logger = get_logger(__name__)
settings = get_settings()

MAX_API_TOP_K = 20
MAX_TEXT_INGEST_CHARS = 200_000
UPLOAD_READ_CHUNK_BYTES = 64 * 1024
URL_CONNECT_TIMEOUT_SECONDS = 3
URL_READ_TIMEOUT_SECONDS = 10


class TextIngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_INGEST_CHARS)
    source: str | None = Field(default=None, max_length=2048)


class URLIngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int | None = Field(default=None, ge=1, le=MAX_API_TOP_K)


class IngestResponse(BaseModel):
    document_count: int
    ids: list[str]
    request_id: str


class SearchResult(BaseModel):
    id: str | None
    source: str | None
    content: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    request_id: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult]
    request_id: str


def _request_id(request: Request) -> str:
    state = request.scope.get("state", {})
    request_id = state.get("request_id") if isinstance(state, dict) else None
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("Request ID middleware did not assign a request ID")
    return request_id


def _effective_top_k(requested: int | None) -> int:
    top_k = min(settings.top_k, MAX_API_TOP_K) if requested is None else requested
    if not 1 <= top_k <= MAX_API_TOP_K:
        raise HTTPException(
            status_code=422,
            detail=f"top_k must be between 1 and {MAX_API_TOP_K}",
        )
    return top_k


def _validated_query(query: str) -> str:
    try:
        return validate_query(query)
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error


def _result_payload(document: Document) -> SearchResult:
    source = document.metadata.get("source") or document.metadata.get("title")
    return SearchResult(
        id=str(document.id) if document.id is not None else None,
        source=str(source) if source is not None else None,
        content=document.page_content,
    )


def _retrieve_and_rerank(query: str, top_k: int) -> list[Document]:
    candidate_k = min(top_k * 4, MAX_API_TOP_K * 4)
    candidates = HYBRID_SEARCH(query, top_k=candidate_k)
    return RERANK_RESULTS(query, candidates, top_k=top_k)


def _chunk_and_store(documents: list[Document]) -> list[str]:
    chunks: list[Document] = []
    for document in documents:
        if not document.page_content.strip():
            continue

        chunk_texts = CHUNKER(
            CHUNKING(
                docs=document.page_content,
                chunking_size=settings.chunk_size,
                chunking_overlap=settings.chunk_overlap,
            )
        )
        for index, chunk_text in enumerate(chunk_texts):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            chunks.append(Document(page_content=chunk_text, metadata=metadata))

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No text content was found to ingest",
        )
    return VECTORDB(chunks)


def _safe_source_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid source URL") from error

    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise HTTPException(
            status_code=400,
            detail="URL ingestion only accepts HTTPS URLs on the default port",
        )

    try:
        hostname = hostname.encode("idna").decode("ascii").rstrip(".").lower()
    except UnicodeError as error:
        raise HTTPException(status_code=400, detail="Invalid URL hostname") from error

    allowed_hosts = {
        allowed.encode("idna").decode("ascii").rstrip(".").lower()
        for allowed in settings.ingest_allowed_hosts
    }
    if not allowed_hosts:
        raise HTTPException(
            status_code=503,
            detail="URL ingestion is disabled; configure RAG_INGEST_ALLOWED_HOSTS",
        )
    if hostname not in allowed_hosts:
        raise HTTPException(status_code=403, detail="URL host is not allow-listed")

    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise HTTPException(status_code=403, detail="IP-literal URLs are not allowed")

    try:
        addresses = socket.getaddrinfo(
            hostname,
            port or 443,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as error:
        raise HTTPException(status_code=502, detail="Could not resolve URL host") from error

    resolved_addresses = {entry[4][0] for entry in addresses}
    if not resolved_addresses or any(
        not ipaddress.ip_address(address).is_global
        for address in resolved_addresses
    ):
        raise HTTPException(
            status_code=403,
            detail="URL host must resolve only to public IP addresses",
        )
    return url


def _load_url_document(url: str) -> Document:
    safe_url = _safe_source_url(url)
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    session = requests.Session()
    session.trust_env = False

    try:
        with session.get(
            safe_url,
            headers={"User-Agent": "RAGDocumentIngestion/1.0"},
            timeout=(URL_CONNECT_TIMEOUT_SECONDS, URL_READ_TIMEOUT_SECONDS),
            allow_redirects=False,
            stream=True,
        ) as response:
            if 300 <= response.status_code < 400:
                raise HTTPException(
                    status_code=400,
                    detail="URL redirects are not accepted",
                )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                raise HTTPException(
                    status_code=415,
                    detail="URL ingestion only accepts HTML content",
                )

            content = bytearray()
            for block in response.iter_content(UPLOAD_READ_CHUNK_BYTES):
                if not block:
                    continue
                if len(content) + len(block) > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Remote document exceeds the configured size limit",
                    )
                content.extend(block)
    except Timeout as error:
        raise HTTPException(status_code=504, detail="URL fetch timed out") from error
    except RequestException as error:
        raise HTTPException(status_code=502, detail="Could not fetch URL") from error
    finally:
        session.close()

    soup = BeautifulSoup(content, "html.parser")
    main_content = soup.find(["article", "main"])
    html_content = main_content if main_content is not None else soup
    text_content = html_content.get_text("\n", strip=True)
    if not text_content.strip():
        raise HTTPException(
            status_code=422,
            detail="Remote HTML contained no extractable text",
        )
    return Document(page_content=text_content, metadata={"source": safe_url})


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe; does not depend on external services."""
    return {"status": "ok"}


@router.get("/health/ready")
def readiness() -> dict[str, str]:
    """Readiness probe for PostgreSQL, pgvector, and the ParadeDB BM25 index."""
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT
                      to_regclass('langchain_pg_embedding') IS NOT NULL AS has_embeddings,
                      EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                      ) AS has_vector,
                      EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'pg_search'
                      )
                      AND to_regclass('ix_langchain_pg_embedding_bm25') IS NOT NULL
                        AS has_bm25
                    """
                )
            ).mappings().one()
    except SQLAlchemyError as error:
        logger.exception("Readiness check could not reach PostgreSQL")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PostgreSQL is unavailable",
        ) from error

    if not row["has_embeddings"] or not row["has_vector"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PostgreSQL pgvector storage is not initialized",
        )
    return {
        "status": "ready",
        "search_mode": "hybrid_bm25" if row["has_bm25"] else "hybrid_postgres_fts",
    }


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
@router.post(
    "/ingest/text",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
def ingest_text(payload: TextIngestRequest, request: Request) -> IngestResponse:
    """Chunk and store supplied UTF-8 text."""
    source = payload.source or "api:text"
    ids = _chunk_and_store(
        [Document(page_content=payload.text, metadata={"source": source})]
    )
    return IngestResponse(
        document_count=len(ids),
        ids=ids,
        request_id=_request_id(request),
    )


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_file(
    request: Request,
    file: UploadFile = File(...),
) -> IngestResponse:
    """Parse and ingest one PDF, text, Markdown, or HTML upload."""
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.supported_extensions:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type",
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    temporary_path: Path | None = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix="rag-upload-",
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while block := file.file.read(UPLOAD_READ_CHUNK_BYTES):
                total_bytes += len(block)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the configured size limit",
                    )
                temporary_file.write(block)

        if total_bytes == 0:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")
        try:
            documents = SMART_PARSER(str(temporary_path))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        file.file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    for document in documents:
        document.metadata["source"] = filename
    ids = _chunk_and_store(documents)
    return IngestResponse(
        document_count=len(ids),
        ids=ids,
        request_id=_request_id(request),
    )


@router.post(
    "/ingest/url",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_url(payload: URLIngestRequest, request: Request) -> IngestResponse:
    """Fetch allow-listed public HTML and ingest its extracted text."""
    document = _load_url_document(payload.url)
    ids = _chunk_and_store([document])
    return IngestResponse(
        document_count=len(ids),
        ids=ids,
        request_id=_request_id(request),
    )


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Run hybrid retrieval followed by CrossEncoder reranking."""
    query = _validated_query(payload.query)
    if guardrail_response(query) is not None:
        raise HTTPException(status_code=400, detail="Search query rejected by guardrails")
    top_k = _effective_top_k(payload.top_k)
    if greeting_response(query) is not None:
        return SearchResponse(
            query=query,
            results=[],
            request_id=_request_id(request),
        )
    documents = _retrieve_and_rerank(query, top_k)
    return SearchResponse(
        query=query,
        results=[_result_payload(document) for document in documents],
        request_id=_request_id(request),
    )


@router.post("/query", response_model=QueryResponse)
def query(payload: SearchRequest, request: Request) -> QueryResponse:
    """Retrieve, rerank, and generate an answer from the retrieved sources."""
    normalized_query = _validated_query(payload.query)
    request_id = _request_id(request)

    blocked_response = guardrail_response(normalized_query)
    if blocked_response is not None:
        return QueryResponse(
            query=normalized_query,
            answer=blocked_response,
            sources=[],
            request_id=request_id,
        )

    canned_response = greeting_response(normalized_query)
    if canned_response is not None:
        return QueryResponse(
            query=normalized_query,
            answer=canned_response,
            sources=[],
            request_id=request_id,
        )

    documents = _retrieve_and_rerank(
        normalized_query,
        _effective_top_k(payload.top_k),
    )
    answer = GENERATE_ANSWER(normalized_query, documents)
    return QueryResponse(
        query=normalized_query,
        answer=answer,
        sources=[_result_payload(document) for document in documents],
        request_id=request_id,
    )


async def unhandled_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    state = request.scope.get("state", {})
    request_id = state.get("request_id") if isinstance(state, dict) else None
    with LoggingContext(request_id if isinstance(request_id, str) else "unknown"):
        logger.exception("Unhandled API exception", exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )


def create_app(
    *,
    api_key: str | None = None,
    redis_url: str | None = None,
) -> FastAPI:
    """Create the API application with auth, rate limiting, and request IDs."""
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )
    application.add_exception_handler(Exception, unhandled_exception_handler)
    application.include_router(router)
    application.add_middleware(
        AuthRateLimitRequestIDMiddleware,
        api_key=api_key,
        redis_url=redis_url,
        public_paths=("/health", "/health/ready"),
    )
    return application


app = create_app()


def run() -> None:
    """Run the API using configured host and port."""
    import uvicorn

    uvicorn.run(
        "rag.api.routes:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )