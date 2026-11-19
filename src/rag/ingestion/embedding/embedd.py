
from collections.abc import Iterable

from google.genai.types import EmbedContentConfig
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from rag.util.config import get_settings
from rag.util.logging import LoggingContext, get_logger

logger = get_logger(__name__)

settings = get_settings()
VECTOR_STORE_COLLECTION = settings.collection_name
VECTOR_STORE_CONNECTION = settings.database_url


class RAGGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """Use Gemini Embedding 2's documented prompt format without task_type."""

    def _uses_embedding_2(self) -> bool:
        return self.model.removeprefix("models/") == "gemini-embedding-2"

    def _embed_embedding_2(
        self,
        contents: str,
        output_dimensionality: int | None = None,
    ) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=contents,
            config=EmbedContentConfig(
                output_dimensionality=(
                    output_dimensionality or self.output_dimensionality
                ),
            ),
        )
        vectors = response.embeddings
        if not vectors:
            raise RuntimeError("Gemini Embedding 2 returned no embeddings")
        return [list(vector.values) for vector in vectors]

    def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 100,
        task_type: str | None = None,
        titles: list[str] | None = None,
        output_dimensionality: int | None = None,
    ) -> list[list[float]]:
        if not self._uses_embedding_2():
            return super().embed_documents(
                texts,
                batch_size=batch_size,
                task_type=task_type,
                titles=titles,
                output_dimensionality=output_dimensionality,
            )
        if task_type is not None or self.task_type is not None:
            raise ValueError("gemini-embedding-2 does not support task_type")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if titles is not None and len(titles) != len(texts):
            raise ValueError("titles must have one entry per document")
        if not texts:
            return []

        dimensionality = output_dimensionality or self.output_dimensionality
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            title_batch = titles[start : start + batch_size] if titles else None
            for content, title in zip(
                batch,
                title_batch or ["none"] * len(batch),
                strict=True,
            ):
                # Gemini Embedding 2 aggregates multiple inputs into one vector.
                response_vectors = self._embed_embedding_2(
                    f"title: {title or 'none'} | text: {content}",
                    output_dimensionality=dimensionality,
                )
                if len(response_vectors) != 1:
                    raise RuntimeError(
                        "Gemini Embedding 2 must return exactly one vector "
                        "for each document"
                    )
                vectors.append(response_vectors[0])
        return vectors

    def embed_query(
        self,
        text: str,
        *,
        task_type: str | None = None,
        title: str | None = None,
        output_dimensionality: int | None = None,
    ) -> list[float]:
        if not self._uses_embedding_2():
            return super().embed_query(
                text,
                task_type=task_type,
                title=title,
                output_dimensionality=output_dimensionality,
            )
        if task_type is not None or self.task_type is not None:
            raise ValueError("gemini-embedding-2 does not support task_type")
        query = f"task: question answering | query: {text}"
        vectors = self._embed_embedding_2(
            query,
            output_dimensionality=(
                output_dimensionality or self.output_dimensionality
            ),
        )
        if len(vectors) != 1:
            raise RuntimeError("Gemini Embedding 2 returned an invalid query response")
        return vectors[0]


embeddings = RAGGoogleGenerativeAIEmbeddings(
    model=settings.default_embedding_model,
    google_api_key=settings.google_api_key,
)


def _to_documents(chunk_docs: str | Document | Iterable[str | Document]) -> list[Document]:
    """Normalize a string or chunk collection into LangChain documents."""
    if isinstance(chunk_docs, Document):
        documents = [chunk_docs]
    elif isinstance(chunk_docs, str):
        documents = [Document(page_content=chunk_docs)]
    elif isinstance(chunk_docs, Iterable):
        documents = [
            chunk if isinstance(chunk, Document) else Document(page_content=chunk)
            for chunk in chunk_docs
        ]
    else:
        raise TypeError("Chunk documents must be a string, Document, or iterable of chunks")

    if not documents:
        raise ValueError("Document content cannot be empty")

    normalized: list[Document] = []
    for document in documents:
        if not isinstance(document.page_content, str) or not document.page_content.strip():
            raise ValueError("Document content cannot be empty")
        normalized.append(document)
    return normalized


def VECTORDB(
    chunk_docs: str | Document | Iterable[str | Document],
) -> list[str]:
    """Embed document chunks and store them in PGVector."""

    with LoggingContext("converting chunk into embedding and storing in pgvector"):
        try:
            documents = _to_documents(chunk_docs)

            vector_store = PGVector(
                embeddings=embeddings,
                collection_name=VECTOR_STORE_COLLECTION,
                connection=VECTOR_STORE_CONNECTION,
                use_jsonb=True,
            )

            ids = vector_store.add_documents(documents)
            logger.info("Successfully stored %d document chunks in PGVector", len(documents))
            return ids
        except (TypeError, ValueError) as error:
            logger.error("Error embedding documents: %s", error)
            raise

