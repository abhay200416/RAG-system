from rag.util.logging import get_logger, LoggingContext
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.util.setting import CHUNKING

logger = get_logger(__name__)


def CHUNKER(doc: CHUNKING) -> list[str]:
    """Split document text into chunks for retrieval."""

    with LoggingContext("chunking for retrieval"):
        try:
            if not doc.docs.strip():
                raise ValueError("Document content cannot be empty")
            if doc.chunking_overlap >= doc.chunking_size:
                raise ValueError("Chunk overlap must be smaller than chunk size")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=doc.chunking_size,
                chunk_overlap=doc.chunking_overlap,
            )
            chunks = splitter.split_text(doc.docs)
            logger.info("Successfully split document into %d chunks", len(chunks))
            return chunks
        except (TypeError, ValueError) as error:
            logger.error("Error chunking document: %s", error)
            raise
            