from langchain_community.document_loaders import PyMuPDFLoader
from rag.util.logging import get_logger, LoggingContext

logger = get_logger(__name__)

def PDF_LOADER(path: str):
    """Load a local PDF file."""

    with LoggingContext("pdf loading"):
        docs = PyMuPDFLoader(path).load()
        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError("PDF loaded without extractable text")

        logger.info("PDF has loaded successfully")
        return docs
