from langchain_community.document_loaders import TextLoader
from rag.util.logging import get_logger, LoggingContext

logger = get_logger(__name__)

def TEXT_LOADER(path: str):
    """Load a local text or Markdown file."""

    with LoggingContext("text loading"):
        docs = TextLoader(path, encoding="utf-8").load()
        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError("Text file loaded without content")

        logger.info("Text file has loaded successfully")
        return docs