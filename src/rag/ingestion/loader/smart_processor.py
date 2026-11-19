from rag.util.logging import get_logger, LoggingContext
from pathlib import Path
from urllib.parse import urlparse

from .html import HTML_LOADER
from .pdf import PDF_LOADER
from .text import TEXT_LOADER

logger = get_logger(__name__)


def SMART_PARSER(path: str):
    """Parse a local document or remote HTML resource according to its type."""

    with LoggingContext("smart parser"):
        if not isinstance(path, str) or not path.strip():
            raise ValueError("A file path or URL must be provided")

        source = path.strip()
        parsed = urlparse(source)
        is_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)

        if not is_url:
            file_path = Path(source)
            if not file_path.is_file():
                logger.error("Path is missing or is not a file: %s", source)
                raise FileNotFoundError(f"File not found: {source}")

        suffix = Path(parsed.path if is_url else source).suffix.lower()

        if is_url:
            if suffix not in {"", ".html", ".htm"}:
                raise ValueError(
                    f"Unsupported URL type: {suffix or '(no extension)'}. "
                    "Remote parsing supports HTML URLs."
                )
            full_doc = HTML_LOADER(source)
        elif suffix == ".pdf":
            full_doc = PDF_LOADER(source)
        elif suffix in {".txt", ".text", ".md", ".markdown"}:
            full_doc = TEXT_LOADER(source)
        elif suffix in {".html", ".htm"}:
            full_doc = HTML_LOADER(source)
        else:
            logger.error("Unsupported file type: %s", suffix)
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                "Supported types: .pdf, .txt, .text, .md, .markdown, .html, .htm"
            )

        logger.info("Successfully parsed source: %s", source)
        return full_doc