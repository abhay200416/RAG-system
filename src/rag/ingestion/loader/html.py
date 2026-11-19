import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from rag.util.logging import get_logger, LoggingContext
from pathlib import Path
import re
from urllib.parse import urlparse


logger = get_logger(__name__)

def HTML_LOADER(path: str):
    """Load HTML from a local file or HTTP(S) URL."""

    with LoggingContext("Html loader"):
        parsed = urlparse(path)
        is_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)

        if is_url:
            strainer = bs4.SoupStrainer(name=["article", "main"])
            docs = WebBaseLoader(
                web_paths=(path,),
                bs_kwargs={"parse_only": strainer},
                bs_get_text_kwargs={"separator": "\n", "strip": True},
            ).load()
        else:
            file_path = Path(path)
            if not file_path.is_file():
                raise FileNotFoundError(f"HTML file not found: {path}")
            soup = bs4.BeautifulSoup(file_path.read_text(encoding="utf-8"), "html.parser")
            content = soup.find(["article", "main"])
            content = content if content is not None else soup
            docs = [Document(page_content=content.get_text("\n", strip=True), metadata={"source": str(file_path)})]

        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError("Page loaded without matching HTML content")

        for doc in docs:
            doc.page_content = re.sub(r"\n{3,}", "\n\n", doc.page_content)

        logger.info("HTML has loaded successfully")
        return docs



