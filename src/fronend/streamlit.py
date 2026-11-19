from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

import requests
import streamlit as st

from rag.util.config import get_settings

settings = get_settings()


class APIError(RuntimeError):
    def __init__(self, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


def _validate_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
    except ValueError as error:
        raise APIError("Enter a valid API base URL.") from error
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise APIError("API base URL must start with http:// or https://.")
    if parsed.username is not None or parsed.password is not None:
        raise APIError("Do not put credentials in the API URL; use the API key field.")
    return normalized


def request_api(
    method: str,
    path: str,
    *,
    base_url: str,
    api_key: str = "",
    json_body: dict[str, Any] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    timeout: tuple[float, float] = (5, 120),
) -> dict[str, Any]:
    """Call one RAG API endpoint and surface HTTP/transport failures."""
    normalized_base_url = _validate_base_url(base_url)
    headers = {"Accept": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    try:
        response = requests.request(
            method,
            f"{normalized_base_url}/{path.lstrip('/')}",
            headers=headers,
            json=json_body,
            files=files,
            timeout=timeout,
        )
    except requests.Timeout as error:
        raise APIError("The RAG API request timed out.") from error
    except requests.ConnectionError as error:
        raise APIError(
            f"The RAG API is not reachable at {normalized_base_url}. "
            "Start it in another terminal with `uv run rag-api`, then confirm "
            "the API base URL here."
        ) from error
    except requests.RequestException as error:
        raise APIError(f"Could not reach the RAG API: {error}") from error

    request_id = response.headers.get("x-request-id")
    try:
        payload = response.json()
    except ValueError as error:
        if not response.ok:
            detail = response.text[:1000] or response.reason
            raise APIError(
                f"API returned HTTP {response.status_code}: {detail}",
                request_id,
            ) from error
        raise APIError(
            "The RAG API returned a response that was not valid JSON.",
            request_id,
        ) from error

    if not isinstance(payload, dict):
        raise APIError(
            "The RAG API returned an unexpected response format.",
            request_id,
        )
    if not request_id and isinstance(payload.get("request_id"), str):
        request_id = payload["request_id"]
    if not response.ok:
        detail = payload.get("detail", response.reason)
        if not isinstance(detail, str):
            detail = json.dumps(detail, ensure_ascii=False)
        if (
            response.status_code == 503
            and detail == "API authentication is not configured"
        ):
            detail = (
                "The API server has no RAG_API_AUTH_KEY configured. Set a long, "
                "random key in the backend .env file, restart `uv run rag-api`, "
                "and enter the same key in the Streamlit sidebar."
            )
        raise APIError(
            f"API returned HTTP {response.status_code}: {detail}",
            request_id,
        )
    return payload


def _display_request_id(payload: dict[str, Any]) -> None:
    request_id = payload.get("request_id")
    if isinstance(request_id, str) and request_id:
        st.caption(f"Request ID: `{request_id}`")


def _render_sources(sources: Any) -> None:
    if not isinstance(sources, list) or not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                continue
            title = source.get("source") or source.get("id") or f"Source {index}"
            st.markdown(f"**{title}**")
            content = source.get("content")
            if isinstance(content, str) and content:
                st.text(content)


def _show_api_error(error: APIError) -> None:
    st.error(str(error))
    if error.request_id:
        st.caption(f"Request ID: `{error.request_id}`")


def _render_health_check(base_url: str) -> None:
    if not st.sidebar.button("Check API status", use_container_width=True):
        return

    try:
        liveness = request_api(
            "GET",
            "/health",
            base_url=base_url,
            timeout=(3, 5),
        )
        st.sidebar.success(f"API is live: {liveness.get('status', 'ok')}")
    except APIError as error:
        st.sidebar.error(f"API liveness check failed: {error}")
        return

    try:
        readiness = request_api(
            "GET",
            "/health/ready",
            base_url=base_url,
            timeout=(3, 10),
        )
    except APIError as error:
        st.sidebar.warning(f"API is live but not ready: {error}")
    else:
        st.sidebar.success(f"API is ready: {readiness.get('status', 'ready')}")


def _render_chat(base_url: str, api_key: str, top_k: int) -> None:
    st.subheader("Ask the indexed documents")
    st.caption(
        "Each prompt is answered independently from indexed documents; "
        "this chat history is displayed locally."
    )
    st.session_state.setdefault("chat_messages", [])

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_sources(message.get("sources"))
            if message.get("request_id"):
                st.caption(f"Request ID: `{message['request_id']}`")

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Searching and generating an answer..."):
            payload = request_api(
                "POST",
                "/query",
                base_url=base_url,
                api_key=api_key,
                json_body={"query": prompt, "top_k": top_k},
            )
        answer = payload.get("answer")
        if not isinstance(answer, str):
            raise APIError(
                "The query endpoint did not return an answer.",
                payload.get("request_id"),
            )
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": payload.get("sources", []),
            "request_id": payload.get("request_id"),
        }
    except APIError as error:
        assistant_message = {
            "role": "assistant",
            "content": f"Request failed: {error}",
            "sources": [],
            "request_id": error.request_id,
        }

    st.session_state.chat_messages.append(assistant_message)
    st.rerun()


def _render_search(base_url: str, api_key: str) -> None:
    st.subheader("Search documents")
    with st.form("search_form"):
        search_query = st.text_input("Search query")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Search")

    if not submitted:
        return
    if not search_query.strip():
        st.warning("Enter a search query.")
        return
    try:
        with st.spinner("Searching and reranking..."):
            payload = request_api(
                "POST",
                "/search",
                base_url=base_url,
                api_key=api_key,
                json_body={"query": search_query, "top_k": top_k},
            )
    except APIError as error:
        _show_api_error(error)
        return

    results = payload.get("results")
    if not isinstance(results, list):
        st.error("The search endpoint returned an unexpected response.")
        return
    st.write(f"Found {len(results)} result(s).")
    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue
        source = result.get("source") or result.get("id") or f"Result {index}"
        with st.expander(f"{index}. {source}", expanded=index == 1):
            st.write(result.get("content", ""))
    _display_request_id(payload)


def _render_ingestion(base_url: str, api_key: str) -> None:
    st.subheader("Add documents")
    input_type = st.radio(
        "Choose an input type",
        ("Text", "File", "URL"),
        horizontal=True,
    )

    if input_type == "Text":
        with st.form("text_ingestion_form"):
            source = st.text_input("Source label (optional)", placeholder="notes.txt")
            text_content = st.text_area(
                "Text to index",
                height=220,
                max_chars=200_000,
            )
            submitted = st.form_submit_button("Index text")
        if not submitted:
            return
        if not text_content.strip():
            st.warning("Enter some text before submitting.")
            return
        try:
            payload = request_api(
                "POST",
                "/ingest",
                base_url=base_url,
                api_key=api_key,
                json_body={
                    "text": text_content,
                    "source": source.strip() or None,
                },
            )
        except APIError as error:
            _show_api_error(error)
            return
        _show_ingestion_result(payload)
        return

    if input_type == "File":
        with st.form("file_ingestion_form"):
            uploaded_file = st.file_uploader(
                "Choose a PDF, TXT, Markdown, or HTML file",
                type=["pdf", "txt", "md", "html", "htm"],
            )
            submitted = st.form_submit_button("Upload and index")
        if not submitted:
            return
        if uploaded_file is None:
            st.warning("Choose a file before submitting.")
            return
        if uploaded_file.size > settings.max_file_size_mb * 1024 * 1024:
            st.warning(
                f"The file exceeds the configured {settings.max_file_size_mb} MB limit."
            )
            return
        try:
            payload = request_api(
                "POST",
                "/ingest/file",
                base_url=base_url,
                api_key=api_key,
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                timeout=(5, 180),
            )
        except APIError as error:
            _show_api_error(error)
            return
        _show_ingestion_result(payload)
        return

    st.info(
        "URL ingestion accepts only HTTPS pages on exact hosts configured by "
        "the API operator."
    )
    with st.form("url_ingestion_form"):
        source_url = st.text_input("HTTPS page URL")
        submitted = st.form_submit_button("Fetch and index")
    if not submitted:
        return
    if not source_url.strip():
        st.warning("Enter a URL before submitting.")
        return
    try:
        payload = request_api(
            "POST",
            "/ingest/url",
            base_url=base_url,
            api_key=api_key,
            json_body={"url": source_url},
            timeout=(5, 180),
        )
    except APIError as error:
        _show_api_error(error)
        return
    _show_ingestion_result(payload)


def _show_ingestion_result(payload: dict[str, Any]) -> None:
    count = payload.get("document_count")
    if not isinstance(count, int):
        st.error("The ingestion endpoint returned an unexpected response.")
        return
    st.success(f"Indexed {count} chunk(s).")
    _display_request_id(payload)
    ids = payload.get("ids")
    if isinstance(ids, list) and ids:
        with st.expander("Stored chunk IDs"):
            st.code("\n".join(str(item) for item in ids))


def main() -> None:
    st.set_page_config(
        page_title="RAG Assistant",
        layout="wide",
    )
    st.title("RAG Assistant")
    st.write("Ingest documents, search them, and ask questions grounded in sources.")

    with st.sidebar:
        st.header("API connection")
        base_url = st.text_input(
            "API base URL",
            value=settings.api_base_url,
            help="The FastAPI server address, for example http://localhost:8000.",
        )
        api_key = settings.api_auth_key or ""
        if not api_key:
            st.warning(
                "RAG_API_AUTH_KEY is not configured in .env. "
                "Protected API requests will fail."
            )
        top_k = st.slider(
            "Answer sources",
            min_value=1,
            max_value=20,
            value=min(max(settings.top_k, 1), 20),
        )
        _render_health_check(base_url)

    chat_tab, search_tab, ingestion_tab = st.tabs(
        ("Chat", "Search", "Ingest")
    )
    with chat_tab:
        _render_chat(base_url, api_key, top_k)
    with search_tab:
        _render_search(base_url, api_key)
    with ingestion_tab:
        _render_ingestion(base_url, api_key)


if __name__ == "__main__":
    main()