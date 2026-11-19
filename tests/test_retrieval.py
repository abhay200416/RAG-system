from langchain_core.documents import Document

from rag.retrieval import retrieval


class _Result:
    def __init__(self, *, has_bm25: bool = False, rows=None) -> None:
        self.has_bm25 = has_bm25
        self.rows = rows or []

    def scalar_one(self) -> bool:
        return self.has_bm25

    def mappings(self):
        return self

    def all(self):
        return self.rows


class _Connection:
    def __init__(self, *, has_bm25: bool, rows: list[dict]) -> None:
        self.has_bm25 = has_bm25
        self.rows = rows
        self.statements = []

    def execute(self, statement, params=None):
        self.statements.append((statement, params))
        if statement is retrieval._SEARCH_CAPABILITIES:
            return _Result(has_bm25=self.has_bm25)
        return _Result(rows=self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


class _Engine:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def connect(self) -> _Connection:
        return self.connection


class _VectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def similarity_search_with_score(self, query: str, *, k: int):
        return [(document, 0.1) for document in self.documents[:k]]


def test_hybrid_search_falls_back_to_native_postgres_full_text(monkeypatch) -> None:
    lexical_document = Document(
        id="fts-1",
        page_content="A matching lexical result",
        metadata={"source": "manual.txt"},
    )
    connection = _Connection(
        has_bm25=False,
        rows=[
            {
                "id": lexical_document.id,
                "document": lexical_document.page_content,
                "cmetadata": lexical_document.metadata,
            }
        ],
    )
    monkeypatch.setattr(retrieval, "engine", _Engine(connection))
    monkeypatch.setattr(retrieval, "_get_vector_store", lambda: _VectorStore([]))

    results = retrieval.HYBRID_SEARCH("matching words", top_k=2)

    assert results == [lexical_document]
    assert connection.statements[1][0] is retrieval._POSTGRES_FTS_SEARCH
    assert connection.statements[1][1] == {
        "query": "matching words",
        "limit": 4,
    }


def test_hybrid_search_uses_bm25_when_extension_and_index_exist(monkeypatch) -> None:
    bm25_document = Document(
        id="bm25-1",
        page_content="A matching BM25 result",
        metadata={"source": "guide.md"},
    )
    connection = _Connection(
        has_bm25=True,
        rows=[
            {
                "id": bm25_document.id,
                "document": bm25_document.page_content,
                "cmetadata": bm25_document.metadata,
            }
        ],
    )
    monkeypatch.setattr(retrieval, "engine", _Engine(connection))
    monkeypatch.setattr(retrieval, "_get_vector_store", lambda: _VectorStore([]))

    results = retrieval.HYBRID_SEARCH("matching words", top_k=2)

    assert results == [bm25_document]
    assert connection.statements[1][0] is retrieval._BM25_SEARCH
