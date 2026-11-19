import pytest
from pydantic import ValidationError

from rag.util.config import Settings


def test_database_url_must_come_from_environment_or_env_file(monkeypatch) -> None:
    monkeypatch.delenv("RAG_DATABASE_URL", raising=False)

    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)
