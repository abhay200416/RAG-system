FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir uv \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app /opt/venv \
    && chown -R app:app /app /opt/venv

WORKDIR /app

COPY --chown=app:app pyproject.toml uv.lock README.md ./
COPY --chown=app:app src ./src

USER app

RUN uv sync --frozen --no-dev

EXPOSE 8000 8501

CMD ["/opt/venv/bin/rag-api"]
