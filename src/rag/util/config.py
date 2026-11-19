"""
Application configuration management for the RAG project.

Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

from functools import lru_cache
from typing import Optional, List
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "RAG System"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, description="Enable debug mode")

    # Logging
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_format: str = Field(default="console", description="Log format: 'console' or 'json'")
    log_file: Optional[str] = Field(default=None, description="Path to log file")

    # API Keys (accepts either GOOGLE_API_KEY or RAG_GOOGLE_API_KEY)
    google_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "RAG_GOOGLE_API_KEY"),
        description="Google API key for Gemini",
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "RAG_GROQ_API_KEY"),
        description="Groq API key",
    )

    # Model Configuration
    default_embedding_model: str = Field(default="gemini-embedding-2", description="Default embedding model")
    default_llm_model: str = Field(default="gemini-3.5-flash-lite", description="Default LLM model")
    groq_model: str = Field(default="qwen/qwen3.8-27b", description="Groq fallback LLM model")

    # Vector Store
    database_url: str = Field(
        min_length=1,
        description="SQLAlchemy connection URL for PostgreSQL and PGVector",
    )
    collection_name: str = Field(default="rag_documents", description="Vector store collection name")

    # Chunking
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap")

    # Retrieval
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for retrieval")

    # Redis answer cache (disabled when no URL is configured)
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    redis_cache_ttl_seconds: int = Field(
        default=3600,
        ge=1,
        description="Answer cache lifetime in seconds",
    )
    redis_cache_prefix: str = Field(default="rag", description="Prefix for Redis cache keys")

    # Server (if API is added later)
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="FastAPI base URL used by the Streamlit frontend",
    )
    api_auth_key: Optional[str] = Field(
        default=None,
        description="Bearer API key; protected API routes fail closed when unset",
    )
    api_rate_limit_requests: int = Field(
        default=60,
        ge=1,
        description="Requests permitted per rate-limit window and client IP",
    )
    api_rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        description="API rate-limit window duration",
    )
    api_rate_limit_prefix: str = Field(
        default="rag:api-rate-limit",
        description="Redis key prefix for API rate limits",
    )

    # Ingestion
    supported_extensions: List[str] = Field(
        default=[".txt", ".md", ".pdf", ".html", ".htm"],
        description="Supported file extensions for ingestion",
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        description="Maximum file size in MB",
    )
    ingest_allowed_hosts: List[str] = Field(
        default_factory=list,
        description="Exact hostnames permitted by the URL-ingestion endpoint",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for direct access
settings = get_settings()
