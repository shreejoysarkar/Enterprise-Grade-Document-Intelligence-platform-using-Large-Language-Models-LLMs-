"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # HuggingFace Configuration
    HF_API_KEY: str

    # Pinecone Configuration
    PINECONE_API_KEY: str
    pinecone_index_name: str = "doc-intel-hybrid"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "dotproduct"
    pinecone_namespace: str = "default"

    # Embedding Model Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64

    # Document Processing Settings
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # LLM Configuration
    llm_model: str = "llama3.1"
    llm_temperature: float = 0.0

    # Retrieval Settings
    retrieval_k: int = 2

    # Logging
    log_level: str = "INFO"

    # RAGAS Evaluation Settings
    enable_ragas_evaluation: bool = True
    ragas_timeout_seconds: float = 30.0
    ragas_log_results: bool = True
    ragas_llm_model: str | None = None 
    ragas_llm_temperature: float | None = None 
    ragas_embedding_model: str | None = None 

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Application Info
    app_name: str = "doc_intel_platform"
    app_version: str = "0.1.0"

    # Data Directory
    data_directory: str = "Data/Input"



@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()