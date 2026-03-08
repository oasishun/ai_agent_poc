"""Application configuration via pydantic-settings."""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_model_fallback: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL_FALLBACK")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # RAG
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    rag_chunk_size: int = Field(default=500, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")

    # Mock API
    mock_latency_min: float = Field(default=0.5, alias="MOCK_LATENCY_MIN")
    mock_latency_max: float = Field(default=2.0, alias="MOCK_LATENCY_MAX")
    mock_error_rate: float = Field(default=0.05, alias="MOCK_ERROR_RATE")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file_dir: str = Field(default="./logs", alias="LOG_FILE_DIR")

    # Agent
    agent_max_iterations: int = Field(default=25, alias="AGENT_MAX_ITERATIONS")
    agent_timeout_seconds: int = Field(default=60, alias="AGENT_TIMEOUT_SECONDS")

    # Derived paths
    @property
    def data_dir(self) -> Path:
        return Path(__file__).parent.parent / "data"

    @property
    def structured_dir(self) -> Path:
        return self.data_dir / "structured"

    @property
    def unstructured_dir(self) -> Path:
        return self.data_dir / "unstructured"

    @property
    def tribal_dir(self) -> Path:
        return self.data_dir / "tribal"

    @property
    def traces_file(self) -> Path:
        return self.data_dir / "decision_traces" / "traces.json"

    @property
    def log_dir(self) -> Path:
        return Path(self.log_file_dir)


# Singleton
config = Config()
