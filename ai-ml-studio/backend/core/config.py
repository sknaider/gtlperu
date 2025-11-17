"""
Configuration Management for AI ML Studio
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application Settings"""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )

    # API Keys
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    wandb_api_key: str = Field(default="", env="WANDB_API_KEY")
    huggingface_token: str = Field(default="", env="HUGGINGFACE_TOKEN")

    # Database
    database_url: str = Field(
        default="postgresql://aimluser:aimlpassword@localhost:5432/aimlstudio",
        env="DATABASE_URL"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")

    # Vector Database (ChromaDB)
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default=8001, env="CHROMA_PORT")
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        env="CHROMA_PERSIST_DIRECTORY"
    )

    # Storage Paths
    model_storage_path: str = Field(default="./models", env="MODEL_STORAGE_PATH")
    data_storage_path: str = Field(default="./data", env="DATA_STORAGE_PATH")
    checkpoint_path: str = Field(
        default="./models/checkpoints",
        env="CHECKPOINT_PATH"
    )
    logs_path: str = Field(default="./logs", env="LOGS_PATH")

    # Training Configuration
    default_batch_size: int = Field(default=32, env="DEFAULT_BATCH_SIZE")
    default_epochs: int = Field(default=10, env="DEFAULT_EPOCHS")
    default_learning_rate: float = Field(default=0.001, env="DEFAULT_LEARNING_RATE")
    mixed_precision: bool = Field(default=True, env="MIXED_PRECISION")
    gradient_checkpointing: bool = Field(
        default=False,
        env="GRADIENT_CHECKPOINTING"
    )

    # GPU Configuration
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    use_gpu: bool = Field(default=True, env="USE_GPU")
    multi_gpu: bool = Field(default=False, env="MULTI_GPU")
    gpu_memory_fraction: float = Field(default=0.9, env="GPU_MEMORY_FRACTION")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="ai-ml-studio",
        env="MLFLOW_EXPERIMENT_NAME"
    )
    mlflow_artifact_root: str = Field(default="./mlruns", env="MLFLOW_ARTIFACT_ROOT")

    # Security
    secret_key: str = Field(
        default="your_secret_key_change_this_in_production",
        env="SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1",
        env="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2",
        env="CELERY_RESULT_BACKEND"
    )

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")

    # Cache
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")

    # Model Defaults
    default_model_name: str = Field(default="gpt2", env="DEFAULT_MODEL_NAME")
    default_tokenizer: str = Field(default="gpt2", env="DEFAULT_TOKENIZER")
    max_seq_length: int = Field(default=512, env="MAX_SEQ_LENGTH")

    # Dataset Generation
    synthetic_data_samples: int = Field(
        default=1000,
        env="SYNTHETIC_DATA_SAMPLES"
    )
    claude_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        env="CLAUDE_MODEL"
    )
    claude_max_tokens: int = Field(default=4096, env="CLAUDE_MAX_TOKENS")
    claude_temperature: float = Field(default=1.0, env="CLAUDE_TEMPERATURE")

    # RAG Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")

    # Environment
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    environment: str = Field(default="development", env="ENVIRONMENT")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.model_storage_path,
            self.data_storage_path,
            self.checkpoint_path,
            self.logs_path,
            self.chroma_persist_directory,
            self.mlflow_artifact_root,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Global settings instance
settings = Settings()

# Ensure all directories exist
settings.ensure_directories()
