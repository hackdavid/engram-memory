"""Engram SDK configuration via Pydantic BaseSettings."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings

from engram_memory.constants import (
    DEFAULT_ARCHIVE_THRESHOLD,
    DEFAULT_CACHE_MAX_SIZE,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_DECAY_FACTOR,
    DEFAULT_DECAY_INTERVAL_HOURS,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HIERARCHY_REBUILD_INTERVAL_HOURS,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_RATE_LIMIT_BURST,
    DEFAULT_LLM_RATE_LIMIT_RPM,
    DEFAULT_NEO4J_DATABASE,
    DEFAULT_NEO4J_MAX_POOL_SIZE,
    DEFAULT_SCORE_ALPHA,
    DEFAULT_SCORE_BETA,
    DEFAULT_SCORE_GAMMA,
    DEFAULT_TRAVERSAL_DECAY,
    DEFAULT_TRAVERSAL_MAX_DEPTH,
    DEFAULT_TRAVERSAL_MIN_SCORE,
    DEFAULT_USER_ID_PATTERN,
    DEFAULT_WEIGHT_LEARNING_INTERVAL_HOURS,
)


class Config(BaseSettings):
    """Engram SDK configuration.

    All fields can be set via environment variables (case-insensitive).

    LLM configuration uses LiteLLM's model naming convention:
      - OpenAI:    llm_model="gpt-4o-mini"
      - Anthropic: llm_model="anthropic/claude-sonnet-4-20250514"
      - Azure:     llm_model="azure/<deployment-name>"
                   + llm_api_base, llm_api_version
      - Timeouts:  llm_request_timeout (seconds) forwarded to LiteLLM/httpx; optional.
      - Google:    llm_model="gemini/gemini-2.5-pro"
      - Ollama:    llm_model="ollama/llama3"
    """

    model_config = {"env_prefix": "", "case_sensitive": False}

    # -- Neo4j --
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str = DEFAULT_NEO4J_DATABASE
    neo4j_max_pool_size: int = DEFAULT_NEO4J_MAX_POOL_SIZE

    # -- LLM (via LiteLLM) --
    llm_model: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None
    llm_api_version: Optional[str] = None
    llm_max_tokens: int = 4096
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES
    # Seconds; passed to LiteLLM/httpx. None = provider default (often 600s).
    llm_request_timeout: Optional[float] = None
    llm_rate_limit_rpm: int = DEFAULT_LLM_RATE_LIMIT_RPM
    llm_rate_limit_burst: int = DEFAULT_LLM_RATE_LIMIT_BURST

    # -- Embedding --
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    embedding_api_key: Optional[str] = None
    two_tier_embedding: bool = False

    # -- Scoring --
    score_alpha: float = DEFAULT_SCORE_ALPHA
    score_beta: float = DEFAULT_SCORE_BETA
    score_gamma: float = DEFAULT_SCORE_GAMMA

    # -- Traversal --
    traversal_decay: float = DEFAULT_TRAVERSAL_DECAY
    traversal_max_depth: int = DEFAULT_TRAVERSAL_MAX_DEPTH
    traversal_min_score: float = DEFAULT_TRAVERSAL_MIN_SCORE

    # -- Background tasks --
    enable_background_tasks: bool = True
    decay_interval_hours: int = DEFAULT_DECAY_INTERVAL_HOURS
    hierarchy_rebuild_interval_hours: int = DEFAULT_HIERARCHY_REBUILD_INTERVAL_HOURS
    weight_learning_interval_hours: int = DEFAULT_WEIGHT_LEARNING_INTERVAL_HOURS
    decay_factor: float = DEFAULT_DECAY_FACTOR
    archive_threshold: float = DEFAULT_ARCHIVE_THRESHOLD

    # -- Cache --
    cache_enabled: bool = True
    cache_max_size: int = DEFAULT_CACHE_MAX_SIZE
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS

    # -- Schema --
    auto_migrate: bool = True

    # -- Observability --
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"
    enable_tracing: bool = False

    # -- Validation --
    user_id_pattern: str = DEFAULT_USER_ID_PATTERN
