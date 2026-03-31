"""Phase 1: Verify Config loads from env vars and validates correctly."""

import re

import pytest

from engram.config import Config


def _make_config(**overrides):
    """Helper to build a Config with required fields pre-filled.

    LLM fields are set explicitly so real ``LLM_*`` in the process env cannot
    override expected test values (pydantic-settings merges env with init data).
    """
    defaults = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "test",
        "llm_model": "gpt-4o-mini",
        "llm_api_key": None,
        "llm_api_base": None,
        "llm_api_version": None,
        "llm_request_timeout": None,
    }
    defaults.update(overrides)
    return Config(**defaults)


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_API_BASE",
        "LLM_API_VERSION",
        "LLM_MAX_TOKENS",
        "LLM_REQUEST_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_config_from_env(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    cfg = Config()
    assert cfg.neo4j_uri == "bolt://localhost:7687"
    assert cfg.llm_model == "gpt-4o-mini"


def test_config_defaults():
    cfg = _make_config()
    assert cfg.llm_model == "gpt-4o-mini"
    assert cfg.llm_api_key is None
    assert cfg.llm_api_base is None
    assert cfg.llm_api_version is None
    assert cfg.llm_request_timeout is None
    assert cfg.llm_max_tokens == 4096
    assert cfg.score_alpha == 0.5
    assert cfg.score_beta == 0.35
    assert cfg.score_gamma == 0.15
    assert cfg.traversal_max_depth == 5
    assert cfg.cache_enabled is True
    assert cfg.auto_migrate is True
    assert cfg.log_level == "INFO"
    assert cfg.log_format == "text"


def test_config_azure_fields():
    cfg = _make_config(
        llm_model="azure/my-gpt4-deployment",
        llm_api_key="azure-key-123",
        llm_api_base="https://myresource.openai.azure.com/",
        llm_api_version="2024-02-01",
    )
    assert cfg.llm_model == "azure/my-gpt4-deployment"
    assert cfg.llm_api_base == "https://myresource.openai.azure.com/"
    assert cfg.llm_api_version == "2024-02-01"


def test_config_anthropic_model():
    cfg = _make_config(
        llm_model="anthropic/claude-sonnet-4-20250514",
        llm_api_key="sk-ant-test",
    )
    assert cfg.llm_model == "anthropic/claude-sonnet-4-20250514"


def test_config_openai_model():
    cfg = _make_config(
        llm_model="gpt-4o",
        llm_api_key="sk-openai-test",
    )
    assert cfg.llm_model == "gpt-4o"


def test_config_llm_request_timeout():
    cfg = _make_config(llm_request_timeout=45.0)
    assert cfg.llm_request_timeout == 45.0


def test_config_non_azure_no_base():
    cfg = _make_config(llm_model="gpt-4o-mini", llm_api_key="sk-123")
    assert cfg.llm_api_base is None
    assert cfg.llm_api_version is None


def test_user_id_pattern_default():
    cfg = _make_config()
    assert re.match(cfg.user_id_pattern, "valid-user_123")
    assert not re.match(cfg.user_id_pattern, "bad user!@#")
    assert not re.match(cfg.user_id_pattern, "")


def test_config_custom_scoring():
    cfg = _make_config(score_alpha=0.7, score_beta=0.2, score_gamma=0.1)
    assert cfg.score_alpha == 0.7


def test_config_neo4j_pool_size():
    cfg = _make_config(neo4j_max_pool_size=100)
    assert cfg.neo4j_max_pool_size == 100


def test_config_embedding_api_key():
    cfg = _make_config(embedding_provider="openai", embedding_api_key="sk-embed")
    assert cfg.embedding_api_key == "sk-embed"


def test_config_from_env_llm_fields(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    monkeypatch.setenv("LLM_MODEL", "azure/gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "my-azure-key")
    monkeypatch.setenv("LLM_API_BASE", "https://myres.openai.azure.com/")
    monkeypatch.setenv("LLM_API_VERSION", "2024-06-01")
    cfg = Config()
    assert cfg.llm_model == "azure/gpt-4"
    assert cfg.llm_api_key == "my-azure-key"
    assert cfg.llm_api_base == "https://myres.openai.azure.com/"
    assert cfg.llm_api_version == "2024-06-01"
