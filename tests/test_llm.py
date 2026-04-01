"""LLM adapter tests (retry, circuit breaker, JSON parsing, LiteLLM)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from engram_memory.exceptions import CircuitOpenError, ExtractionError
from engram_memory.llm.base import BaseLLM


def test_base_llm_is_abstract():
    with pytest.raises(TypeError):
        BaseLLM()


# ── LiteLLM Adapter ─────────────────────────────────────────────────


def _make_litellm_adapter(**kwargs):
    """Instantiate LiteLLMAdapter and swap its internal litellm ref with a mock."""
    from engram_memory.llm.litellm_adapter import LiteLLMAdapter

    defaults = {"model": "gpt-4o-mini", "api_key": "sk-test"}
    defaults.update(kwargs)
    adapter = LiteLLMAdapter(**defaults)
    adapter._base_delay = 0.01

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock()
    adapter._litellm = mock_litellm
    return adapter, mock_litellm


def _mock_completion(text: str):
    """Create a mock litellm completion response."""
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


@pytest.mark.asyncio
async def test_litellm_generate_json_success():
    adapter, mock_ll = _make_litellm_adapter()
    mock_ll.acompletion.return_value = _mock_completion(
        '{"nodes": [], "relationships": []}'
    )
    result = await adapter.generate_json(system="Extract entities.", user="Alice works at Google.")
    assert "nodes" in result
    assert "relationships" in result
    mock_ll.acompletion.assert_awaited_once()


@pytest.mark.asyncio
async def test_litellm_passes_api_key_and_model():
    adapter, mock_ll = _make_litellm_adapter(
        model="anthropic/claude-sonnet-4-20250514", api_key="sk-ant-123"
    )
    mock_ll.acompletion.return_value = _mock_completion('{"ok": true}')
    await adapter.generate_json(system="test", user="test")
    call_kwargs = mock_ll.acompletion.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-sonnet-4-20250514"
    assert call_kwargs["api_key"] == "sk-ant-123"


@pytest.mark.asyncio
async def test_litellm_azure_passes_base_and_version():
    adapter, mock_ll = _make_litellm_adapter(
        model="azure/my-deployment",
        api_key="azure-key",
        api_base="https://myresource.openai.azure.com/",
        api_version="2024-02-01",
    )
    mock_ll.acompletion.return_value = _mock_completion('{"result": "ok"}')
    await adapter.generate_json(system="test", user="test")
    call_kwargs = mock_ll.acompletion.call_args.kwargs
    assert call_kwargs["model"] == "azure/my-deployment"
    assert call_kwargs["api_base"] == "https://myresource.openai.azure.com/"
    assert call_kwargs["api_version"] == "2024-02-01"


@pytest.mark.asyncio
async def test_litellm_non_azure_omits_base_and_version():
    adapter, mock_ll = _make_litellm_adapter(model="gpt-4o-mini", api_key="sk-123")
    mock_ll.acompletion.return_value = _mock_completion('{"ok": true}')
    await adapter.generate_json(system="test", user="test")
    call_kwargs = mock_ll.acompletion.call_args.kwargs
    assert "api_base" not in call_kwargs
    assert "api_version" not in call_kwargs


@pytest.mark.asyncio
async def test_litellm_retry_on_failure():
    adapter, mock_ll = _make_litellm_adapter(max_retries=3)
    mock_ll.acompletion.side_effect = [
        Exception("API error"),
        Exception("API error"),
        _mock_completion('{"nodes": []}'),
    ]
    result = await adapter.generate_json(system="test", user="test")
    assert mock_ll.acompletion.call_count == 3
    assert "nodes" in result


@pytest.mark.asyncio
async def test_litellm_extraction_error_after_all_retries():
    adapter, mock_ll = _make_litellm_adapter(max_retries=2)
    mock_ll.acompletion.side_effect = Exception("API error")
    with pytest.raises(ExtractionError):
        await adapter.generate_json(system="test", user="test")


@pytest.mark.asyncio
async def test_litellm_circuit_breaker_opens():
    adapter, mock_ll = _make_litellm_adapter(
        max_retries=1, circuit_breaker_threshold=3
    )
    mock_ll.acompletion.side_effect = Exception("fail")
    for _ in range(3):
        with pytest.raises(ExtractionError):
            await adapter.generate_json(system="test", user="test")

    with pytest.raises(CircuitOpenError):
        await adapter.generate_json(system="test", user="test")


@pytest.mark.asyncio
async def test_litellm_extra_params_forwarded():
    adapter, mock_ll = _make_litellm_adapter(
        extra_params={"temperature": 0.0, "top_p": 0.9}
    )
    mock_ll.acompletion.return_value = _mock_completion('{"ok": true}')
    await adapter.generate_json(system="test", user="test")
    call_kwargs = mock_ll.acompletion.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["top_p"] == 0.9


@pytest.mark.asyncio
async def test_litellm_ping_success():
    adapter, mock_ll = _make_litellm_adapter()
    mock_ll.acompletion.return_value = _mock_completion("OK")
    assert await adapter.ping() is True


@pytest.mark.asyncio
async def test_litellm_ping_failure():
    adapter, mock_ll = _make_litellm_adapter()
    mock_ll.acompletion.side_effect = Exception("timeout")
    assert await adapter.ping() is False


@pytest.mark.asyncio
async def test_litellm_max_tokens_forwarded():
    adapter, mock_ll = _make_litellm_adapter(max_tokens=2048)
    mock_ll.acompletion.return_value = _mock_completion('{"ok": true}')
    await adapter.generate_json(system="test", user="test")
    call_kwargs = mock_ll.acompletion.call_args.kwargs
    assert call_kwargs["max_tokens"] == 2048


# ── JSON parsing ────────────────────────────────────────────────────


def test_parse_json_plain():
    result = BaseLLM._parse_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_with_code_fence():
    raw = '```json\n{"key": "value"}\n```'
    result = BaseLLM._parse_json(raw)
    assert result == {"key": "value"}


def test_parse_json_invalid_raises():
    with pytest.raises(Exception):
        BaseLLM._parse_json("not json at all")
