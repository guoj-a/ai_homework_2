"""Unit tests for ResultValidator with a mocked OpenAI client."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.models.query import ResultValidationResult
from pg_mcp.services.result_validator import ResultValidator


def _make_validator(enabled: bool = True) -> ResultValidator:
    """Build a validator with a mocked AsyncOpenAI client."""
    validator = ResultValidator(
        openai_config=OpenAIConfig(api_key="sk-test"),
        validation_config=ValidationConfig(enabled=enabled, confidence_threshold=70),
    )
    validator.client = MagicMock()
    validator.client.chat.completions.create = AsyncMock()
    return validator


def _response(content: str | None, choices: int = 1) -> MagicMock:
    """Build a fake ChatCompletion response."""
    response = MagicMock()
    response.choices = []
    for _ in range(choices):
        choice = MagicMock()
        choice.message.content = content
        response.choices.append(choice)
    return response


class TestResultValidator:
    """Tests for the LLM-based result validation."""

    async def test_disabled_returns_full_confidence(self) -> None:
        """Disabled validation short-circuits with confidence 100."""
        validator = _make_validator(enabled=False)
        result = await validator.validate("q", "SELECT 1", [{"a": 1}], 1)
        assert result.confidence == 100
        assert result.is_acceptable
        validator.client.chat.completions.create.assert_not_awaited()

    async def test_valid_json_response(self) -> None:
        """A valid JSON verdict is mapped to a result."""
        validator = _make_validator()
        verdict = {
            "confidence": 85,
            "explanation": "matches",
            "suggestion": None,
        }
        validator.client.chat.completions.create.return_value = _response(json.dumps(verdict))
        result = await validator.validate("q", "SELECT 1", [{"a": 1}], 1)
        assert result.confidence == 85
        assert result.is_acceptable
        assert result.explanation == "matches"

    async def test_low_confidence_not_acceptable(self) -> None:
        """Below-threshold confidence is marked unacceptable."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _response(
            json.dumps({"confidence": 30, "explanation": "wrong"})
        )
        result = await validator.validate("q", "SELECT 1", [], 0)
        assert result.confidence == 30
        assert not result.is_acceptable

    async def test_empty_choices_raises_llm_error(self) -> None:
        """An empty choices list raises LLMError."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _response(None, choices=0)
        with pytest.raises(LLMError, match="empty response"):
            await validator.validate("q", "SELECT 1", [], 0)

    async def test_empty_content_raises_llm_error(self) -> None:
        """Empty message content raises LLMError."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _response(None)
        with pytest.raises(LLMError, match="empty message content"):
            await validator.validate("q", "SELECT 1", [], 0)

    async def test_invalid_json_returns_moderate_confidence(self) -> None:
        """Unparseable JSON degrades to confidence 60 instead of raising."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _response("not json")
        result = await validator.validate("q", "SELECT 1", [], 0)
        assert result.confidence == 60
        assert not result.is_acceptable

    async def test_out_of_bounds_confidence_clamped(self) -> None:
        """Float confidence is clamped into [0, 100]."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _response(
            json.dumps({"confidence": 250.0})
        )
        result = await validator.validate("q", "SELECT 1", [], 0)
        assert result.confidence == 100

    async def test_non_numeric_confidence_defaults_50(self) -> None:
        """A non-numeric confidence falls back to 50."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _response(
            json.dumps({"confidence": "high"})
        )
        result = await validator.validate("q", "SELECT 1", [], 0)
        assert result.confidence == 50

    async def test_timeout_converted(self) -> None:
        """TimeoutError from the client becomes LLMTimeoutError."""
        validator = _make_validator()
        validator.client.chat.completions.create.side_effect = TimeoutError()
        with pytest.raises(LLMTimeoutError, match="timed out"):
            await validator.validate("q", "SELECT 1", [], 0)

    async def test_auth_error_converted(self) -> None:
        """Authentication failures become LLMUnavailableError."""
        validator = _make_validator()
        validator.client.chat.completions.create.side_effect = Exception("invalid api_key provided")
        with pytest.raises(LLMUnavailableError, match="authentication"):
            await validator.validate("q", "SELECT 1", [], 0)

    async def test_rate_limit_error_converted(self) -> None:
        """Rate limit failures become LLMUnavailableError."""
        validator = _make_validator()
        validator.client.chat.completions.create.side_effect = Exception("rate_limit exceeded")
        with pytest.raises(LLMUnavailableError, match="rate limit"):
            await validator.validate("q", "SELECT 1", [], 0)

    async def test_generic_error_converted(self) -> None:
        """Unknown client errors become LLMError."""
        validator = _make_validator()
        validator.client.chat.completions.create.side_effect = Exception("weird")
        with pytest.raises(LLMError, match="Result validation failed"):
            await validator.validate("q", "SELECT 1", [], 0)

    async def test_sample_rows_limit(self) -> None:
        """Results are sampled down to validation_config.sample_rows."""
        validator = _make_validator()
        validator.validation_config.sample_rows = 2
        validator.client.chat.completions.create.return_value = _response(
            json.dumps({"confidence": 90})
        )
        rows = [{"i": i} for i in range(10)]
        await validator.validate("q", "SELECT 1", rows, 10)
        call_kwargs = validator.client.chat.completions.create.call_args.kwargs
        prompt = call_kwargs["messages"][1]["content"]
        assert '"i": 9' not in prompt  # only first 2 rows included

    def test_acceptable_result_model(self) -> None:
        """ResultValidationResult enforces confidence bounds at model level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResultValidationResult(
                confidence=101, explanation="", suggestion=None, is_acceptable=True
            )
        result = ResultValidationResult(
            confidence=100, explanation="", suggestion=None, is_acceptable=True
        )
        assert result.confidence == 100
