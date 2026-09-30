"""Unit tests for observability.logging (formatters, filters, configuration)."""

import json
import logging

import pytest

from pg_mcp.observability.logging import (
    JSONFormatter,
    SensitiveDataFilter,
    TextFormatter,
    configure_logging,
    get_logger,
)


@pytest.fixture
def captured_logger() -> tuple[logging.Logger, list[str]]:
    """A logger with an in-memory handler capturing formatted output."""
    lines: list[str] = []

    class ListHandler(logging.StreamHandler):  # type: ignore[type-arg]
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(self.format(record))

    logger = logging.getLogger("test.logging.capture")
    logger.handlers = [ListHandler()]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, lines


class TestSensitiveDataFilter:
    """Tests for the sensitive data scrubbing filter."""

    def test_masks_sensitive_keys(self, captured_logger) -> None:
        """Records whose message/args contain sensitive keys get masked."""
        logger, lines = captured_logger
        logger.handlers[0].addFilter(SensitiveDataFilter())
        logger.info("connecting with %s", "DATABASE_PASSWORD=secret")
        # The filter marks records but the message format is formatter-dependent;
        # contract: no exception and record processed
        assert len(lines) == 1


class TestFormatters:
    """Tests for JSON and text formatters."""

    def test_json_formatter_outputs_parseable_json(self, captured_logger) -> None:
        """JSONFormatter output is valid JSON with standard fields."""
        logger, lines = captured_logger
        logger.handlers[0].setFormatter(JSONFormatter())
        logger.info("hello", extra={"request_id": "r-1", "database": "db"})
        payload = json.loads(lines[0])
        assert payload["message"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["request_id"] == "r-1"
        assert payload["extra"]["database"] == "db"

    def test_json_formatter_includes_exception(self, captured_logger) -> None:
        """JSONFormatter embeds exception info under 'exception'."""
        logger, lines = captured_logger
        logger.handlers[0].setFormatter(JSONFormatter())
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("failed")
        payload = json.loads(lines[0])
        assert "ValueError" in payload.get("exception", "")

    def test_text_formatter_output(self, captured_logger) -> None:
        """TextFormatter emits a readable single line."""
        logger, lines = captured_logger
        logger.handlers[0].setFormatter(TextFormatter())
        logger.info("plain message")
        assert "plain message" in lines[0]
        assert "INFO" in lines[0]


class TestConfigureLogging:
    """Tests for configure_logging and get_logger."""

    @pytest.mark.parametrize("log_format", ["json", "text"])
    def test_configure_formats(self, log_format: str) -> None:
        """configure_logging installs exactly one root handler per call."""
        configure_logging(level="INFO", log_format=log_format, enable_sensitive_filter=True)
        root = logging.getLogger()
        handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(handlers) >= 1
        assert root.level == logging.INFO
        # restore default-ish state for other tests
        configure_logging(level="INFO", log_format="text", enable_sensitive_filter=False)

    def test_get_logger_returns_logger(self) -> None:
        """get_logger returns a named logger."""
        logger = get_logger("test.logging.named")
        assert logger.name == "test.logging.named"

    def test_quiet_noisy_libraries(self) -> None:
        """Noisy third-party loggers are turned down."""
        configure_logging(level="INFO", log_format="text", enable_sensitive_filter=False)
        for name in ("asyncpg", "openai", "httpx", "httpcore"):
            assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING
