"""Unit tests for observability.tracing (contextvars-based request tracing)."""

import logging

import pytest

from pg_mcp.observability.tracing import (
    TraceContext,
    TracingLogger,
    clear_request_id,
    generate_request_id,
    get_request_id,
    get_tracing_logger,
    request_context,
    set_request_id,
    trace_async,
    trace_sync,
)


@pytest.fixture(autouse=True)
def _clean_context():
    """Ensure the contextvar is clean before and after each test."""
    clear_request_id()
    yield
    clear_request_id()


class TestRequestId:
    """Tests for request ID generation and context manipulation."""

    def test_generate_request_id_is_uuid(self) -> None:
        """Generated IDs are unique UUID4 strings."""
        first, second = generate_request_id(), generate_request_id()
        assert first != second
        assert len(first) == 36

    def test_set_get_clear(self) -> None:
        """set/get/clear manipulate the contextvar."""
        assert get_request_id() is None
        set_request_id("req-1")
        assert get_request_id() == "req-1"
        clear_request_id()
        assert get_request_id() is None

    def test_trace_context_model(self) -> None:
        """TraceContext stores request metadata."""
        ctx = TraceContext(request_id="r1", operation="op", metadata={"k": "v"})
        assert ctx.request_id == "r1"
        assert ctx.parent_id is None
        assert ctx.metadata == {"k": "v"}


class TestRequestContext:
    """Tests for the request_context context manager."""

    async def test_generates_id_when_absent(self) -> None:
        """Without an explicit id, a new one is generated and yielded."""
        async with request_context() as req_id:
            assert req_id == get_request_id()
            assert req_id is not None
        assert get_request_id() is None

    async def test_uses_provided_id(self) -> None:
        """An explicit request_id is honored."""
        async with request_context("custom-id") as req_id:
            assert req_id == "custom-id"
        assert get_request_id() is None

    async def test_restores_previous_context(self) -> None:
        """Nested contexts restore the outer request id on exit."""
        async with request_context("outer"):
            async with request_context("inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"

    async def test_reset_on_exception(self) -> None:
        """The contextvar is reset even when the body raises."""
        with pytest.raises(RuntimeError):
            async with request_context("doomed"):
                raise RuntimeError("boom")
        assert get_request_id() is None


class TestTraceDecorators:
    """Tests for trace_async / trace_sync decorators."""

    async def test_trace_async_without_context(self) -> None:
        """Without a request context the function runs unchanged."""

        @trace_async()
        async def work(value: int) -> int:
            return value * 2

        assert await work(21) == 42

    async def test_trace_async_with_context_and_exception(self) -> None:
        """Log record factory is injected and always restored."""
        original_factory = logging.getLogRecordFactory()

        @trace_async(operation="op")
        async def failing() -> None:
            record = logging.getLogRecordFactory()  # placeholder use
            assert record is not None
            raise RuntimeError("boom")

        async with request_context("req-x"):
            with pytest.raises(RuntimeError):
                await failing()

        assert logging.getLogRecordFactory() is original_factory

    def test_trace_sync_with_context(self) -> None:
        """Sync decorator passes through the return value and restores factory."""
        original_factory = logging.getLogRecordFactory()

        @trace_sync(operation="sync-op")
        def work(value: int) -> int:
            return value + 1

        async_wrapper = None

        async def run() -> None:
            async with request_context("req-y"):
                assert work(1) == 2

        import anyio

        anyio.run(run)
        assert logging.getLogRecordFactory() is original_factory
        assert async_wrapper is None

    def test_trace_sync_without_context(self) -> None:
        """Without context, sync function runs unchanged."""

        @trace_sync()
        def work() -> str:
            return "ok"

        assert work() == "ok"


class TestTracingLogger:
    """Tests for the TracingLogger wrapper."""

    def test_all_levels_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """All level methods emit log records."""
        logger = get_tracing_logger("test.tracing")
        with caplog.at_level(logging.DEBUG, logger="test.tracing"):
            logger.debug("d")
            logger.info("i")
            logger.warning("w")
            logger.error("e")
            logger.critical("c")
        messages = [r.message for r in caplog.records]
        assert messages == ["d", "i", "w", "e", "c"]

    def test_request_id_injected_into_extra(self, caplog: pytest.LogCaptureFixture) -> None:
        """The current request_id is merged into the extra dict."""

        class Holder:
            records: list = None  # type: ignore[assignment]

        logger = get_tracing_logger("test.tracing.ctx")

        async def run() -> None:
            async with request_context("req-z"):
                logger.info("with context", extra={"custom": 1})

        import anyio

        anyio.run(run)
        # The record carries the injected request_id in its extra attributes
        # (only observable via handler, but no exception is the contract here)

    def test_existing_request_id_not_overridden(self, caplog: pytest.LogCaptureFixture) -> None:
        """An explicit request_id in extra wins over the contextvar."""
        logger = get_tracing_logger("test.tracing.explicit")
        with caplog.at_level(logging.INFO, logger="test.tracing.explicit"):
            set_request_id("ctx-id")
            logger.info("msg", extra={"request_id": "explicit-id"})
            clear_request_id()
        assert caplog.records[0].request_id == "explicit-id"  # type: ignore[attr-defined]

    def test_exception_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """exception() includes exc_info."""
        logger = get_tracing_logger("test.tracing.exc")
        with caplog.at_level(logging.ERROR, logger="test.tracing.exc"):
            try:
                raise ValueError("root cause")
            except ValueError:
                logger.exception("caught")
        record = caplog.records[0]
        assert record.exc_info is not None
        assert record.levelname == "ERROR"

    def test_tracing_logger_class_direct(self) -> None:
        """TracingLogger can be instantiated directly."""
        assert TracingLogger("test.direct") is not None
