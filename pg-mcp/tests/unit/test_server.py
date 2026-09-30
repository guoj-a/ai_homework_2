"""Unit tests for the MCP server module.

Covers the lifespan startup/shutdown wiring (with fully mocked database and
service components) and the query tool's parameter validation and response
paths — without requiring a live PostgreSQL or LLM.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import pg_mcp.server as server_module
from pg_mcp.models.errors import LLMError
from pg_mcp.models.query import QueryResponse


@pytest.fixture(autouse=True)
def _reset_server_globals():
    """Save and restore server module globals around each test."""
    saved = (
        server_module._settings,
        server_module._pools,
        server_module._schema_cache,
        server_module._orchestrator,
        server_module._metrics,
        server_module._rate_limiter,
    )
    yield
    (
        server_module._settings,
        server_module._pools,
        server_module._schema_cache,
        server_module._orchestrator,
        server_module._metrics,
        server_module._rate_limiter,
    ) = saved


def _fake_pool() -> MagicMock:
    """Minimal async pool stand-in."""
    pool = MagicMock()
    pool.close = AsyncMock()
    return pool


class TestQueryTool:
    """Tests for the query tool function (no lifespan needed)."""

    async def test_not_initialized(self) -> None:
        """Query before initialization returns SERVER_NOT_INITIALIZED."""
        server_module._orchestrator = None
        result = await server_module.query(question="test")
        assert result["success"] is False
        assert result["error"]["code"] == "SERVER_NOT_INITIALIZED"

    async def test_invalid_return_type(self) -> None:
        """Invalid return_type returns INVALID_PARAMETER."""
        server_module._orchestrator = MagicMock()
        result = await server_module.query(question="test", return_type="bogus")
        assert result["error"]["code"] == "INVALID_PARAMETER"
        assert "bogus" in result["error"]["message"]

    async def test_invalid_request(self) -> None:
        """A request failing QueryRequest validation returns INVALID_REQUEST."""
        server_module._orchestrator = MagicMock()
        result = await server_module.query(question="")  # empty question rejected
        assert result["error"]["code"] == "INVALID_REQUEST"

    async def test_success_passthrough(self) -> None:
        """Successful orchestration returns the response dict with tokens_used."""
        response = QueryResponse(success=True, generated_sql="SELECT 1")
        orchestrator = MagicMock()
        orchestrator.execute_query = AsyncMock(return_value=response)
        server_module._orchestrator = orchestrator

        result = await server_module.query(question="test", database="db1")

        assert result["success"] is True
        assert result["generated_sql"] == "SELECT 1"
        assert result["tokens_used"] == 0  # guaranteed by to_dict
        orchestrator.execute_query.assert_awaited_once()

    async def test_unexpected_exception(self) -> None:
        """An unexpected orchestrator crash returns INTERNAL_ERROR."""
        orchestrator = MagicMock()
        orchestrator.execute_query = AsyncMock(side_effect=RuntimeError("boom"))
        server_module._orchestrator = orchestrator

        result = await server_module.query(question="test")

        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_ERROR"
        assert result["tokens_used"] == 0


class TestLifespan:
    """Tests for lifespan startup and shutdown wiring."""

    @pytest.fixture
    def wired_lifespan(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        """Patch all external dependencies and run lifespan startup.

        Returns a dict of captured constructor kwargs for assertions.
        """
        captured: dict = {}
        pools = {"dbA": _fake_pool(), "dbB": _fake_pool()}
        schema = MagicMock()
        schema.tables = []

        async def fake_create_pools(configs):
            captured["db_configs"] = configs
            return pools

        # Crafted settings with per-database policies
        from pg_mcp.config.settings import DatabaseConfig, OpenAIConfig, Settings

        settings = Settings(
            _env_file=None,
            openai=OpenAIConfig(api_key="sk-test"),
            database=DatabaseConfig(_env_file=None, name="dbA", blocked_tables=["secret"]),
            database_extra=[{"name": "dbB", "blocked_columns": "ssn"}],
        )
        monkeypatch.setattr(server_module, "Settings", lambda: settings)
        monkeypatch.setattr(server_module, "create_pools", fake_create_pools)
        monkeypatch.setattr(server_module, "configure_logging", MagicMock())

        schema_cache = MagicMock()
        schema_cache.load = AsyncMock(return_value=schema)
        schema_cache.stop_auto_refresh = AsyncMock()
        monkeypatch.setattr(server_module, "SchemaCache", lambda _cfg: schema_cache)

        executor = MagicMock()
        monkeypatch.setattr(server_module, "SQLExecutor", MagicMock(return_value=executor))
        monkeypatch.setattr(server_module, "SQLGenerator", MagicMock())
        monkeypatch.setattr(server_module, "SQLValidator", MagicMock())
        monkeypatch.setattr(server_module, "ResultValidator", MagicMock())

        def fake_orchestrator(**kwargs):
            captured["orchestrator_kwargs"] = kwargs
            orch = MagicMock()
            orch.circuit_breaker = MagicMock()
            return orch

        monkeypatch.setattr(server_module, "QueryOrchestrator", fake_orchestrator)
        return captured

    @pytest.mark.asyncio
    async def test_startup_creates_multi_db_components(self, wired_lifespan: dict) -> None:
        """Startup builds pools, executors and orchestrator for every database."""
        async with server_module.lifespan(server_module.mcp):
            kwargs = wired_lifespan["orchestrator_kwargs"]
            assert set(kwargs["sql_executors"].keys()) == {"dbA", "dbB"}
            assert set(kwargs["pools"].keys()) == {"dbA", "dbB"}
            assert kwargs["rate_limiter"] is not None
            # Per-database policies built from each database's own config
            policies = kwargs["database_policies"]
            assert policies["dbA"].blocked_tables == ["secret"]
            assert policies["dbB"].blocked_columns == ["ssn"]
            assert policies["dbA"].has_restrictions
            assert not DatabasePolicyEmpty(policies)  # helper asserts below

            # Globals wired for the query tool
            assert server_module._orchestrator is not None
            assert server_module._pools is not None
            assert server_module._rate_limiter is not None

    @pytest.mark.asyncio
    async def test_shutdown_closes_pools_and_cache(self, wired_lifespan: dict) -> None:
        """Shutdown stops auto-refresh and closes every pool."""
        async with server_module.lifespan(server_module.mcp):
            pools = server_module._pools
        for pool in pools.values():
            pool.close.assert_awaited_once()


def DatabasePolicyEmpty(policies: dict) -> bool:
    """True when no policy has restrictions."""
    return all(not p.has_restrictions for p in policies.values())


class TestLifespanShutdownResilience:
    """Shutdown must not raise even when cleanup itself fails."""

    @pytest.fixture
    def failing_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Wire lifespan with a schema cache whose stop_auto_refresh fails."""
        from pg_mcp.config.settings import DatabaseConfig, OpenAIConfig, Settings

        settings = Settings(
            _env_file=None,
            openai=OpenAIConfig(api_key="sk-test"),
            database=DatabaseConfig(_env_file=None, name="dbX"),
        )

        async def fake_create_pools(configs):
            return {cfg.name: _fake_pool() for cfg in configs}

        monkeypatch.setattr(server_module, "Settings", lambda: settings)
        monkeypatch.setattr(server_module, "create_pools", fake_create_pools)
        monkeypatch.setattr(server_module, "configure_logging", MagicMock())

        schema_cache = MagicMock()
        schema_cache.load = AsyncMock(return_value=MagicMock(tables=[]))
        schema_cache.stop_auto_refresh = AsyncMock(side_effect=RuntimeError("stop failed"))
        monkeypatch.setattr(server_module, "SchemaCache", lambda _cfg: schema_cache)
        monkeypatch.setattr(server_module, "SQLExecutor", MagicMock())
        monkeypatch.setattr(server_module, "SQLGenerator", MagicMock())
        monkeypatch.setattr(server_module, "SQLValidator", MagicMock())
        monkeypatch.setattr(server_module, "ResultValidator", MagicMock())
        monkeypatch.setattr(
            server_module,
            "QueryOrchestrator",
            MagicMock(**{"return_value.circuit_breaker": MagicMock()}),
        )
        return schema_cache

    @pytest.mark.asyncio
    async def test_shutdown_swallows_stop_auto_refresh_error(
        self, failing_cleanup: MagicMock
    ) -> None:
        """stop_auto_refresh failure is logged, not raised."""
        async with server_module.lifespan(server_module.mcp):
            pass
        failing_cleanup.stop_auto_refresh.assert_awaited_once()


class TestOrchestratorErrorPaths:
    """Verify orchestrator error mapping stays compatible with the server."""

    def test_llm_error_is_pg_mcp_error(self) -> None:
        """LLMError carries a code consumable by the server error mapping."""
        error = LLMError("circuit open")
        assert error.code is not None
        assert error.to_error_detail().code == error.code.value
