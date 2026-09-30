"""Unit tests for db.pool helpers with mocked asyncpg."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import DatabaseConfig
from pg_mcp.db.pool import close_pools, create_pool, create_pools


def _config(name: str = "db1") -> DatabaseConfig:
    return DatabaseConfig(_env_file=None, name=name, password="pw")


def _fake_asyncpg_pool() -> MagicMock:
    pool = MagicMock()
    pool.close = AsyncMock()
    return pool


class TestCreatePool:
    """Tests for pool creation (asyncpg mocked)."""

    async def test_create_pool_passes_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Connection kwargs mirror the DatabaseConfig values."""
        captured: dict = {}

        async def fake_create_pool(**kwargs):
            captured.update(kwargs)
            return _fake_asyncpg_pool()

        monkeypatch.setattr("asyncpg.create_pool", fake_create_pool)
        cfg = DatabaseConfig(
            _env_file=None,
            name="mydb",
            user="u",
            password="p",
            host="h",
            port=5433,
            min_pool_size=2,
            max_pool_size=7,
            pool_timeout=11.0,
            command_timeout=13.0,
        )
        pool = await create_pool(cfg)
        assert pool is not None
        assert captured["database"] == "mydb"
        assert captured["port"] == 5433
        assert captured["min_size"] == 2
        assert captured["max_size"] == 7
        assert captured["timeout"] == 11.0
        assert captured["command_timeout"] == 13.0

    async def test_create_pool_rejects_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A None pool from asyncpg raises RuntimeError."""

        async def fake_create_pool(**kwargs):
            return None

        monkeypatch.setattr("asyncpg.create_pool", fake_create_pool)
        with pytest.raises(RuntimeError, match="Failed to create connection pool"):
            await create_pool(_config())


class TestCreatePools:
    """Tests for multi-database pool creation."""

    async def test_creates_one_pool_per_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pools are keyed by database name."""
        created: list[str] = []

        async def fake_create_pool(**kwargs):
            created.append(kwargs["database"])
            return _fake_asyncpg_pool()

        monkeypatch.setattr("asyncpg.create_pool", fake_create_pool)
        configs = [_config("dbA"), _config("dbB")]
        pools = await create_pools(configs)
        assert set(pools.keys()) == {"dbA", "dbB"}
        assert sorted(created) == ["dbA", "dbB"]


class TestClosePools:
    """Tests for graceful pool shutdown."""

    async def test_graceful_close(self) -> None:
        """Pools are closed and not terminated on success."""
        pools = {"dbA": _fake_asyncpg_pool(), "dbB": _fake_asyncpg_pool()}
        await close_pools(pools, grace_period=5.0)
        for pool in pools.values():
            pool.close.assert_awaited_once()
            pool.terminate.assert_not_called()

    async def test_timeout_forces_terminate(self) -> None:
        """A pool that exceeds the grace period is terminated."""
        slow = _fake_asyncpg_pool()

        async def hang():
            import asyncio

            await asyncio.sleep(10)

        slow.close = hang
        fast = _fake_asyncpg_pool()
        await close_pools({"slow": slow, "fast": fast}, grace_period=0.05)
        slow.terminate.assert_called_once()
        fast.terminate.assert_not_called()

    async def test_error_terminates_and_continues(self) -> None:
        """A pool raising on close is terminated; remaining pools still close."""
        bad = _fake_asyncpg_pool()
        bad.close = AsyncMock(side_effect=RuntimeError("close failed"))
        good = _fake_asyncpg_pool()
        await close_pools({"bad": bad, "good": good}, grace_period=1.0)
        bad.terminate.assert_called_once()
        good.close.assert_awaited_once()
