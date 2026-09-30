"""Security tests: per-database access policy enforcement.

Verifies that database-scoped table/column restrictions are enforced in the
orchestrator pipeline and cannot be bypassed via the request path.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import (
    DatabaseAccessPolicy,
    DatabaseConfig,
    ResilienceConfig,
    ValidationConfig,
)
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_validator import SQLValidator


def _orchestrator(
    policies: dict[str, DatabaseAccessPolicy] | None,
    generate_sides: list[str],
    validator: SQLValidator,
) -> QueryOrchestrator:
    """Real SQLValidator + scripted LLM outputs, everything else mocked."""
    generator = AsyncMock()
    generator.generate.side_effect = generate_sides
    cache = MagicMock()
    cache.get.return_value = MagicMock(tables=[])
    return QueryOrchestrator(
        sql_generator=generator,
        sql_validator=validator,
        sql_executors={
            "public_db": AsyncMock(),
            "hr_db": AsyncMock(),
        },
        result_validator=MagicMock(),
        schema_cache=cache,
        pools={"public_db": MagicMock(), "hr_db": MagicMock()},
        resilience_config=ResilienceConfig(max_retries=0, retry_delay=0.1),
        validation_config=ValidationConfig(enabled=False),
        database_policies=policies,
    )


class TestPerDatabaseIsolation:
    """The same SQL must be judged differently per target database."""

    @pytest.fixture
    def validator(self) -> SQLValidator:
        return SQLValidator(config=ValidationSecConfig())

    @pytest.mark.asyncio
    async def test_sensitive_table_blocked_only_on_policy_database(self) -> None:
        """A table blocked on hr_db still queries fine on public_db."""
        validator = SQLValidator(config=SecCfg())
        policies = {"hr_db": DatabaseAccessPolicy(blocked_tables=["salaries"])}

        # hr_db: rejected
        orch = _orchestrator(policies, ["SELECT * FROM salaries;"], validator)
        response = await orch.execute_query(
            QueryRequest(question="q", database="hr_db", return_type=ReturnType.SQL)
        )
        assert response.success is False
        assert response.error is not None
        assert response.error.code == "security_violation"
        assert "salaries" in response.error.message

        # public_db: allowed
        orch = _orchestrator(policies, ["SELECT * FROM salaries;"], validator)
        response = await orch.execute_query(
            QueryRequest(question="q", database="public_db", return_type=ReturnType.SQL)
        )
        assert response.success is True

    @pytest.mark.asyncio
    async def test_column_policy_enforced_before_execution(self) -> None:
        """Blocked columns are rejected before the executor is reached."""
        validator = SQLValidator(config=SecCfg())
        executor = AsyncMock()
        policies = {"hr_db": DatabaseAccessPolicy(blocked_columns=["ssn"])}

        generator = AsyncMock()
        generator.generate.side_effect = ["SELECT ssn FROM employees;"]
        cache = MagicMock()
        cache.get.return_value = MagicMock(tables=[])
        orch = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executors={"hr_db": executor},
            result_validator=MagicMock(),
            schema_cache=cache,
            pools={"hr_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=0),
            validation_config=ValidationConfig(enabled=False),
            database_policies=policies,
        )
        response = await orch.execute_query(
            QueryRequest(question="q", database="hr_db", return_type=ReturnType.RESULT)
        )
        assert response.success is False
        executor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_policy_database_has_full_access_within_global_rules(
        self,
    ) -> None:
        """Without policies, only global rules apply (regression guard)."""
        validator = SQLValidator(config=SecCfg())
        orch = _orchestrator(None, ["SELECT * FROM anything;"], validator)
        response = await orch.execute_query(
            QueryRequest(question="q", database="public_db", return_type=ReturnType.SQL)
        )
        assert response.success is True

    @pytest.mark.asyncio
    async def test_policy_violation_reports_metrics_and_fails_request(self) -> None:
        """A policy violation counts as a rejected SQL and fails the request."""
        from prometheus_client import REGISTRY

        validator = SQLValidator(config=SecCfg())
        policies = {"hr_db": DatabaseAccessPolicy(blocked_tables=["salaries"])}
        orch = _orchestrator(policies, ["SELECT * FROM salaries;"], validator)

        labels = {"reason": "SecurityViolationError"}
        before = REGISTRY.get_sample_value("pg_mcp_sql_rejected_total", labels) or 0.0
        response = await orch.execute_query(
            QueryRequest(question="q", database="hr_db", return_type=ReturnType.SQL)
        )
        after = REGISTRY.get_sample_value("pg_mcp_sql_rejected_total", labels) or 0.0

        assert response.success is False
        assert after - before == 1.0

    def test_database_config_carries_policy(self) -> None:
        """DatabaseConfig exposes per-database blocked objects for wiring."""
        cfg = DatabaseConfig(_env_file=None, name="hr", blocked_tables="salaries, payroll")
        assert cfg.blocked_tables == ["salaries", "payroll"]
        policy = DatabaseAccessPolicy(
            blocked_tables=cfg.blocked_tables, blocked_columns=cfg.blocked_columns
        )
        assert policy.has_restrictions


def SecCfg():
    """Default security config (avoids clashing with fixtures)."""
    from pg_mcp.config.settings import SecurityConfig

    return SecurityConfig()


def ValidationSecConfig():
    """Alias retained for import compatibility."""
    return SecCfg()
