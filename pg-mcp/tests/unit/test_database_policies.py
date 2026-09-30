"""Unit tests for per-database access policies.

Covers: DatabaseConfig policy fields, DatabaseAccessPolicy, SQLValidator.
check_object_access, and orchestrator policy enforcement per target database.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlglot import parse_one

from pg_mcp.config.settings import (
    DatabaseAccessPolicy,
    DatabaseConfig,
    ResilienceConfig,
    ValidationConfig,
)
from pg_mcp.models.errors import SecurityViolationError
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_validator import SQLValidator


class TestDatabaseConfigPolicy:
    """Tests for per-database policy fields on DatabaseConfig."""

    def test_policy_defaults_empty(self) -> None:
        """Policy fields default to empty lists."""
        config = DatabaseConfig(_env_file=None)
        assert config.blocked_tables == []
        assert config.blocked_columns == []

    def test_policy_from_lists(self) -> None:
        """Policy fields accept lists."""
        config = DatabaseConfig(_env_file=None, blocked_tables=["t1"], blocked_columns=["c1"])
        assert config.blocked_tables == ["t1"]
        assert config.blocked_columns == ["c1"]

    def test_policy_from_comma_string(self) -> None:
        """Policy fields parse comma-separated strings."""
        config = DatabaseConfig(_env_file=None, blocked_tables="a, b", blocked_columns="x ,y")
        assert config.blocked_tables == ["a", "b"]
        assert config.blocked_columns == ["x", "y"]


class TestDatabaseAccessPolicy:
    """Tests for the DatabaseAccessPolicy model."""

    def test_has_restrictions(self) -> None:
        """has_restrictions reflects whether any object is blocked."""
        assert DatabaseAccessPolicy().has_restrictions is False
        assert DatabaseAccessPolicy(blocked_tables=["t"]).has_restrictions is True
        assert DatabaseAccessPolicy(blocked_columns=["c"]).has_restrictions is True


class TestCheckObjectAccess:
    """Tests for SQLValidator.check_object_access."""

    @pytest.fixture
    def validator(self) -> SQLValidator:
        """Validator with no global blocks (policy is passed per-call)."""
        from pg_mcp.config.settings import SecurityConfig

        return SQLValidator(config=SecurityConfig())

    def test_blocked_table_rejected(self, validator: SQLValidator) -> None:
        """Query referencing a policy-blocked table raises."""
        with pytest.raises(SecurityViolationError, match="table 'salaries'"):
            validator.check_object_access("SELECT * FROM salaries", blocked_tables=["salaries"])

    def test_blocked_column_rejected(self, validator: SQLValidator) -> None:
        """Query referencing a policy-blocked column raises."""
        with pytest.raises(SecurityViolationError, match="column 'password'"):
            validator.check_object_access(
                "SELECT password FROM users", blocked_columns=["password"]
            )

    def test_qualified_column_rejected(self, validator: SQLValidator) -> None:
        """Qualified table.column references are caught."""
        with pytest.raises(SecurityViolationError, match="column 'ssn'"):
            validator.check_object_access("SELECT u.ssn FROM users u", blocked_columns=["ssn"])

    def test_cte_reference_caught(self, validator: SQLValidator) -> None:
        """Blocked table referenced inside a CTE is caught."""
        with pytest.raises(SecurityViolationError, match="table 'audit_log'"):
            validator.check_object_access(
                "WITH recent AS (SELECT * FROM audit_log) SELECT * FROM recent",
                blocked_tables=["audit_log"],
            )

    def test_clean_query_passes(self, validator: SQLValidator) -> None:
        """Query touching no blocked objects passes."""
        validator.check_object_access("SELECT id, name FROM users", blocked_tables=["salaries"])

    def test_empty_policy_is_noop(self, validator: SQLValidator) -> None:
        """Empty block lists never reject."""
        validator.check_object_access("SELECT * FROM anything")

    def test_case_insensitive(self, validator: SQLValidator) -> None:
        """Matching is case-insensitive on both sides."""
        with pytest.raises(SecurityViolationError):
            validator.check_object_access("SELECT * FROM SALARIES", blocked_tables=["Salaries"])

    def test_invalid_sql_raises_parse_error(self, validator: SQLValidator) -> None:
        """Unparseable SQL raises SQLParseError."""
        from pg_mcp.models.errors import SQLParseError

        with pytest.raises(SQLParseError):
            validator.check_object_access("SELECT * FROM (", blocked_tables=["x"])

    def test_table_extraction_consistency(
        self, validator: SQLValidator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """check_object_access sees the same tables as extract_tables."""
        sql = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        tables = validator.extract_tables(sql)
        assert "users" in tables and "orders" in tables
        # Blocking any extracted table must reject the query
        for table in tables:
            with pytest.raises(SecurityViolationError):
                validator.check_object_access(sql, blocked_tables=[table])
        assert parse_one(sql, read="postgres") is not None  # sanity: sqlglot parses


class TestOrchestratorPolicyEnforcement:
    """Tests for orchestrator-level per-database policy enforcement."""

    @pytest.fixture
    def mock_schema(self) -> MagicMock:
        """Schema mock with table count attribute."""
        schema = MagicMock()
        schema.tables = []
        return schema

    def _orchestrator(
        self,
        policies: dict[str, DatabaseAccessPolicy] | None,
        validator: MagicMock,
        generator: AsyncMock,
    ) -> QueryOrchestrator:
        """Build orchestrator with mocked deps and two databases."""
        cache = MagicMock()
        cache.get.return_value = MagicMock(tables=[])
        return QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executors={"db_safe": AsyncMock(), "db_locked": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=cache,
            pools={"db_safe": MagicMock(), "db_locked": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=0),
            validation_config=ValidationConfig(enabled=False),
            database_policies=policies,
        )

    @pytest.mark.asyncio
    async def test_policy_blocks_query_on_restricted_database(self, mock_schema: MagicMock) -> None:
        """A query hitting a policy-blocked table fails on that database only."""
        validator = MagicMock()
        validator.validate_or_raise.return_value = None
        validator.check_object_access.side_effect = SecurityViolationError(
            "Access to table 'salaries' is not allowed for this database"
        )
        generator = AsyncMock()
        generator.generate.return_value = "SELECT * FROM salaries;"

        orchestrator = self._orchestrator(
            policies={
                "db_locked": DatabaseAccessPolicy(blocked_tables=["salaries"]),
                "db_safe": DatabaseAccessPolicy(),
            },
            validator=validator,
            generator=generator,
        )

        request = QueryRequest(question="q", database="db_locked", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "security_violation"
        assert "salaries" in response.error.message
        validator.check_object_access.assert_called_once()

    @pytest.mark.asyncio
    async def test_same_sql_allowed_on_unrestricted_database(self, mock_schema: MagicMock) -> None:
        """Identical SQL succeeds on a database without restrictions."""
        validator = MagicMock()
        validator.validate_or_raise.return_value = None
        generator = AsyncMock()
        generator.generate.return_value = "SELECT * FROM salaries;"

        orchestrator = self._orchestrator(
            policies={"db_locked": DatabaseAccessPolicy(blocked_tables=["salaries"])},
            validator=validator,
            generator=generator,
        )

        request = QueryRequest(question="q", database="db_safe", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is True
        validator.check_object_access.assert_not_called()

    @pytest.mark.asyncio
    async def test_policy_violation_feeds_back_to_llm(self) -> None:
        """Policy violations participate in the retry-with-feedback loop."""
        validator = MagicMock()
        # First attempt violates policy, second attempt is clean
        validator.validate_or_raise.return_value = None
        validator.check_object_access.side_effect = [
            SecurityViolationError("blocked table 'salaries'"),
            None,
        ]
        generator = AsyncMock()
        generator.generate.side_effect = [
            "SELECT * FROM salaries;",
            "SELECT COUNT(*) FROM users;",
        ]

        cache = MagicMock()
        cache.get.return_value = MagicMock(tables=[])
        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executors={"db_locked": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=cache,
            pools={"db_locked": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=2, retry_delay=0.1),
            validation_config=ValidationConfig(enabled=False),
            database_policies={"db_locked": DatabaseAccessPolicy(blocked_tables=["salaries"])},
        )

        request = QueryRequest(question="q", database="db_locked", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert response.generated_sql == "SELECT COUNT(*) FROM users;"
        # The retry carried the policy violation message as feedback
        retry_call = generator.generate.call_args_list[1]
        assert "salaries" in retry_call.kwargs["error_feedback"]
