"""Security tests: blocked operations, dangerous functions, and EXPLAIN policy."""

import pytest

from pg_mcp.config.settings import SecurityConfig
from pg_mcp.models.errors import SecurityViolationError
from pg_mcp.services.sql_validator import SQLValidator


class TestWriteOperations:
    """All data-modification statements must be rejected by default."""

    @pytest.fixture
    def validator(self) -> SQLValidator:
        return SQLValidator(config=SecurityConfig())

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO users VALUES (1)",
            "UPDATE users SET name = 'x'",
            "DELETE FROM users",
            "TRUNCATE users",
            "DROP TABLE users",
            "ALTER TABLE users ADD COLUMN x int",
            "CREATE TABLE t (id int)",
            "CREATE INDEX idx ON users (id)",
            "DROP DATABASE blog_small",
            "REVOKE ALL ON users FROM public",
            "VACUUM",
            "ANALYZE users",
            "CALL some_procedure()",
            "LISTEN channel",
            "NOTIFY channel",
        ],
    )
    def test_write_statements_rejected(self, validator: SQLValidator, sql: str) -> None:
        ok, _ = validator.validate(sql)
        assert not ok, sql

    def test_write_operations_raise_security_violation(self, validator: SQLValidator) -> None:
        """validate_or_raise raises SecurityViolationError with context."""
        with pytest.raises(SecurityViolationError, match="not allowed"):
            validator.validate_or_raise("DELETE FROM users")


class TestDangerousFunctions:
    """Blocked functions must be rejected even inside valid SELECT shapes."""

    @pytest.fixture
    def validator(self) -> SQLValidator:
        return SQLValidator(config=SecurityConfig())

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT pg_sleep(10)",
            "SELECT pg_terminate_backend(123)",
            "SELECT pg_cancel_backend(123)",
            "SELECT pg_reload_conf()",
            "SELECT pg_read_file('/etc/passwd')",
            "SELECT pg_read_binary_file('/etc/shadow')",
            "SELECT * FROM pg_ls_dir('/')",
            "SELECT lo_import('/tmp/f')",
            "SELECT lo_export(1, '/tmp/f')",
            "SELECT pg_stat_file('/tmp')",
            "SELECT dblink_connect('h=evil')",
            "SELECT pg_write_file('/tmp/x', 'data')",
            "SELECT length(pg_read_file('/etc/hosts'))",
        ],
    )
    def test_dangerous_function_rejected(self, validator: SQLValidator, sql: str) -> None:
        ok, _ = validator.validate(sql)
        assert not ok, sql

    def test_custom_blocked_functions(self) -> None:
        """Operator-supplied function blacklist extends the builtin one."""
        config = SecurityConfig(blocked_functions="my_evil_fn")
        validator = SQLValidator(config=config)
        ok, msg = validator.validate("SELECT my_evil_fn(1)")
        assert not ok
        assert "my_evil_fn" in (msg or "")

    def test_safe_functions_allowed(self, validator: SQLValidator) -> None:
        """Ordinary functions keep working."""
        ok, _ = validator.validate(
            "SELECT count(*), upper(name), coalesce(age, 0) FROM users GROUP BY name"
        )
        assert ok


class TestExplainPolicy:
    """EXPLAIN is denied by default and allowed when configured."""

    def _validator(self, allow: bool) -> SQLValidator:
        return SQLValidator(config=SecurityConfig(), allow_explain=allow)

    def test_explain_denied_by_default(self) -> None:
        ok, msg = self._validator(False).validate("EXPLAIN SELECT * FROM users")
        assert not ok
        assert "EXPLAIN" in (msg or "")

    def test_explain_allowed_when_configured(self) -> None:
        ok, _ = self._validator(True).validate("EXPLAIN SELECT * FROM users")
        assert ok

    def test_explain_analyze_also_gated(self) -> None:
        """EXPLAIN ANALYZE follows the same switch."""
        assert not self._validator(False).validate("EXPLAIN ANALYZE SELECT 1")[0]
        assert self._validator(True).validate("EXPLAIN ANALYZE SELECT 1")[0]

    def test_other_commands_never_allowed(self) -> None:
        """EXPLAIN permission does not open the door to other commands."""
        ok, _ = self._validator(True).validate("VACUUM users")
        assert not ok


class TestTableColumnGlobalBlocks:
    """Global blocked tables/columns (SecurityConfig) are enforced."""

    def _validator(self) -> SQLValidator:
        config = SecurityConfig(
            blocked_tables="internal_secrets, payroll",
            blocked_columns="password, ssn",
        )
        return SQLValidator(
            config=config,
            blocked_tables=config.blocked_tables,
            blocked_columns=config.blocked_columns,
        )

    def test_global_blocked_table(self) -> None:
        ok, msg = self._validator().validate("SELECT * FROM internal_secrets")
        assert not ok
        assert "internal_secrets" in (msg or "")

    def test_global_blocked_column(self) -> None:
        ok, msg = self._validator().validate("SELECT password FROM users")
        assert not ok
        assert "password" in (msg or "")

    def test_qualified_column_blocked(self) -> None:
        ok, _ = self._validator().validate("SELECT users.password FROM users")
        assert not ok

    def test_subquery_reference_blocked(self) -> None:
        ok, _ = self._validator().validate("SELECT * FROM (SELECT ssn FROM employees) t")
        assert not ok
