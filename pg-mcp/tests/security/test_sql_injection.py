"""Security tests: SQL injection and evasion attempt rejection.

These tests verify that SQLValidator rejects common SQL injection patterns
that could slip past naive allow-listing of SELECT statements.
"""

import pytest

from pg_mcp.config.settings import SecurityConfig
from pg_mcp.services.sql_validator import SQLValidator


@pytest.fixture
def validator() -> SQLValidator:
    """Validator with default security config."""
    return SQLValidator(config=SecurityConfig())


class TestSQLInjectionAttempts:
    """Injection payloads must never validate as safe."""

    def test_stacked_query_injection(self, validator: SQLValidator) -> None:
        """SELECT; DROP stacking is rejected."""
        ok, msg = validator.validate("SELECT * FROM users; DROP TABLE users;--")
        assert not ok
        assert "Multiple statements" in (msg or "")

    def test_stacked_delete_injection(self, validator: SQLValidator) -> None:
        """SELECT; DELETE stacking is rejected."""
        ok, _ = validator.validate("SELECT 1; DELETE FROM users WHERE 1=1")
        assert not ok

    def test_union_branches_subject_to_blocklists(self) -> None:
        """UNION shape is allowed by design, but every branch is subject to
        the same blocked-table/column checks."""
        config = SecurityConfig(blocked_tables="passwords")
        validator = SQLValidator(config=config, blocked_tables=config.blocked_tables)
        ok, msg = validator.validate("SELECT name FROM users UNION SELECT * FROM passwords")
        assert not ok
        assert "passwords" in (msg or "")

    def test_union_shape_allowed_within_blocklists(self, validator: SQLValidator) -> None:
        """UNION between allowed tables passes (documented design)."""
        ok, _ = validator.validate("SELECT a FROM t1 UNION SELECT b FROM t2")
        assert ok

    def test_or_tautology(self, validator: SQLValidator) -> None:
        """OR 1=1 tautology is syntactically valid SELECT - allowed here but
        documented: read-only execution + row limits bound the damage."""
        ok, _ = validator.validate("SELECT * FROM users WHERE id = 1 OR 1=1")
        assert ok  # still read-only; defense-in-depth is the readonly txn

    def test_comment_evasion(self, validator: SQLValidator) -> None:
        """Inline comments cannot smuggle writes."""
        ok, _ = validator.validate("SELECT * /* c */ FROM users; -- DROP TABLE x")
        assert not ok

    def test_pg_sleep_via_select(self, validator: SQLValidator) -> None:
        """Dangerous functions embedded in SELECT are blocked."""
        ok, msg = validator.validate("SELECT pg_sleep(100)")
        assert not ok
        assert "pg_sleep" in (msg or "")

    def test_dblink_access(self, validator: SQLValidator) -> None:
        """dblink calls are blocked."""
        ok, _ = validator.validate("SELECT * FROM dblink('host=-x dbname=y', 'SELECT 1')")
        assert not ok

    def test_file_read_via_function(self, validator: SQLValidator) -> None:
        """Server file reads are blocked."""
        ok, _ = validator.validate("SELECT pg_read_file('/etc/passwd')")
        assert not ok

    def test_copy_program(self, validator: SQLValidator) -> None:
        """COPY ... PROGRAM is not a SELECT and is rejected."""
        ok, _ = validator.validate("COPY users TO PROGRAM 'sh -c evil'")
        assert not ok

    def test_insert_via_cte(self, validator: SQLValidator) -> None:
        """Write hidden in a WITH clause is rejected (regression: bypass)."""
        ok, msg = validator.validate(
            "WITH w AS (INSERT INTO logs VALUES (1) RETURNING *) SELECT * FROM w"
        )
        assert not ok
        assert "WITH clause" in (msg or "")

    def test_delete_via_cte(self, validator: SQLValidator) -> None:
        """DELETE hidden in a WITH clause is rejected."""
        ok, _ = validator.validate("WITH d AS (DELETE FROM users) SELECT * FROM d")
        assert not ok

    def test_update_via_cte(self, validator: SQLValidator) -> None:
        """UPDATE hidden in a WITH clause is rejected."""
        ok, _ = validator.validate("WITH u AS (UPDATE users SET admin = true) SELECT * FROM u")
        assert not ok

    def test_legitimate_ctes_still_allowed(self, validator: SQLValidator) -> None:
        """Read-only CTEs keep working."""
        ok, _ = validator.validate(
            "WITH top AS (SELECT user_id, count(*) c FROM actions GROUP BY user_id) "
            "SELECT * FROM top ORDER BY c DESC LIMIT 10"
        )
        assert ok

    def test_do_statement(self, validator: SQLValidator) -> None:
        """DO anonymous code blocks are rejected."""
        ok, _ = validator.validate("DO $$ BEGIN DELETE FROM users; END $$;")
        assert not ok

    def test_set_statement(self, validator: SQLValidator) -> None:
        """Session tampering via SET is rejected."""
        ok, _ = validator.validate("SET search_path TO pg_catalog")
        assert not ok

    def test_grant_privilege_escalation(self, validator: SQLValidator) -> None:
        """GRANT is rejected."""
        ok, _ = validator.validate("GRANT ALL ON users TO attacker")
        assert not ok

    def test_transaction_control(self, validator: SQLValidator) -> None:
        """Transaction control statements are rejected."""
        for sql in ("BEGIN", "COMMIT", "ROLLBACK"):
            ok, _ = validator.validate(sql)
            assert not ok, sql

    def test_case_evasion(self, validator: SQLValidator) -> None:
        """Mixed-case write keywords are still caught."""
        ok, _ = validator.validate("DeLeTe FrOm users")
        assert not ok

    def test_whitespace_evasion(self, validator: SQLValidator) -> None:
        """Newlines/tabs cannot hide a write statement."""
        ok, _ = validator.validate("\n\tDELETE\nFROM\tusers\n")
        assert not ok

    def test_empty_and_whitespace(self, validator: SQLValidator) -> None:
        """Empty input is rejected as parse error."""
        for sql in ("", "   ", ";"):
            ok, _ = validator.validate(sql)
            assert not ok, repr(sql)
