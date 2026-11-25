"""Heavy adversarial/security tests for the SQL compiler and the read-only executor.

Covers: DML/DDL/command injection, forbidden functions, internal sqlite tables,
RBAC bypass attempts (aliases, subqueries, UNION, CTE shadowing), multi-statement
smuggling, unparseable input, read-only enforcement, resource-exhaustion bombs,
row caps, and executor error paths.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from nl2sql.compiler.sql_compiler import SQLSecurityError, compile_and_validate_sql
from nl2sql.evaluation.evaluator import execute_sqlite_readonly
from nl2sql.db.introspect import introspect_database

from conftest import SHOP_DDL, create_sqlite


def rows_none(msg: str) -> bool:
    return bool(msg)


@pytest.fixture(scope="module")
def sec_db(tmp_path_factory: Path) -> Path:
    return create_sqlite(tmp_path_factory.mktemp("sec") / "shop.sqlite", SHOP_DDL)


@pytest.fixture(scope="module")
def sec_bundle(sec_db: Path):
    return introspect_database(f"sqlite:///{sec_db.resolve()}", db_id="shop")


DANGEROUS_STATEMENTS = [
    "INSERT INTO customers VALUES (9, 'Eve', 'TX')",
    "UPDATE customers SET name = 'hacked'",
    "DELETE FROM customers",
    "DROP TABLE customers",
    "ALTER TABLE customers ADD COLUMN evil TEXT",
    "CREATE TABLE evil (x INTEGER)",
    "TRUNCATE TABLE customers",
    "REPLACE INTO customers VALUES (1, 'x', 'y')",
    "ATTACH DATABASE '/tmp/evil.db' AS e",
    "DETACH DATABASE e",
    "PRAGMA journal_mode = DELETE",
    "VACUUM",
    "EXPLAIN SELECT * FROM customers",
    "SELECT 1; DROP TABLE customers",
    "SELECT 1; UPDATE customers SET name = 'x'",
    "WITH x AS (SELECT 1) DELETE FROM customers",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT sp1",
]

FORBIDDEN_FUNCTION_SQL = [
    "SELECT load_extension('lib evil')",
    "SELECT sqlite_load_extension('/tmp/x')",
    "SELECT pg_sleep(10) FROM customers",
    "SELECT lo_import('/etc/passwd')",
    "SELECT lo_export(1, '/tmp/x')",
]


@pytest.mark.parametrize("sql", DANGEROUS_STATEMENTS)
def test_dangerous_statements_rejected(sql: str, sec_bundle) -> None:
    with pytest.raises(SQLSecurityError):
        compile_and_validate_sql(sql, sec_bundle)


@pytest.mark.parametrize("sql", FORBIDDEN_FUNCTION_SQL)
def test_forbidden_functions_rejected(sql: str, sec_bundle) -> None:
    with pytest.raises(SQLSecurityError):
        compile_and_validate_sql(sql, sec_bundle)


UNPARSEABLE = [
    "COMPLETE GARBAGE !@#",
    "SELECT FROM WHERE",
    "SELECT * FROM",
    "'''",
    "",
    "   ",
    "\n\t\n",
    "SELECT * FROM customers WHERE",  # dangling clause
]


@pytest.mark.parametrize("sql", UNPARSEABLE)
def test_unparseable_sql_rejected_cleanly(sql: str, sec_bundle) -> None:
    with pytest.raises(SQLSecurityError):
        compile_and_validate_sql(sql, sec_bundle)


@pytest.mark.parametrize("table", ["sqlite_master", "sqlite_schema", "sqlite_temp_master", "ghost_table"])
def test_internal_or_unknown_tables_rejected(table: str, sec_bundle) -> None:
    with pytest.raises(SQLSecurityError):
        compile_and_validate_sql(f"SELECT * FROM {table}", sec_bundle)


class TestRbacBypassAttempts:
    def test_alias_does_not_bypass(self, sec_bundle) -> None:
        with pytest.raises(SQLSecurityError, match="RBAC"):
            compile_and_validate_sql(
                "SELECT c.name FROM customers c WHERE c.state = 'CA'", sec_bundle, allowed_tables={"orders"}
            )

    def test_subquery_does_not_bypass(self, sec_bundle) -> None:
        with pytest.raises(SQLSecurityError, match="RBAC"):
            compile_and_validate_sql(
                "SELECT * FROM customers WHERE id IN (SELECT customer_id FROM orders)",
                sec_bundle,
                allowed_tables={"customers"},
            )

    def test_union_second_arm_enforced(self, sec_bundle) -> None:
        with pytest.raises(SQLSecurityError, match="RBAC"):
            compile_and_validate_sql(
                "SELECT id FROM customers UNION SELECT id FROM orders",
                sec_bundle,
                allowed_tables={"customers"},
            )

    def test_cte_shadowing_is_allowed_and_safe(self, sec_bundle) -> None:
        # A CTE that shadows a real table only references its own subquery, so no
        # RBAC-restricted data is touched and compilation must succeed.
        res = compile_and_validate_sql(
            "WITH customers AS (SELECT 1 AS id) SELECT id FROM customers",
            sec_bundle,
            allowed_tables={"orders"},
        )
        assert res.used_tables == []
        assert res.sql.upper().startswith("WITH")

    def test_case_insensitive_table_resolution(self, sec_bundle) -> None:
        res = compile_and_validate_sql("SELECT * FROM CUSTOMERS", sec_bundle)
        assert res.used_tables == ["customers"]

    def test_allowlist_matching_is_case_insensitive(self, sec_bundle) -> None:
        res = compile_and_validate_sql("SELECT * FROM customers", sec_bundle, allowed_tables={"CUSTOMERS"})
        assert res.used_tables == ["customers"]


class TestReadOnlyExecutor:
    def test_update_rejected_and_data_intact(self, sec_db: Path) -> None:
        before = sqlite3.connect(sec_db).execute("SELECT COUNT(*), name FROM customers").fetchall()
        ok, rows, _, msg = execute_sqlite_readonly(sec_db, "UPDATE customers SET name = 'hacked'")
        assert ok is False and rows is None
        assert "readonly" in msg.lower() or "write" in msg.lower()
        after = sqlite3.connect(sec_db).execute("SELECT COUNT(*), name FROM customers").fetchall()
        assert before == after

    def test_delete_rejected(self, sec_db: Path) -> None:
        ok, _, _, _ = execute_sqlite_readonly(sec_db, "DELETE FROM customers")
        assert ok is False

    def test_drop_rejected(self, sec_db: Path) -> None:
        ok, _, _, _ = execute_sqlite_readonly(sec_db, "DROP TABLE customers")
        assert ok is False

    def test_attach_cannot_create_file(self, sec_db: Path, tmp_path: Path) -> None:
        evil = tmp_path / "evil.db"
        ok, _, _, _ = execute_sqlite_readonly(sec_db, f"ATTACH DATABASE '{evil}' AS e")
        assert ok is False
        assert not evil.exists()

    def test_with_prefixed_write_blocked_in_depth(self, sec_db: Path) -> None:
        # First keyword is WITH, so the statement gate passes; the connection-level
        # query_only guard must still block the write.
        before = sqlite3.connect(sec_db).execute("SELECT COUNT(*) FROM customers").fetchone()
        ok, _, _, msg = execute_sqlite_readonly(sec_db, "WITH x AS (SELECT 1) DELETE FROM customers")
        assert ok is False and rows_none(msg)
        after = sqlite3.connect(sec_db).execute("SELECT COUNT(*) FROM customers").fetchone()
        assert before == after

    def test_multi_statement_rejected(self, sec_db: Path) -> None:
        ok, _, _, _ = execute_sqlite_readonly(sec_db, "SELECT 1; SELECT 2")
        assert ok is False

    def test_recursive_cte_bomb_is_time_limited(self, sec_db: Path) -> None:
        bomb = (
            "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt) "
            "SELECT COUNT(*) FROM cnt"
        )
        t0 = time.perf_counter()
        ok, rows, elapsed_ms, msg = execute_sqlite_readonly(sec_db, bomb, timeout_sec=2.0)
        wall = time.perf_counter() - t0
        assert ok is False and rows is None
        assert wall < 5.0, f"deadline not enforced, wall={wall:.1f}s"
        assert elapsed_ms < 5000.0

    def test_row_cap_enforced(self, sec_db: Path) -> None:
        sql = (
            "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt LIMIT 500) "
            "SELECT x FROM cnt"
        )
        ok, rows, _, _ = execute_sqlite_readonly(sec_db, sql, max_rows=7)
        assert ok is True
        assert len(rows or []) == 7

    def test_missing_db_file_fails_cleanly(self, tmp_path: Path) -> None:
        ok, rows, _, _ = execute_sqlite_readonly(tmp_path / "does_not_exist.sqlite", "SELECT 1")
        assert ok is False and rows is None

    def test_invalid_sql_fails_cleanly(self, sec_db: Path) -> None:
        ok, rows, _, msg = execute_sqlite_readonly(sec_db, "NOT SQL AT ALL")
        assert ok is False and rows is None and msg

    def test_huge_union_all_bounded(self, sec_db: Path) -> None:
        # 2000-row result set via UNION ALL must respect max_rows and stay fast.
        sql = "SELECT 1 AS v FROM customers UNION ALL SELECT 2 FROM customers UNION ALL SELECT 3 FROM customers LIMIT 2000"
        ok, rows, _, _ = execute_sqlite_readonly(sec_db, sql, max_rows=5)
        assert ok is True and len(rows or []) == 5


class TestCompilerLimitSemantics:
    def test_limit_injected_when_missing(self, sec_bundle) -> None:
        res = compile_and_validate_sql("SELECT * FROM customers", sec_bundle)
        assert "LIMIT" in res.sql.upper()

    def test_existing_limit_preserved(self, sec_bundle) -> None:
        res = compile_and_validate_sql("SELECT * FROM customers LIMIT 5", sec_bundle)
        assert "5" in res.sql and "200" not in res.sql

    def test_order_by_survives_limit_injection(self, sec_bundle) -> None:
        res = compile_and_validate_sql("SELECT name FROM customers ORDER BY name", sec_bundle)
        upper = res.sql.upper()
        assert "ORDER BY" in upper and "LIMIT" in upper
        # ORDER BY must come before the injected LIMIT
        assert upper.index("ORDER BY") < upper.index("LIMIT")

    def test_aggregate_query_not_limited(self, sec_bundle) -> None:
        res = compile_and_validate_sql("SELECT COUNT(*) FROM customers", sec_bundle)
        assert "LIMIT" not in res.sql.upper()

    def test_union_top_level_not_limited_but_valid(self, sec_bundle) -> None:
        res = compile_and_validate_sql("SELECT id FROM customers UNION SELECT id FROM orders", sec_bundle)
        assert "UNION" in res.sql.upper()
        # documented behavior: LIMIT guard applies to plain Select, not Union
        assert "LIMIT" not in res.sql.upper()

    def test_default_limit_zero_disables_injection(self, sec_bundle) -> None:
        res = compile_and_validate_sql("SELECT * FROM customers", sec_bundle, default_limit=0)
        assert "LIMIT" not in res.sql.upper()

    def test_subquery_table_validated(self, sec_bundle) -> None:
        with pytest.raises(SQLSecurityError, match="unknown table"):
            compile_and_validate_sql(
                "SELECT * FROM customers WHERE id IN (SELECT ghost_id FROM ghost_table)", sec_bundle
            )
