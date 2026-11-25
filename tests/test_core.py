import sqlite3
from pathlib import Path

import pytest
import torch

from nl2sql.compiler.sql_compiler import SQLSecurityError, compile_and_validate_sql
from nl2sql.db.introspect import introspect_database
from nl2sql.db.value_index import ValueIndex
from nl2sql.training.checkpoint import CheckpointManager
from nl2sql.utils.repro import set_seed


def _create_sample_sqlite(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, state TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
        CREATE TABLE tickets (id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT);
        INSERT INTO customers VALUES (1, 'Alice', 'CA'), (2, 'Bob', 'NY');
        INSERT INTO orders VALUES (10, 1, 120.5), (11, 1, 80.0), (12, 2, 50.0);
        INSERT INTO tickets VALUES (100, 1, 'open'), (101, 2, 'closed');
        """
    )
    conn.close()
    return f"sqlite:///{db_path.resolve()}"


def test_introspect_infers_missing_fks_and_value_index(tmp_path: Path) -> None:
    db_file = tmp_path / "shop.sqlite"
    url = _create_sample_sqlite(db_file)
    bundle = introspect_database(url, db_id="shop", infer_missing_fks=True)

    # Declared FKs were omitted intentionally; missing-FK probe must infer orders->customers and tickets->customers
    inferred = {(e.left_table, e.left_column, e.right_table, e.right_column, e.cardinality) for e in bundle.join_edges}
    assert ("orders", "customer_id", "customers", "id", "N:1") in inferred
    assert ("tickets", "customer_id", "customers", "id", "N:1") in inferred

    v_idx = ValueIndex(tmp_path / "vals.sqlite")
    v_idx.build_from_db(url, bundle)
    hits = v_idx.probe("How many open tickets for Alice?")
    hit_vals = {(h.table_name, h.column_name, h.value) for h in hits}
    assert ("tickets", "status", "open") in hit_vals
    assert ("customers", "name", "Alice") in hit_vals


def test_sqlglot_compiler_join_synthesis_fanout_and_security(tmp_path: Path) -> None:
    db_file = tmp_path / "shop.sqlite"
    url = _create_sample_sqlite(db_file)
    bundle = introspect_database(url, db_id="shop", infer_missing_fks=True)

    # 1. Missing ON clause repaired deterministically
    res = compile_and_validate_sql("SELECT customers.name, orders.amount FROM customers JOIN orders", bundle)
    assert "ON" in res.sql.upper()
    assert len(res.join_repairs) == 1

    # 2. Diverging 1:N tables aggregated together: auto CTE pre-aggregation now
    # REWRITES the query (per-parent pre-aggregated CTEs + LEFT JOINs) instead of
    # only flagging it: the ambiguity flag disappears and the warning records the
    # rewrite, so the engine no longer abstains on a provably fixable query.
    fan = compile_and_validate_sql(
        "SELECT c.name, SUM(orders.amount), COUNT(tickets.id) FROM customers c "
        "JOIN orders ON c.id = orders.customer_id JOIN tickets ON c.id = tickets.customer_id GROUP BY c.name",
        bundle,
    )
    assert "WITH" in fan.sql.upper() and "__preagg_" in fan.sql.lower()
    assert fan.ambiguity_flags == []
    assert any("cte pre-aggregation" in w.lower() for w in fan.fanout_warnings)

    # 3. DML / RBAC blocked deterministically
    with pytest.raises(SQLSecurityError):
        compile_and_validate_sql("DROP TABLE orders", bundle)
    with pytest.raises(SQLSecurityError):
        compile_and_validate_sql("SELECT * FROM orders", bundle, allowed_tables={"customers"})


def test_checkpoint_save_and_rng_resume(tmp_path: Path) -> None:
    set_seed(123)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path / "run", keep_last_k=2)

    ckpt_dir = mgr.save("parser", model, opt, None, None, epoch=1, global_step=10, best_metric=0.5, config_dict={}, metrics_history=[], is_best=True)
    expected_next_rand = torch.rand(3)

    # Perturb weights and RNG, then restore
    with torch.no_grad():
        model.weight.zero_()
    _ = torch.rand(5)

    resumed = mgr.resolve_resume_path("parser", "auto")
    assert resumed == ckpt_dir
    mgr.load(resumed, model=model, optimizer=opt)
    restored_next_rand = torch.rand(3)
    assert torch.allclose(expected_next_rand, restored_next_rand)
