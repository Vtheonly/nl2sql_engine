"""Runtime engine tests: Layer 0-7 answer flow, abstention gate, template cache,
RBAC filtering, and the FastAPI surface."""
from __future__ import annotations

from pathlib import Path

import pytest

from nl2sql.compiler.sql_compiler import SQLSecurityError, compile_and_validate_sql
from nl2sql.db.value_index import ValueIndex
from nl2sql.engine import NL2SQLEngine, QueryRequest, create_app
from nl2sql.models.graphix_parser import GraphixCodeT5Parser


def _build_engine(tmp_path: Path, cfg) -> NL2SQLEngine:
    engine = NL2SQLEngine(cfg, artifacts_dir=Path(cfg.training.output_dir) / "artifacts", device="cpu")
    return engine


def _register(engine: NL2SQLEngine, shop_db: Path, bundle, tmp_path: Path) -> None:
    v_idx = ValueIndex(tmp_path / "engine_vi.sqlite")
    v_idx.build_from_db(f"sqlite:///{shop_db.resolve()}", bundle)
    engine.db_registry["shop"] = (bundle, v_idx, shop_db)


def test_engine_answer_success_and_cache(
    tmp_path: Path, engine_config, shop_db: Path, shop_bundle, monkeypatch
) -> None:
    engine = _build_engine(tmp_path, engine_config)
    _register(engine, shop_db, shop_bundle, tmp_path)

    gold = "SELECT COUNT(*) FROM tickets WHERE status = 'open'"
    monkeypatch.setattr(
        GraphixCodeT5Parser, "generate_candidates",
        lambda self, source_text, bundle, tables, value_matches, **kw: [gold, "SELECT 1"],
    )

    resp = engine.answer(db_id="shop", question="How many open tickets for Alice?")
    assert not resp.abstained and resp.sql is not None
    assert "COUNT(*)" in resp.sql.upper().replace(" ", "")
    assert "tickets" in resp.sql
    assert resp.confidence >= engine_config.models.abstention_threshold
    assert resp.linked_tables and set(resp.linked_tables) <= {"customers", "orders", "tickets"}
    assert 0 < resp.latency_ms < 5000
    assert resp.cache_hit is False

    # Second identical question (no digits/strings) hits the fast-path template cache
    resp2 = engine.answer(db_id="shop", question="How many open tickets for Alice?")
    assert resp2.cache_hit is True and resp2.sql == resp.sql
    assert resp2.confidence == 0.99

    # Executing variant returns materialized rows
    resp3 = engine.answer(db_id="shop", question="How many open tickets for Alice?", execute=True)
    assert resp3.rows == [[1]]


def test_engine_abstains_when_all_candidates_invalid(
    tmp_path: Path, engine_config, shop_db: Path, shop_bundle, monkeypatch
) -> None:
    engine = _build_engine(tmp_path, engine_config)
    _register(engine, shop_db, shop_bundle, tmp_path)

    monkeypatch.setattr(
        GraphixCodeT5Parser, "generate_candidates",
        lambda self, source_text, bundle, tables, value_matches, **kw: ["DELETE FROM customers", "SELECT * FROM ghost"],
    )
    resp = engine.answer(db_id="shop", question="How many open tickets for Alice?")
    assert resp.abstained is True and resp.sql is None
    assert resp.reason


def test_engine_abstains_on_failed_execution(
    tmp_path: Path, engine_config, shop_db: Path, shop_bundle, monkeypatch
) -> None:
    engine = _build_engine(tmp_path, engine_config)
    _register(engine, shop_db, shop_bundle, tmp_path)

    # Valid schema-wise but broken at runtime: unknown column
    monkeypatch.setattr(
        GraphixCodeT5Parser, "generate_candidates",
        lambda self, source_text, bundle, tables, value_matches, **kw: ["SELECT missing_col FROM tickets"],
    )
    resp = engine.answer(db_id="shop", question="How many open tickets for Alice?")
    assert resp.abstained is True


def test_engine_rbac_blocks_disallowed_tables(
    tmp_path: Path, engine_config, shop_db: Path, shop_bundle, monkeypatch
) -> None:
    engine = _build_engine(tmp_path, engine_config)
    _register(engine, shop_db, shop_bundle, tmp_path)

    gold = "SELECT COUNT(*) FROM tickets WHERE status = 'open'"
    monkeypatch.setattr(
        GraphixCodeT5Parser, "generate_candidates",
        lambda self, source_text, bundle, tables, value_matches, **kw: [gold],
    )
    resp = engine.answer(db_id="shop", question="How many open tickets?", allowed_tables=["customers"])
    assert resp.abstained is True  # tickets is outside the caller's allowlist


def test_engine_unknown_database_raises(engine_config) -> None:
    engine = NL2SQLEngine(engine_config, artifacts_dir=Path("/nonexistent-artifacts"), device="cpu")
    with pytest.raises(KeyError, match="not registered"):
        engine.answer(db_id="ghost", question="anything")


def test_fastapi_query_endpoint(
    tmp_path: Path, engine_config, shop_db: Path, shop_bundle, monkeypatch
) -> None:
    from fastapi.testclient import TestClient

    engine = _build_engine(tmp_path, engine_config)
    _register(engine, shop_db, shop_bundle, tmp_path)

    gold = "SELECT COUNT(*) FROM tickets WHERE status = 'open'"
    monkeypatch.setattr(
        GraphixCodeT5Parser, "generate_candidates",
        lambda self, source_text, bundle, tables, value_matches, **kw: [gold],
    )

    app = create_app(engine)
    client = TestClient(app)

    resp = client.post("/query", json=QueryRequest(db_id="shop", question="How many open tickets?").model_dump())
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is False and "tickets" in body["sql"]

    not_found = client.post("/query", json={"db_id": "ghost", "question": "?"})
    assert not_found.status_code == 404


def test_compiler_security_function_blocklist(shop_bundle) -> None:
    with pytest.raises(SQLSecurityError, match="(?i)forbidden"):
        compile_and_validate_sql("SELECT load_extension('x') FROM customers", shop_bundle)
    with pytest.raises(SQLSecurityError):
        compile_and_validate_sql("SELECT * FROM customers; DROP TABLE customers", shop_bundle)


def test_compiler_dialect_transpilation(shop_bundle) -> None:
    res = compile_and_validate_sql(
        "SELECT c.name FROM customers c WHERE c.state = 'CA' LIMIT 5", shop_bundle, target_dialect="postgres"
    )
    assert "'" in res.sql  # string literal survives
    lowered = res.sql.lower()
    assert "from customers" in lowered and "limit 5" in lowered
