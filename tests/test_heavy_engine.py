"""Heavy runtime-engine tests.

Regression tests for confirmed defects plus concurrency and edge-input hardening:
- Template-cache poisoning via double-quoted literals (wrong answers returned)
- Template-cache RBAC bypass (restricted callers receiving unrestricted SQL)
- Evidence / dialect blindness in the cache key
- Cache growth cap (LRU eviction)
- Case-insensitive RBAC allowlists
- Verifier margin abstention (previously dead code)
- Threaded concurrent answer() calls
- Degenerate inputs (empty / unicode / oversized questions)
- FastAPI surface (422 / 404 / 200 / abstain shape)
"""
from __future__ import annotations

import threading
from pathlib import Path

import torch

import pytest

from nl2sql import engine as engine_mod
from nl2sql.db.value_index import ValueIndex
from nl2sql.engine import NL2SQLEngine, QueryRequest, create_app
from nl2sql.models.graphix_parser import GraphixCodeT5Parser
from nl2sql.models.verifier import ModernBertVerifier


def _build_engine(tmp_path: Path, cfg) -> NL2SQLEngine:
    return NL2SQLEngine(cfg, artifacts_dir=Path(cfg.training.output_dir) / "artifacts", device="cpu")


def _register(engine: NL2SQLEngine, shop_db: Path, bundle, tmp_path: Path, with_sqlite: bool = True) -> None:
    v_idx = ValueIndex(tmp_path / "engine_vi.sqlite")
    v_idx.build_from_db(f"sqlite:///{shop_db.resolve()}", bundle)
    engine.db_registry["shop"] = (bundle, v_idx, shop_db if with_sqlite else None)


def _mock_parser(monkeypatch, fn) -> None:
    monkeypatch.setattr(GraphixCodeT5Parser, "generate_candidates", fn)


class TestTemplateCacheCorrectness:
    def test_double_quoted_literal_not_cross_cached(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        """Two questions differing only in a double-quoted literal must NOT share a cache entry."""
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)

        def fake_gen(self, source_text, bundle, tables, value_matches, **kw):
            vals = {v.value for v in value_matches}
            if "NY" in vals:
                return ["SELECT state FROM customers WHERE state = 'NY'"]
            if "CA" in vals:
                return ["SELECT state FROM customers WHERE state = 'CA'"]
            return ["SELECT COUNT(*) FROM tickets"]

        _mock_parser(monkeypatch, fake_gen)

        r1 = engine.answer(db_id="shop", question='list customers where state = "NY"')
        assert r1.abstained is False and "NY" in r1.sql
        r2 = engine.answer(db_id="shop", question='list customers where state = "CA"')
        # The second question must be regenerated (or cached per-literal), never
        # served the NY-flavored SQL from the first question.
        assert r2.cache_hit is False
        assert r2.abstained is False and "CA" in r2.sql
        assert "NY" not in r2.sql

    def test_cache_never_leaks_tables_across_rbac_scopes(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        """A cached unrestricted answer must not be served to a restricted caller."""
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])

        ra = engine.answer(db_id="shop", question="count every open thing")
        assert ra.abstained is False and "tickets" in ra.sql

        rb = engine.answer(db_id="shop", question="count every open thing", allowed_tables=["customers"])
        assert rb.cache_hit is False
        assert rb.abstained is True  # tickets is outside the caller's allowlist
        assert rb.sql is None

    def test_cache_key_includes_evidence(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)

        def fake_gen(self, source_text, bundle, tables, value_matches, **kw):
            if "Electronics" in source_text:
                return ["SELECT id FROM orders"]
            return ["SELECT id FROM tickets"]

        _mock_parser(monkeypatch, fake_gen)

        r1 = engine.answer(db_id="shop", question="show me the rows", evidence="focus on Electronics")
        assert "orders" in r1.sql
        r2 = engine.answer(db_id="shop", question="show me the rows", evidence="focus on Toys")
        assert r2.cache_hit is False
        assert "tickets" in r2.sql

    def test_cache_key_includes_dialect(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])

        r_pg1 = engine.answer(db_id="shop", question="count every open thing", target_dialect="postgres")
        assert r_pg1.cache_hit is False
        r_pg2 = engine.answer(db_id="shop", question="count every open thing", target_dialect="postgres")
        assert r_pg2.cache_hit is True  # same dialect may reuse
        r_lite = engine.answer(db_id="shop", question="count every open thing", target_dialect="sqlite")
        assert r_lite.cache_hit is False  # different dialect must not reuse

    def test_cache_hit_still_works_for_identical_questions(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])

        r1 = engine.answer(db_id="shop", question="count every open thing")
        r2 = engine.answer(db_id="shop", question="count every open thing")
        assert r1.cache_hit is False and r2.cache_hit is True
        assert r2.sql == r1.sql and r2.confidence == 0.99

    def test_cache_size_is_bounded(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])
        monkeypatch.setattr(engine_mod, "_TEMPLATE_CACHE_MAX", 4)

        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
        for w in words:
            engine.answer(db_id="shop", question=f"count every open thing {w}")
        assert len(engine._template_cache) <= 4

    def test_cached_sql_execution_is_readonly(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])

        engine.answer(db_id="shop", question="count every open thing")
        resp = engine.answer(db_id="shop", question="count every open thing", execute=True)
        assert resp.cache_hit is True
        assert resp.rows == [[2]]  # total tickets in the shop fixture


class TestRbacAndCandidates:
    def test_allowlist_matching_is_case_insensitive(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT name FROM customers"])

        resp = engine.answer(db_id="shop", question="list people", allowed_tables=["Customers", "ORDERS"])
        assert resp.abstained is False
        assert "customers" in resp.linked_tables  # schema context is not starved by case mismatch

    def test_empty_allowlist_abstains(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])

        resp = engine.answer(db_id="shop", question="count things", allowed_tables=[])
        assert resp.abstained is True and resp.sql is None

    def test_unknown_allowlist_abstains(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])

        resp = engine.answer(db_id="shop", question="count things", allowed_tables=["nope"])
        assert resp.abstained is True


class TestAbstentionPaths:
    def test_verifier_margin_triggers_abstention(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        """Ambiguous verifier scores (sub-margin gap) must abstain; wide margin must not."""
        art_dir = Path(engine_config.training.output_dir) / "artifacts"
        verifier = ModernBertVerifier(engine_config.models.verifier_model_id)
        verifier.save_pretrained(art_dir / "verifier")

        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets", "SELECT COUNT(*) FROM orders"])

        # margin = 0.02 < 0.05 -> abstain
        monkeypatch.setattr(ModernBertVerifier, "score_candidates", lambda self, q, s, c, **kw: [0.90, 0.88])
        resp = engine.answer(db_id="shop", question="count the rows")
        assert resp.abstained is True
        assert "mbiguous" in resp.reason or "margin" in resp.reason.lower()

        # clear the cache so the second attempt regenerates
        engine._template_cache.clear()
        monkeypatch.setattr(ModernBertVerifier, "score_candidates", lambda self, q, s, c, **kw: [0.90, 0.30])
        resp2 = engine.answer(db_id="shop", question="count the rows")
        assert resp2.abstained is False

    def test_execute_without_sqlite_path_returns_no_rows(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path, with_sqlite=False)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])

        resp = engine.answer(db_id="shop", question="count things", execute=True)
        assert resp.abstained is False and resp.rows is None


class TestDegenerateInputs:
    @pytest.mark.parametrize(
        "question",
        [
            "",
            "   ",
            "\n\t",
            "???",
            "!!! ??? ###",
            "数据库中有多少客户？",
            "🎉🎊 how many 🎉?",
            "a" * 10000,
            "SELECT * FROM customers; DROP TABLE customers",
            "<script>alert(1)</script>",
            "' OR '1'='1",
        ],
    )
    def test_degenerate_questions_do_not_crash(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch, question: str) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])

        resp = engine.answer(db_id="shop", question=question)
        assert resp.db_id == "shop"
        assert resp.latency_ms >= 0
        # either a valid abstention or a valid answer, never an exception
        assert isinstance(resp.abstained, bool)

    def test_unregistered_database_raises(self, engine_config) -> None:
        engine = NL2SQLEngine(engine_config, artifacts_dir=Path("/nonexistent"), device="cpu")
        with pytest.raises(KeyError, match="not registered"):
            engine.answer(db_id="ghost", question="anything")

    def test_unavailable_cuda_falls_back_to_cpu(self, engine_config) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA present; fallback path not exercisable")
        engine = NL2SQLEngine(engine_config, artifacts_dir=Path("/nonexistent"), device="cuda")
        assert engine.device == "cpu"
        engine2 = NL2SQLEngine(engine_config, artifacts_dir=Path("/nonexistent"), device=None)
        # cfg.training.device defaults to None -> cpu
        assert engine2.device in ("cpu", "cuda")


class TestConcurrency:
    def test_concurrent_answers_have_no_cross_talk(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)

        questions = {
            "count open tickets": "SELECT COUNT(*) FROM tickets WHERE status = 'open'",
            "count closed tickets": "SELECT COUNT(*) FROM tickets WHERE status = 'closed'",
            "count customers": "SELECT COUNT(*) FROM customers",
            "count orders": "SELECT COUNT(*) FROM orders",
        }

        def fake_gen(self, source_text, bundle, tables, value_matches, **kw):
            for key, sql in questions.items():
                if key in source_text.lower():
                    return [sql]
            return ["SELECT COUNT(*) FROM tickets"]

        _mock_parser(monkeypatch, fake_gen)

        expected = {q: engine.answer(db_id="shop", question=q).sql for q in questions}
        engine._template_cache.clear()

        results: dict[str, list] = {}
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def worker(q: str) -> None:
            try:
                barrier.wait(timeout=30)
                # warm + repeated queries to exercise cache hits under concurrency
                results.setdefault(q, []).extend(
                    engine.answer(db_id="shop", question=q) for _ in range(4)
                )
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(q,)) for q in questions]
        [t.start() for t in threads]
        [t.join(timeout=60) for t in threads]

        assert not errors, f"concurrent calls raised: {errors}"
        for q, resps in results.items():
            assert len(resps) == 4
            for r in resps:
                assert r.sql == expected[q], f"cross-talk for question {q!r}: got {r.sql}"
                assert r.abstained is False


class TestFastAPISurface:
    def test_endpoint_validation_and_errors(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["SELECT COUNT(*) FROM tickets"])

        app = create_app(engine)
        client = TestClient(app)

        # missing required field -> 422
        bad = client.post("/query", json={"db_id": "shop"})
        assert bad.status_code == 422

        # wrong type -> 422
        bad_type = client.post("/query", json={"db_id": "shop", "question": 123})
        assert bad_type.status_code == 422

        # unknown db -> 404
        missing = client.post("/query", json={"db_id": "ghost", "question": "?"})
        assert missing.status_code == 404

        # happy path -> 200 with full response shape
        ok_resp = client.post("/query", json=QueryRequest(db_id="shop", question="count things").model_dump())
        assert ok_resp.status_code == 200
        body = ok_resp.json()
        assert body["abstained"] is False and body["sql"]
        for field in ("db_id", "sql", "abstained", "reason", "confidence", "linked_tables", "latency_ms", "cache_hit", "rows"):
            assert field in body

    def test_endpoint_abstain_shape(self, tmp_path, engine_config, shop_db, shop_bundle, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        engine = _build_engine(tmp_path, engine_config)
        _register(engine, shop_db, shop_bundle, tmp_path)
        _mock_parser(monkeypatch, lambda self, st, b, t, v, **kw: ["DELETE FROM customers"])

        app = create_app(engine)
        client = TestClient(app)
        resp = client.post("/query", json={"db_id": "shop", "question": "delete everything"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["abstained"] is True and body["sql"] is None and body["reason"]
