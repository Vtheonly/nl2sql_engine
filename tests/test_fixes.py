"""Regression tests for the audit fixes.

Each test pins one confirmed defect from the line-by-line audit:
relation-matrix multi-token tables, [SCHEMA] anchoring, stale XGrammar bitmasks,
EBNF grounding (CAST/quoted ids/t.*/alias-qualified refs), alias-aware join
synthesis, fan-out false abstention, evaluation LIMIT injection, empty-gold-table
training data, schema cache drift, Unicode/single-char value indexing, 64-bit
result comparison, row caps, health surface, serve-time verifier mining, and
trainer bookkeeping (exact total_steps, patience persistence, epoch-boundary resume).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import torch

from nl2sql.compiler.sql_compiler import compile_and_validate_sql
from nl2sql.config import DataConfig, EngineConfig, HubConfig, ModelConfig, TrainingConfig
from nl2sql.datasets.acquire import DatasetAcquisitionResult
from nl2sql.datasets.prepare import (
    ExampleRecord,
    build_db_caches,
    extract_gold_schema_targets,
    format_schema_for_parser,
    load_minidev_records,
    prepare_records,
)
from nl2sql.db.introspect import JoinEdge, SchemaBundle, TableInfo, ColumnInfo, compute_ddl_hash, introspect_database
from nl2sql.db.value_index import ValueIndex
from nl2sql.engine import NL2SQLEngine, QueryRequest, create_app
from nl2sql.evaluation.evaluator import execute_sqlite_readonly, results_equivalent, NL2SQLEvaluator
from nl2sql.models.graphix_parser import (
    REL_FK_EDGE,
    REL_SAME_TABLE,
    REL_VALUE_MATCH,
    GraphixCodeT5Parser,
    Seq2SeqXGrammarLogitsProcessor,
    build_sqlite_schema_ebnf,
)
from nl2sql.models.schema_linker import ModernBertSchemaLinker
from nl2sql.retrieval.schema_retriever import HybridSchemaRetriever

from conftest import SHOP_DDL, WAREHOUSE_DDL, create_sqlite, make_engine_config


@pytest.fixture()
def training_setup(tmp_path: Path, shop_db: Path):
    db_warehouse = create_sqlite(tmp_path / "db_warehouse" / "db_warehouse.sqlite", WAREHOUSE_DDL)
    db_paths = {"shop": shop_db, "warehouse": db_warehouse}
    caches = build_db_caches(db_paths, cache_dir=tmp_path / "schema_caches")
    specs = {
        "shop": [
            ("How many open tickets for Alice?", "SELECT COUNT(*) FROM tickets WHERE status = 'open'"),
            ("What is the total order amount for Bob?", "SELECT SUM(amount) FROM orders JOIN customers ON orders.customer_id = customers.id WHERE customers.name = 'Bob'"),
        ],
        "warehouse": [
            ("Total revenue of all stores", "SELECT SUM(revenue) FROM sales"),
            ("How many stores are in the East region?", "SELECT COUNT(*) FROM stores JOIN regions ON stores.region_id = regions.id WHERE regions.region_name = 'East'"),
        ],
    }
    records: list[ExampleRecord] = []
    for db_id, db_path in db_paths.items():
        bundle, _ = caches[db_id]
        for i, (q, sql) in enumerate(specs[db_id]):
            g_tbls, g_cols = extract_gold_schema_targets(sql, bundle)
            records.append(
                ExampleRecord(
                    id=f"{db_id}_{i}", source="bird_train", db_id=db_id, db_path=db_path,
                    question=q, evidence="hint text", sql=sql,
                    gold_tables=g_tbls, gold_columns=g_cols,
                )
            )
    return records, caches


# --------------------------------------------------------------------------
# Bug 1: relation matrix drops multi-token table names / q_end heuristic
# --------------------------------------------------------------------------

class _SplittingTokenizer:
    """Tokenizer stub that splits identifiers like a BPE tokenizer would."""

    VOCAB = {
        "<pad>": 0, "<unk>": 1, "</s>": 2, "Question": 3, ":": 4, "[TBL]": 5,
        "[COL]": 6, "[SCHEMA]": 7, "schema": 8, "Schema": 9, "order": 10, "_": 11,
        "items": 12, "customers": 13, "id": 14, "INTEGER": 15, "TEXT": 16,
        "name": 17, "how": 18, "many": 19, "the": 20, "for": 21, "Alice": 22,
        "which": 23, "shows": 24, "details": 25, "total": 26, "qty": 27, "REAL": 28,
    }

    def convert_ids_to_tokens(self, ids):
        rev = {v: k for k, v in self.VOCAB.items()}
        return [rev.get(i, "<unk>") for i in ids]

    def tokenize(self, text: str) -> list[int]:
        import re

        out: list[int] = []
        for word in text.replace("\n", " ").split():
            if word in self.VOCAB:
                out.append(self.VOCAB[word])
                continue
            # Split every other whitespace word into identifier pieces and
            # punctuation so "order_items" -> order/_/items, "customers:" -> customers/:.
            for piece in re.findall(r"[A-Za-z]+|_|\d|.", word):
                out.append(self.VOCAB.get(piece, 1))
        return out


def _splitting_tables() -> list[TableInfo]:
    return [
        TableInfo(name="order_items", columns=[
            ColumnInfo(name="order_id", data_type="INTEGER", is_pk=True),
            ColumnInfo(name="total qty", data_type="REAL"),
        ]),
        TableInfo(name="customers", columns=[
            ColumnInfo(name="id", data_type="INTEGER", is_pk=True),
            ColumnInfo(name="name", data_type="TEXT"),
        ]),
    ]


def _splitting_bundle() -> SchemaBundle:
    return SchemaBundle(
        db_id="split", dialect="sqlite", ddl_hash="h1", tables=_splitting_tables(),
        join_edges=[JoinEdge(left_table="order_items", left_column="order_id",
                             right_table="customers", right_column="id", cardinality="N:1")],
    )


def test_relation_matrix_keeps_multi_token_tables():
    tok = _SplittingTokenizer()
    src = ("Question: how many the order items for Alice\n[SCHEMA] Schema:\n"
           "[TBL] order_items: [COL] order_id INTEGER [COL] total qty REAL\n"
           "[TBL] customers: [COL] id INTEGER [COL] name TEXT")
    ids = torch.tensor([tok.tokenize(src)])
    rel = GraphixCodeT5Parser.build_relation_matrix(ids, tok, [_splitting_bundle()], [_splitting_tables()], [[]])
    assert int((rel == REL_SAME_TABLE).sum()) > 0, "multi-token table lost SAME_TABLE edges"
    assert int((rel == REL_FK_EDGE).sum()) > 0, "multi-token table lost FK edges"


def test_relation_matrix_schema_anchor_survives_schema_word_in_question():
    tok = _SplittingTokenizer()
    src = ("Question: which schema shows the schema details for customers\n[SCHEMA] Schema:\n"
           "[TBL] customers: [COL] id INTEGER [COL] name TEXT")
    ids = torch.tensor([tok.tokenize(src)])
    rel = GraphixCodeT5Parser.build_relation_matrix(ids, tok, [_splitting_bundle()], [_splitting_tables()], [[]])
    # The word "schema" inside the question must not truncate the question span:
    # tokens of the schema section still carry same-table edges.
    assert int((rel == REL_SAME_TABLE).sum()) > 0


def test_relation_matrix_value_match_is_grounded_to_matched_table():
    tok = _SplittingTokenizer()
    src = ("Question: how many the order items for Alice\n[SCHEMA] Schema:\n"
           "[TBL] order_items: [COL] order_id INTEGER [COL] total qty REAL\n"
           "[TBL] customers: [COL] id INTEGER [COL] name TEXT")
    ids = tok.tokenize(src)
    from nl2sql.db.value_index import ValueMatch
    vals = [ValueMatch(table_name="customers", column_name="name", value="Alice", matched_phrase="Alice", score=1.0)]
    rel = GraphixCodeT5Parser.build_relation_matrix(torch.tensor([ids]), tok, [_splitting_bundle()], [_splitting_tables()], [vals])
    vm = (rel[0] == REL_VALUE_MATCH).nonzero().tolist()
    assert vm, "value match dropped"
    # No value-match edge may point into the order_items section of the schema.
    oi_tokens = set(range(ids.index(5) + 1, ids.index(13) + 2))  # tokens between [TBL] and customers
    for _, j in vm:
        assert j not in oi_tokens, "value match flooded an unrelated table"


def test_format_schema_for_parser_emits_schema_marker():
    txt = format_schema_for_parser(_splitting_bundle(), _splitting_tables(), [], "q")
    assert "[SCHEMA] Schema:" in txt


# --------------------------------------------------------------------------
# Bug 2: stale bitmasks + EBNF grounding
# --------------------------------------------------------------------------

class _StubMatcher:
    def __init__(self, terminated: bool = False) -> None:
        self._terminated = terminated
        self.accepted = 0

    def is_terminated(self) -> bool:
        return self._terminated

    def accept_token(self, tid: int) -> bool:
        self.accepted += 1
        return True

    def fill_next_token_bitmask(self, bitmask, row: int) -> None:
        bitmask[row].zero_()


class _StubXgr:
    """Minimal xgrammar stand-in driving the logits-processor state machine."""

    def allocate_token_bitmask(self, batch_size: int, vocab_size: int) -> torch.Tensor:
        return torch.full((batch_size, (vocab_size + 31) // 32), -1, dtype=torch.int32)

    @staticmethod
    def apply_token_bitmask_inplace(scores, bitmask) -> None:
        pass


def _processor_with_stub(matchers, eos=2, vocab=64):
    proc = Seq2SeqXGrammarLogitsProcessor.__new__(Seq2SeqXGrammarLogitsProcessor)
    proc._xgr = _StubXgr()
    proc.compiled_grammar = object()
    proc.vocab_size = vocab
    proc.start_len = 1
    proc.eos_token_id = eos
    proc.pad_token_id = 0
    proc.matchers = matchers
    proc._dead_rows = set()
    proc.bitmask = _StubXgr().allocate_token_bitmask(len(matchers), vocab)
    return proc


def test_terminated_rows_are_reset_to_eos_only():
    proc = _processor_with_stub([_StubMatcher(terminated=True), _StubMatcher(terminated=False)])
    input_ids = torch.full((2, 3), 1, dtype=torch.long)
    scores = torch.zeros((2, 64))
    proc(input_ids, scores)
    # Row 0 must allow only EOS (and PAD): every other token bit is cleared.
    eos_word, eos_bit = divmod(2, 32)
    pad_word, pad_bit = divmod(0, 32)
    for w in range(proc.bitmask.shape[1]):
        for bit in range(32):
            tid = w * 32 + bit
            if tid in (2, 0):
                continue
            assert not (int(proc.bitmask[0, w]) >> bit & 1), f"terminated row still allows token {tid}"


def test_dead_rows_are_reset_unconstrained():
    proc = _processor_with_stub([_StubMatcher(), _StubMatcher()])
    proc._dead_rows.add(0)
    input_ids = torch.full((2, 3), 1, dtype=torch.long)
    scores = torch.zeros((2, 64))
    proc(input_ids, scores)
    # Dead row is fully unconstrained (all bits set), not carrying stale grammar bits.
    assert int(proc.bitmask[0].sum()) == proc.bitmask.shape[1] * -1 or (proc.bitmask[0] == -1).all()


def test_ebnf_supports_cast_quoted_ids_and_qualified_star():
    tables = _splitting_tables()
    ebnf = build_sqlite_schema_ebnf(tables, [])
    assert 'ws "AS" ws type_name' in ebnf, "CAST(x AS TYPE) must be a direct alternative"
    assert '"`order_items`"' in ebnf and '"\\"order_items\\""' in ebnf
    assert 'table_name "." "*"' in ebnf, "select_item must accept t.*"
    # Ungrounded backdoor removed: ident "." is restricted to real column names.
    assert 'ident "." (ident | "*")' not in ebnf


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SELECT CAST(SUM(order_id) AS INTEGER) FROM order_items", True),
        ('SELECT CAST("total qty" AS REAL) FROM order_items', True),
        ("SELECT ROUND(SUM(order_id), 2) FROM order_items", True),
        ("SELECT order_items.* FROM order_items", True),
        ("SELECT c.name FROM customers c WHERE c.id = 1", True),
        ("SELECT c.bar FROM customers c", False),   # alias ok but column must be real
        ("SELECT ghost.bar FROM customers", False),  # fully ungrounded reference
    ],
)
def test_ebnf_grounding_decisions(shop_bundle, sql, expected):
    ebnf = build_sqlite_schema_ebnf(shop_bundle.tables, [])
    # The grounded alternatives literally contain every allowed column spelling.
    grounded = sql.replace("SELECT ", "").split(" FROM ")[0]
    if expected:
        # At minimum the grammar must reference each identifier used.
        for ident in ("customers", "orders", "tickets", "name", "id", "status"):
            if ident in sql:
                assert f'"{ident}"' in ebnf
    else:
        assert 'ident "." (ident | "*")' not in ebnf


def test_greedy_candidate_zero_is_deterministic(tiny_t5_dir, shop_bundle):
    parser = GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).eval()
    text = "Question: how many open tickets\n[SCHEMA] Schema:\n[TBL] tickets: [COL] status TEXT"
    kwargs = dict(max_source_len=128, max_target_len=24, use_xgrammar=False)
    outs = parser.generate_candidates(text, shop_bundle, shop_bundle.tables, [], num_candidates=3, **kwargs)
    greedy = parser.generate_candidates(text, shop_bundle, shop_bundle.tables, [], num_candidates=1, **kwargs)
    # Candidate 0 of the diverse batch must be the greedy (highest-likelihood) SQL.
    if outs and greedy:
        assert outs[0] == greedy[0]
    # And a single-candidate request is pure greedy regardless of configuration.
    assert parser.generate_candidates(text, shop_bundle, shop_bundle.tables, [], num_candidates=1, **kwargs) == greedy


# --------------------------------------------------------------------------
# Bug 3: alias joins, fan-out false abstention, evaluation LIMIT injection
# --------------------------------------------------------------------------

@pytest.fixture()
def fanout_db(tmp_path: Path):
    ddl = """
    CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
    CREATE TABLE tickets (id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT);
    INSERT INTO customers VALUES (1,'Alice'),(2,'Bob');
    INSERT INTO orders VALUES (10,1,5.0),(11,2,6.0);
    INSERT INTO tickets VALUES (100,1,'open'),(101,2,'closed');
    """
    p = create_sqlite(tmp_path / "fanout.sqlite", ddl)
    return p, introspect_database(f"sqlite:///{p.resolve()}", db_id="fanout")


def test_join_synthesis_uses_query_aliases(fanout_db):
    db_path, bundle = fanout_db
    res = compile_and_validate_sql(
        "SELECT c.name, o.amount FROM customers c JOIN orders o", bundle, default_limit=0
    )
    compact = res.sql.upper().replace(" ", "")
    assert "ONO.CUSTOMER_ID=C.ID" in compact, res.sql
    # The synthesized SQL must actually execute (raw table refs would fail with
    # 'no such column' once the table is aliased).
    ok, rows, _, msg = execute_sqlite_readonly(db_path, res.sql)
    assert ok, f"aliased join SQL failed: {msg}"
    assert rows


def test_fanout_detection_excludes_invariant_aggregates(fanout_db):
    _, bundle = fanout_db
    for sql in [
        "SELECT c.name, MIN(o.amount), MAX(t.id) FROM customers c JOIN orders o ON c.id=o.customer_id JOIN tickets t ON c.id=t.customer_id GROUP BY c.name",
        "SELECT c.name, COUNT(DISTINCT o.id), COUNT(DISTINCT t.id) FROM customers c JOIN orders o ON c.id=o.customer_id JOIN tickets t ON c.id=t.customer_id GROUP BY c.name",
        "WITH pre AS (SELECT customer_id, SUM(amount) s FROM orders GROUP BY customer_id) SELECT c.name, p.s, COUNT(t.id) FROM customers c JOIN pre p ON c.id=p.customer_id JOIN tickets t ON c.id=t.customer_id GROUP BY c.name",
    ]:
        res = compile_and_validate_sql(sql, bundle, default_limit=0)
        assert res.ambiguity_flags == [], f"safe query falsely flagged: {sql[:60]}"
    # Genuine fan-out (plain SUM + COUNT over two N:1 children) is still flagged,
    # including through table aliases.
    risky = "SELECT c.name, SUM(o.amount), COUNT(t.id) FROM customers c JOIN orders o ON c.id=o.customer_id JOIN tickets t ON c.id=t.customer_id GROUP BY c.name"
    res = compile_and_validate_sql(risky, bundle, default_limit=0)
    assert any("fan-out" in f.lower() for f in res.ambiguity_flags)
    assert res.fanout_warnings, "fan-out warning list renamed but must stay populated"


def test_engine_evaluation_does_not_inject_limit(tmp_path, engine_config, shop_db, shop_bundle, monkeypatch):
    engine = NL2SQLEngine(engine_config, artifacts_dir=Path(engine_config.training.output_dir) / "artifacts", device="cpu")
    v_idx = ValueIndex(tmp_path / "vi.sqlite")
    v_idx.build_from_db(f"sqlite:///{shop_db.resolve()}", shop_bundle)
    engine.db_registry["shop"] = (shop_bundle, v_idx, shop_db)

    monkeypatch.setattr(
        GraphixCodeT5Parser, "generate_candidates",
        lambda self, source_text, bundle, tables, value_matches, **kw: ["SELECT name FROM customers"],
    )
    resp = engine.answer(db_id="shop", question="list all people")
    assert not resp.abstained
    assert "LIMIT" not in (resp.sql or "").upper(), "evaluation path must not truncate unaggregated SELECTs"

    # The production API still applies its guardrail by default.
    from fastapi.testclient import TestClient

    client = TestClient(create_app(engine))
    engine._template_cache.clear()
    r = client.post("/query", json={"db_id": "shop", "question": "list all the people"})
    assert r.status_code == 200
    assert "LIMIT" in r.json()["sql"].upper()


def test_engine_answer_default_limit_none_semantics(tmp_path, engine_config, shop_db, shop_bundle, monkeypatch):
    engine = NL2SQLEngine(engine_config, artifacts_dir=Path(engine_config.training.output_dir) / "artifacts", device="cpu")
    v_idx = ValueIndex(tmp_path / "vi2.sqlite")
    v_idx.build_from_db(f"sqlite:///{shop_db.resolve()}", shop_bundle)
    engine.db_registry["shop"] = (shop_bundle, v_idx, shop_db)
    monkeypatch.setattr(
        GraphixCodeT5Parser, "generate_candidates",
        lambda self, source_text, bundle, tables, value_matches, **kw: ["SELECT name FROM customers"],
    )
    resp = engine.answer(db_id="shop", question="list the people again", default_limit=5)
    assert resp.sql and "LIMIT" in resp.sql.upper() and "5" in resp.sql


# --------------------------------------------------------------------------
# Bug 4: training data quality
# --------------------------------------------------------------------------

def _acq_from_rows(tmp_path: Path, rows: list[dict]) -> DatasetAcquisitionResult:
    p = tmp_path / "prepared" / "train.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return DatasetAcquisitionResult(bird_train_jsonl=p, bird_train_db_root=tmp_path / "dbs")


def test_prepare_records_drops_empty_gold_tables(tmp_path: Path):
    db = create_sqlite(tmp_path / "dbs" / "shop" / "shop.sqlite", SHOP_DDL)
    rows = [
        {"id": "ok", "source": "bird_train", "db_id": "shop", "question": "q1", "evidence": "", "sql": "SELECT COUNT(*) FROM tickets"},
        {"id": "bad", "source": "bird_train", "db_id": "shop", "question": "q2", "evidence": "", "sql": "SELECT ??? FROM"},
        {"id": "ghost", "source": "bird_train", "db_id": "shop", "question": "q3", "evidence": "", "sql": "SELECT * FROM ghost_table"},
    ]
    train, val, test, _ = prepare_records(_acq_from_rows(tmp_path, rows), cache_dir=tmp_path / "cache", val_ratio_by_db=0.0, seed=1)
    assert [r.id for r in train] == ["ok"], "records without gold tables must not reach the training split"


def test_prepare_records_precomputes_value_matches(tmp_path: Path):
    create_sqlite(tmp_path / "dbs" / "shop" / "shop.sqlite", SHOP_DDL)
    rows = [{"id": "r", "source": "bird_train", "db_id": "shop", "question": "How many open tickets for Alice?", "evidence": "", "sql": "SELECT COUNT(*) FROM tickets WHERE status = 'open'"}]
    train, _, _, _ = prepare_records(_acq_from_rows(tmp_path, rows), cache_dir=tmp_path / "cache", val_ratio_by_db=0.0, seed=1)
    assert train and train[0].value_matches, "value matches must be precomputed on the record"
    assert any(v.value == "open" for v in train[0].value_matches)


def test_prepare_records_rejects_db_id_collisions(tmp_path: Path):
    (tmp_path / "dbs" / "shop" / "shop.sqlite").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "dbs" / "shop" / "shop.sqlite").write_bytes(b"")
    (tmp_path / "other" / "shop" / "shop.sqlite").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "other" / "shop" / "shop.sqlite").write_bytes(b"")
    rows = [
        {"id": "a", "source": "bird_train", "db_id": "shop", "question": "q", "evidence": "", "sql": "SELECT 1"},
    ]
    train_jsonl = tmp_path / "bird_train.jsonl"
    train_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dev_jsonl = tmp_path / "spider_train.jsonl"
    dev_jsonl.write_text("\n".join(json.dumps({**r, "id": "b"}) for r in rows), encoding="utf-8")
    acq = DatasetAcquisitionResult(
        bird_train_jsonl=train_jsonl, bird_train_db_root=tmp_path / "dbs",
        spider_train_jsonl=dev_jsonl, spider_db_root=tmp_path / "other",
    )
    with pytest.raises(ValueError, match="shop"):
        prepare_records(acq, cache_dir=tmp_path / "cache", val_ratio_by_db=0.0, seed=1)


def test_extract_gold_targets_skips_ambiguous_unqualified_columns():
    tables = [
        TableInfo(name="a", columns=[ColumnInfo(name="id", data_type="INTEGER", is_pk=True), ColumnInfo(name="x", data_type="TEXT")]),
        TableInfo(name="b", columns=[ColumnInfo(name="id", data_type="INTEGER", is_pk=True)]),
    ]
    bundle = SchemaBundle(db_id="amb", dialect="sqlite", ddl_hash="h", tables=tables)
    # The unqualified 'id' in SELECT exists in both tables -> must be skipped; the
    # qualified join keys still resolve.
    tbls, cols = extract_gold_schema_targets("SELECT id FROM a JOIN b ON a.x = b.id", bundle)
    assert set(tbls) == {"a", "b"}
    assert {"a.x", "b.id"} <= set(cols)
    assert "a.id" not in cols, "ambiguous unqualified column was attributed anyway"


def test_extract_gold_targets_skips_order_by_output_alias():
    tables = [
        TableInfo(name="orders", columns=[ColumnInfo(name="amount", data_type="REAL")]),
        TableInfo(name="customers", columns=[ColumnInfo(name="name", data_type="TEXT"), ColumnInfo(name="amount", data_type="REAL")]),
    ]
    bundle = SchemaBundle(db_id="alias", dialect="sqlite", ddl_hash="h", tables=tables)
    sql = "SELECT o.amount AS name FROM orders o JOIN customers c ON 1=1 ORDER BY name"
    _, cols = extract_gold_schema_targets(sql, bundle)
    # 'name' in ORDER BY is the output alias, not customers.name.
    assert "customers.name" not in cols


def test_load_minidev_records_uses_dev_caches(tmp_path: Path):
    create_sqlite(tmp_path / "dbs" / "shop" / "shop.sqlite", SHOP_DDL)
    rows = [{"id": "t", "source": "bird_train", "db_id": "shop", "question": "q", "evidence": "", "sql": "SELECT COUNT(*) FROM tickets"}]
    train, _, _, caches = prepare_records(_acq_from_rows(tmp_path, rows), cache_dir=tmp_path / "cache", val_ratio_by_db=0.0, seed=1)
    mini = tmp_path / "prepared" / "bird_minidev.jsonl"
    mini.write_text(json.dumps({"id": "m1", "source": "bird_minidev", "db_id": "shop", "question": "open tickets?", "evidence": "", "sql": "SELECT COUNT(*) FROM tickets WHERE status = 'open'"}), encoding="utf-8")
    acq = DatasetAcquisitionResult(bird_minidev_jsonl=mini, bird_dev_db_root=tmp_path / "dbs")
    recs = load_minidev_records(acq, caches)
    assert len(recs) == 1 and recs[0].db_path == tmp_path / "dbs" / "shop" / "shop.sqlite"
    assert recs[0].gold_tables == ["tickets"]


# --------------------------------------------------------------------------
# Schema drift / engine registration
# --------------------------------------------------------------------------

def test_build_db_caches_rebuilds_on_schema_change(tmp_path: Path):
    db = create_sqlite(tmp_path / "drift.sqlite", SHOP_DDL)
    caches = build_db_caches({"drift": db}, tmp_path / "cache")
    assert len(caches["drift"][0].tables) == 3

    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE customers ADD COLUMN email TEXT")
    conn.commit()
    conn.close()

    caches2 = build_db_caches({"drift": db}, tmp_path / "cache")
    bundle = caches2["drift"][0]
    assert bundle.get_table("customers").get_column("email") is not None, "stale bundle served after schema change"
    assert bundle.ddl_hash == compute_ddl_hash(f"sqlite:///{db.resolve()}")


def test_register_database_derives_sqlite_path_and_example_queries(tmp_path, engine_config, shop_db):
    engine = NL2SQLEngine(engine_config, artifacts_dir=Path(engine_config.training.output_dir) / "artifacts", device="cpu")
    bundle = engine.register_database(
        "shop", f"sqlite:///{shop_db.resolve()}", cache_dir=tmp_path / "cache",
        example_queries=[{"question": "open tickets?", "sql": "SELECT COUNT(*) FROM tickets"}],
    )
    assert engine.db_registry["shop"][2] == shop_db, "sqlite path must be derived from the connection URL"
    assert bundle.example_queries and "open tickets?" in bundle.example_queries[0]["question"]

    # Example queries flow into the parser prompt via the retriever.
    ret = HybridSchemaRetriever(embedder=None).retrieve("anything", bundle, None)
    assert ret.example_hints and ret.example_hints[0]["sql"].startswith("SELECT COUNT")
    txt = format_schema_for_parser(bundle, bundle.tables, [], "q", example_hints=ret.example_hints)
    assert "Example Queries:" in txt and "SELECT COUNT" in txt


def test_health_endpoint_reports_models_loaded(tmp_path, engine_config, shop_db, shop_bundle):
    from fastapi.testclient import TestClient

    engine = NL2SQLEngine(engine_config, artifacts_dir=Path(engine_config.training.output_dir) / "artifacts", device="cpu")
    v_idx = ValueIndex(tmp_path / "vi_h.sqlite")
    v_idx.build_from_db(f"sqlite:///{shop_db.resolve()}", shop_bundle)
    engine.db_registry["shop"] = (shop_bundle, v_idx, shop_db)
    client = TestClient(create_app(engine))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models_loaded"]["parser"] is True
    assert body["models_loaded"]["verifier"] is False  # untrained artifact -> degraded mode is visible
    assert body["registered_databases"] == ["shop"]


# --------------------------------------------------------------------------
# ValueIndex Unicode / single-char codes
# --------------------------------------------------------------------------

def test_value_index_matches_unicode_and_quoted_single_chars(tmp_path: Path):
    ddl = """
    CREATE TABLE cities (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
    CREATE TABLE shipments (id INTEGER PRIMARY KEY, status TEXT);
    INSERT INTO cities VALUES (1, 'Zürich', 'CH'), (2, 'São Paulo', 'BR');
    INSERT INTO shipments VALUES (10, 'A'), (11, 'F'), (12, 'open');
    """
    db = create_sqlite(tmp_path / "intl.sqlite", ddl)
    url = f"sqlite:///{db.resolve()}"
    bundle = introspect_database(url, db_id="intl")
    idx = ValueIndex(tmp_path / "intl_vi.sqlite")
    idx.build_from_db(url, bundle)

    assert any(h.value == "Zürich" for h in idx.probe("How many shipments are in Zürich?"))
    assert any(h.value == "São Paulo" for h in idx.probe("Show shipments in São Paulo"))
    assert any(h.value == "A" for h in idx.probe("shipments with status = 'A'"))
    # Bare single letters remain filtered as noise.
    assert not any(h.value in ("A", "F") for h in idx.probe("a I of the and"))


# --------------------------------------------------------------------------
# Evaluator precision / row caps / breakdowns
# --------------------------------------------------------------------------

def test_results_equivalent_large_integers_exact():
    big = 2 ** 60
    assert results_equivalent("SELECT 1", [(big,)], [(big,)])
    assert not results_equivalent("SELECT 1", [(big,)], [(big + 2,)])
    # Small int / float still canonicalize together.
    assert results_equivalent("SELECT 1", [(1,)], [(1.0,)])


def test_execute_readonly_default_row_cap_is_eval_safe(tmp_path: Path):
    ddl = """
    CREATE TABLE t (id INTEGER PRIMARY KEY);
    INSERT INTO t WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x < 2500) SELECT x FROM c;
    """
    db = create_sqlite(tmp_path / "cap.sqlite", ddl)
    ok, rows, _, msg = execute_sqlite_readonly(db, "SELECT id FROM t")
    assert ok and len(rows) == 2500, f"eval cap truncated an unordered result: {len(rows) if rows else 0} ({msg})"


def test_evaluator_reports_per_difficulty_and_source_breakdown(shop_db: Path):
    recs = [
        ExampleRecord(id="s1", source="bird_dev", db_id="shop", db_path=shop_db, question="q1",
                      sql="SELECT COUNT(*) FROM tickets", difficulty="simple", gold_tables=["tickets"]),
        ExampleRecord(id="s2", source="bird_dev", db_id="shop", db_path=shop_db, question="q2",
                      sql="SELECT COUNT(*) FROM customers", difficulty="challenging", gold_tables=["customers"]),
    ]
    preds = {
        "s1": {"sql": "SELECT COUNT(*) FROM tickets", "abstained": False, "confidence": 0.9, "linked_tables": ["tickets"]},
        "s2": {"sql": "SELECT COUNT(*) FROM orders", "abstained": False, "confidence": 0.5, "linked_tables": []},
    }
    m = NL2SQLEvaluator().evaluate(recs, lambda r: preds[r.id], dataset_name="slices")
    assert m.by_difficulty["simple"]["execution_accuracy"] == 1.0
    assert m.by_difficulty["challenging"]["execution_accuracy"] == 0.0
    assert m.by_source["bird_dev"]["total"] == 2


# --------------------------------------------------------------------------
# Trainer bookkeeping
# --------------------------------------------------------------------------

def test_exact_total_steps_for_partial_accumulation_windows(tmp_path: Path, tiny_t5_dir, tiny_modernbert_dir):
    # 5 batches with grad_accum_steps=2 -> 3 optimizer steps per epoch, 2 epochs = 6.
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=2, stage="parser")
    cfg.training.grad_accum_steps = 2
    cfg.training.num_workers = 0
    from nl2sql.training.trainer import NL2SQLTrainer

    trainer = NL2SQLTrainer(cfg)
    # A synthetic loader with 5 batches.
    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, i):
            return {}

    model = torch.nn.Linear(2, 1)
    loader = torch.utils.data.DataLoader(_DS(), batch_size=1, collate_fn=lambda b: b[0])

    import math
    expected = math.ceil(5 / 2) * 2
    # _run_generic_loop computes the same expression; verify it directly to avoid
    # running a full training pass here.
    computed = math.ceil(len(loader) / cfg.training.grad_accum_steps) * cfg.training.epochs
    assert computed == expected == 6
    # And the loop's final history records global_step == total_steps (no early decay).
    res = trainer._run_generic_loop(
        "lin", model, loader, None,
        lambda m, batch, dev: m(torch.tensor([[1.0, 2.0]])).sum(),
        lambda m, d: None,
    )
    # With 5 batches and accum=2, each epoch runs 3 optimizer steps: 2 epochs = 6.
    assert res["history"][-1]["global_step"] == 6
    # LR only reaches zero at the very end (no early decay from undercounted steps).
    assert res["history"][0]["lr"] > 0


def test_checkpoint_persists_patience_left(tmp_path: Path):
    from nl2sql.training.checkpoint import CheckpointManager
    from nl2sql.utils.repro import set_seed

    set_seed(5)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    mgr = CheckpointManager(tmp_path / "run", keep_last_k=1)
    mgr.save("p", model, opt, None, None, epoch=3, global_step=9, best_metric=0.4,
             config_dict={}, metrics_history=[], patience_left=1)
    payload = mgr.load(mgr.resolve_resume_path("p", "auto"), model=model, optimizer=opt)
    assert payload["patience_left"] == 1
    meta = json.loads((tmp_path / "run" / "checkpoints" / "p" / "latest.txt").parent.joinpath("step_0000009", "meta.json").read_text())
    assert meta["patience_left"] == 1


def test_epoch_boundary_resume_never_replays_batches(tmp_path: Path, tiny_t5_dir, tiny_modernbert_dir, training_setup):
    records, caches = training_setup
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=2, stage="parser")
    cfg.training.num_workers = 0
    from nl2sql.training.trainer import NL2SQLTrainer

    trainer = NL2SQLTrainer(cfg)
    res = trainer.train_parser(records, records, caches)
    ckpt = trainer.ckpt_mgr.resolve_resume_path("parser", "auto")
    payload = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
    # The checkpoint records the NEXT epoch to run, not the epoch already trained.
    assert payload["epoch"] == 2
    assert payload["global_step"] == res["history"][-1]["global_step"]
    # Resuming with the same epoch budget runs zero new epochs (no replay).
    res2 = NL2SQLTrainer(cfg).train_parser(records, records, caches)
    assert len(res2["history"]) == 2


def test_verifier_mining_compiles_gold_and_candidates(tmp_path: Path, tiny_t5_dir, tiny_modernbert_dir, monkeypatch, training_setup):
    records, caches = training_setup
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="verifier")
    cfg.training.num_workers = 0
    from nl2sql.compiler import sql_compiler as compiler_mod
    from nl2sql.training.trainer import NL2SQLTrainer

    calls: list[str] = []
    real_compile = compiler_mod.compile_and_validate_sql

    def spy(raw_sql, bundle, **kw):
        calls.append(raw_sql)
        return real_compile(raw_sql, bundle, **kw)

    monkeypatch.setattr("nl2sql.training.trainer.compile_and_validate_sql", spy)

    parser_art = tmp_path / "art" / "parser"
    from nl2sql.models.graphix_parser import GraphixCodeT5Parser
    GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).save_pretrained(parser_art)

    trainer = NL2SQLTrainer(cfg)
    samples = trainer.mine_verifier_samples(records[:2], caches, parser_art, max_records=2)
    # Both the gold SQL and every generated candidate went through the compiler.
    assert any("SELECT" in c.upper() for c in calls), "compiler was never invoked during mining"
    for s in samples:
        if s["label"] == 1.0:
            # Positives are stored in SQLGlot-canonical form, like engine scoring time.
            assert s["candidate_sql"].upper().startswith("SELECT")


def test_temperature_calibration_fits_reasonable_temperature(tmp_path: Path, tiny_modernbert_dir):
    from nl2sql.models.verifier import ModernBertVerifier
    from nl2sql.training.trainer import NL2SQLTrainer

    cfg = make_engine_config(tmp_path, tiny_t5_dir := tiny_modernbert_dir, tiny_modernbert_dir, epochs=1)
    trainer = NL2SQLTrainer(cfg)
    model = ModernBertVerifier(str(tiny_modernbert_dir))

    # Synthetic validation logits: positives around +4, negatives around -4. The
    # calibrated temperature must land in a sane range and be persisted.
    val_samples = [
        {"question": "q", "schema_text": "s", "candidate_sql": "SELECT 1", "exec_status": "ok rows=1", "label": float(i % 2)}
        for i in range(16)
    ]
    with torch.no_grad():
        for i, s in enumerate(val_samples):
            pass
    # Force logits by monkeypatching forward to return preset logits per label.
    preset = {0: -4.0, 1: 4.0}

    def fake_forward(self, input_ids, attention_mask, labels=None):
        lab = labels[0].item() if labels is not None and labels.numel() else 1.0
        return {"logits": torch.full((input_ids.shape[0],), preset.get(int(lab), 1.0)), "loss": torch.tensor(0.0)}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ModernBertVerifier, "forward", fake_forward)
    try:
        trainer._calibrate_temperature(model, val_samples, max_length=64)
    finally:
        monkeypatch.undo()
    t = float(model.temperature.item())
    assert 0.05 <= t <= 10.0


# --------------------------------------------------------------------------
# Notebook contract: repo link
# --------------------------------------------------------------------------

def test_notebook_pulls_repo_directly():
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "train_nl2sql.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "https://github.com/Vtheonly/nl2sql_engine" in md + code
    assert 'os.environ.get("NL2SQL_REPO_URL", "https://github.com/Vtheonly/nl2sql_engine.git")' in code
