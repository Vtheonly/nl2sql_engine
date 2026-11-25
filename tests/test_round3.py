"""Round-3 fixes: WSD LR schedule, exact mid-epoch resume, deterministic per-epoch
shuffles, batched verifier mining, CTE fan-out rewriting, multi-dialect EXPLAIN
dry-run, composite PKs, FK-inference guards, low-cardinality value indexing,
thread-local OR-batched FTS probing, column-vector caching, the StructCBR query-log
miner, and checkpoint stage validation."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import torch
from sqlalchemy import create_engine

from nl2sql.compiler.sql_compiler import compile_and_validate_sql
from nl2sql.datasets.prepare import ExampleRecord, build_db_caches, extract_gold_schema_targets
from nl2sql.db.introspect import introspect_database
from nl2sql.db.query_log_miner import (
    apply_dbt_descriptions,
    enrich_bundle_from_logs,
    mine_join_edges_from_logs,
)
from nl2sql.db.value_index import ValueIndex
from nl2sql.retrieval.schema_retriever import HybridSchemaRetriever, _TABLE_CACHE_MAX
from nl2sql.training.checkpoint import CheckpointManager
from nl2sql.training.trainer import NL2SQLTrainer, _build_wsd_scheduler

from conftest import WAREHOUSE_DDL, create_sqlite, make_engine_config


_RECS = {
    "shop": [
        ("How many open tickets?", "SELECT COUNT(*) FROM tickets WHERE status = 'open'"),
        ("Total order amount?", "SELECT SUM(amount) FROM orders"),
    ],
    "warehouse": [
        ("Total revenue of all stores", "SELECT SUM(revenue) FROM sales"),
        ("Stores in the East region?", "SELECT COUNT(*) FROM stores JOIN regions ON stores.region_id = regions.id WHERE regions.region_name = 'East'"),
    ],
}


@pytest.fixture()
def training_setup(tmp_path: Path, shop_db: Path):
    db_warehouse = create_sqlite(tmp_path / "db_warehouse" / "db_warehouse.sqlite", WAREHOUSE_DDL)
    db_paths = {"shop": shop_db, "warehouse": db_warehouse}
    caches = build_db_caches(db_paths, cache_dir=tmp_path / "schema_caches")
    records = []
    for db_id, db_path in db_paths.items():
        bundle, _ = caches[db_id]
        for i, (q, sql) in enumerate(_RECS[db_id]):
            g_tbls, g_cols = extract_gold_schema_targets(sql, bundle)
            records.append(
                ExampleRecord(id=f"{db_id}_{i}", source="bird_train", db_id=db_id, db_path=db_path,
                              question=q, evidence="", sql=sql, gold_tables=g_tbls, gold_columns=g_cols)
            )
    return records, caches


# --------------------------------------------------------------------------
# WSD LR schedule
# --------------------------------------------------------------------------

def test_wsd_scheduler_warmup_stable_decay_never_zero():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=1.0)
    sched = _build_wsd_scheduler(opt, total_steps=100, warmup_steps=10, decay_steps=20, final_lr_ratio=0.1)
    lrs = []
    for _ in range(110):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    # Step 0 sits at the warmup origin (0.0) exactly like HF's warmup; every
    # subsequent step stays strictly positive, ending at the 0.1 floor -- never 0.
    assert lrs[0] == 0.0
    assert lrs[5] == pytest.approx(0.5)  # linear warmup
    assert lrs[50] == pytest.approx(1.0)  # stable plateau
    assert lrs[100] == pytest.approx(0.1, abs=1e-6)  # decay floor, never 0.0
    assert lrs[109] == pytest.approx(0.1)  # extension plateau beyond the horizon
    assert all(lr > 0.0 for lr in lrs[1:])


def test_wsd_default_used_by_trainer_and_cosine_still_selectable(tmp_path, tiny_t5_dir, tiny_modernbert_dir, training_setup):
    records, caches = training_setup
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="parser")
    assert cfg.training.lr_schedule == "wsd"
    trainer = NL2SQLTrainer(cfg)
    res = trainer.train_parser(records, records, caches)
    # WSD ends at its LR floor, never at zero -- multi-day/multi-account
    # extensions keep learning.
    assert res["history"][0]["lr"] > 0

    cfg2 = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="parser")
    cfg2.training.output_dir = tmp_path / "run_cosine"
    cfg2.training.lr_schedule = "cosine"
    trainer2 = NL2SQLTrainer(cfg2)
    res2 = trainer2.train_parser(records, records, caches)
    # The fixed-horizon cosine reaches exactly 0.0 at its planned horizon -- the
    # very failure mode WSD exists to avoid (comparative demonstration).
    assert res2["history"][0]["lr"] == 0.0


# --------------------------------------------------------------------------
# Exact mid-epoch resume + deterministic per-epoch shuffles
# --------------------------------------------------------------------------

class _CountingDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 5) -> None:
        self.items = list(range(n))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {"i": self.items[i]}


def _counting_loader(seed: int | None, shuffle: bool) -> torch.utils.data.DataLoader:
    gen = torch.Generator() if (seed is not None and shuffle) else None
    return torch.utils.data.DataLoader(
        _CountingDataset(5), batch_size=1, shuffle=shuffle, generator=gen,
        collate_fn=lambda b: b[0],
    )


def test_per_epoch_seeded_shuffle_is_deterministic(tmp_path, tiny_t5_dir, tiny_modernbert_dir):
    seen_orders: list[list[int]] = []

    def step(m, batch, dev):
        seen_orders[-1].append(batch["i"])
        return m(torch.tensor([[1.0, 2.0]])).sum()

    # Independent runs (fresh trainer + output dir each) must produce identical
    # per-epoch batch order for the same seed.
    for run in range(2):
        seen_orders.append([])
        cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="parser")
        cfg.training.output_dir = tmp_path / f"run_seed_{run}"
        trainer = NL2SQLTrainer(cfg)
        loader = _counting_loader(seed=1, shuffle=True)
        trainer._run_generic_loop("seedcheck", torch.nn.Linear(2, 1), loader, None, step, lambda m, d: None)
    assert seen_orders[0] == seen_orders[1]

    # A different run seed shuffles differently (the seeding is actually live).
    seen_orders.append([])
    cfg_b = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="parser")
    cfg_b.seed = cfg_b.seed + 1
    cfg_b.training.output_dir = tmp_path / "run_seed_alt"
    loader_b = _counting_loader(seed=1, shuffle=True)
    NL2SQLTrainer(cfg_b)._run_generic_loop("seedcheck", torch.nn.Linear(2, 1), loader_b, None, step, lambda m, d: None)
    assert seen_orders[2] != seen_orders[0]


def test_mid_epoch_resume_processes_remaining_batches(tmp_path, tiny_t5_dir, tiny_modernbert_dir):
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="parser")
    cfg.training.num_workers = 0
    trainer = NL2SQLTrainer(cfg)

    model = torch.nn.Linear(2, 1)
    loader = _counting_loader(seed=7, shuffle=True)

    processed: list[int] = []

    def step(m, batch, dev):
        processed.append(batch["i"])
        return m(torch.tensor([[1.0, 2.0]])).sum()

    # "Run" epoch 0 fully, then hand-write a mid-epoch checkpoint at batch 3/5.
    trainer._run_generic_loop("midep", model, loader, None, step, lambda m, d: None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = _build_wsd_scheduler(opt, total_steps=5, warmup_steps=0, decay_steps=2, final_lr_ratio=0.1)
    ckpt_path = trainer.ckpt_mgr.save(
        stage="midep", model=model, optimizer=opt, scheduler=sched, scaler=None,
        epoch=0, global_step=3, best_metric=1.0, config_dict={},
        metrics_history=[], is_best=False, patience_left=3, batch_idx_in_epoch=3,
    )
    payload = torch.load(ckpt_path / "trainer_state.pt", map_location="cpu", weights_only=False)
    assert payload["batch_idx_in_epoch"] == 3

    # Resume: only the 2 remaining batches of epoch 0 may run.
    processed.clear()
    trainer2 = NL2SQLTrainer(cfg)
    res = trainer2._run_generic_loop("midep", torch.nn.Linear(2, 1), loader, None, step, lambda m, d: None)
    assert len(processed) == 2, f"resume must fast-forward past 3 batches, ran {len(processed)}"
    assert res["history"][0]["epoch"] == 1
    # End-of-epoch save resets the marker.
    final = torch.load(
        (Path(cfg.training.output_dir) / "checkpoints" / "midep" / "step_0000005") / "trainer_state.pt",
        map_location="cpu", weights_only=False,
    )
    assert final["batch_idx_in_epoch"] == 0 and final["epoch"] == 1


def test_checkpoint_meta_records_batch_idx_and_stage_guard(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    mgr = CheckpointManager(tmp_path / "run", keep_last_k=1)
    p = mgr.save("parser", model, opt, None, None, epoch=1, global_step=4, best_metric=0.5,
                 config_dict={}, metrics_history=[], batch_idx_in_epoch=4)
    meta = json.loads((p / "meta.json").read_text())
    assert meta["batch_idx_in_epoch"] == 4
    # Same stage resolves; a different stage refuses cross-stage resume.
    assert mgr.resolve_resume_path("parser", str(p)) == p
    assert mgr.resolve_resume_path("linker", str(p)) is None


# --------------------------------------------------------------------------
# Batched verifier mining
# --------------------------------------------------------------------------

def test_mining_batch_size_splits_into_chunks_without_changing_output(tmp_path, tiny_t5_dir, tiny_modernbert_dir, monkeypatch, training_setup):
    records, caches = training_setup
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="verifier")
    cfg.training.num_workers = 0

    parser_art = tmp_path / "art" / "parser"
    from nl2sql.models.graphix_parser import GraphixCodeT5Parser
    GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).save_pretrained(parser_art)

    chunks: list[int] = []
    real_generate = GraphixCodeT5Parser.generate_candidates

    def spy_generate(self, source_text, bundle, tables, value_matches, **kw):
        chunks.append(len(source_text))
        return real_generate(self, source_text, bundle, tables, value_matches, **kw)

    monkeypatch.setattr(GraphixCodeT5Parser, "generate_candidates", spy_generate)

    trainer = NL2SQLTrainer(cfg)
    chunks.clear()
    a = trainer.mine_verifier_samples(records[:4], caches, parser_art, max_records=4, batch_size=2)
    assert len(chunks) == 4  # one parser generation per record
    chunks.clear()
    b = trainer.mine_verifier_samples(records[:4], caches, parser_art, max_records=4, batch_size=7)
    assert len(chunks) == 4
    # All gold positives are produced regardless of chunking.
    assert sum(1 for s in a if s["label"] == 1.0) == sum(1 for s in b if s["label"] == 1.0)


# --------------------------------------------------------------------------
# Multi-dialect EXPLAIN dry-run
# --------------------------------------------------------------------------

def test_explain_dry_run_rolls_back_and_validates(tmp_path, engine_config, shop_db, shop_bundle, monkeypatch):
    from nl2sql.engine import NL2SQLEngine

    engine = NL2SQLEngine(engine_config, artifacts_dir=Path(engine_config.training.output_dir) / "artifacts", device="cpu")
    engine.db_registry["shop"] = (shop_bundle, ValueIndex(tmp_path / "vi_x.sqlite"), shop_db)
    # Non-SQLite path: pretend a SQLite URL is a foreign dialect by caching the
    # engine directly (sqlite EXPLAIN is valid SQL, so the code path is identical).
    engine._conn_urls["shop"] = "sqlite://"
    engine._explain_engines["shop"] = create_engine(f"sqlite:///{shop_db.resolve()}")

    ok, summary = engine._explain_dry_run("shop", "SELECT name FROM customers")
    assert ok and summary == "explain_ok"
    ok_bad, summary_bad = engine._explain_dry_run("shop", "SELECT nope FROM missing_table_xyz")
    assert not ok_bad and "err" in summary_bad
    # Unknown db -> ast_valid passthrough (no connection URL registered).
    ok2, s2 = engine._explain_dry_run("ghost", "SELECT 1")
    assert ok2 and s2 == "ast_valid"
    engine._explain_engines["shop"].dispose()


def test_engine_explain_dry_run_used_for_non_sqlite_candidates(tmp_path, engine_config, shop_db, shop_bundle, monkeypatch):
    from nl2sql.engine import NL2SQLEngine
    from nl2sql.models.graphix_parser import GraphixCodeT5Parser

    engine = NL2SQLEngine(engine_config, artifacts_dir=Path(engine_config.training.output_dir) / "artifacts", device="cpu")
    v_idx = ValueIndex(tmp_path / "vi_e.sqlite")
    v_idx.build_from_db(f"sqlite:///{shop_db.resolve()}", shop_bundle)
    # Registered WITHOUT a sqlite path but WITH a connection URL: the EXPLAIN
    # layer must replace the old "ast_valid" no-op validation.
    engine.db_registry["shop"] = (shop_bundle, v_idx, None)
    engine._conn_urls["shop"] = "sqlite://"
    engine._explain_engines["shop"] = create_engine(f"sqlite:///{shop_db.resolve()}")

    monkeypatch.setattr(
        GraphixCodeT5Parser, "generate_candidates",
        lambda self, source_text, bundle, tables, value_matches, **kw: ["SELECT name FROM customers"],
    )
    resp = engine.answer(db_id="shop", question="list the people")
    assert not resp.abstained
    engine._explain_engines["shop"].dispose()


def test_engine_rbac_empty_allowlist_gives_explicit_reason(tmp_path, engine_config, shop_db, shop_bundle, monkeypatch):
    from nl2sql.engine import NL2SQLEngine

    engine = NL2SQLEngine(engine_config, artifacts_dir=Path(engine_config.training.output_dir) / "artifacts", device="cpu")
    v_idx = ValueIndex(tmp_path / "vi_r.sqlite")
    v_idx.build_from_db(f"sqlite:///{shop_db.resolve()}", shop_bundle)
    engine.db_registry["shop"] = (shop_bundle, v_idx, shop_db)
    resp = engine.answer(db_id="shop", question="list customers", allowed_tables=["ghost_table"])
    assert resp.abstained and "RBAC" in resp.reason


# --------------------------------------------------------------------------
# Composite primary keys & FK-inference guards
# --------------------------------------------------------------------------

JUNCTION_DDL = """
CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT);
CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade TEXT, PRIMARY KEY (student_id, course_id));
INSERT INTO students VALUES (1, 'Ada'), (2, 'Grace');
INSERT INTO courses VALUES (10, 'Math'), (11, 'Physics');
INSERT INTO enrollments VALUES (1, 10, 'A'), (1, 11, 'B'), (2, 10, 'C');
"""


def test_composite_pk_members_get_pk_tag_and_n1_cardinality(tmp_path: Path):
    db = create_sqlite(tmp_path / "junction.sqlite", JUNCTION_DDL)
    bundle = introspect_database(f"sqlite:///{db.resolve()}", db_id="junction")
    enr = bundle.get_table("enrollments")
    sid = enr.get_column("student_id")
    cid = enr.get_column("course_id")
    # Both composite-PK members now carry the [PK] tag (previously dropped).
    assert sid.is_pk and cid.is_pk
    # But each member is NOT individually unique: the junction edge to students
    # must be N:1 (previously a phantom 1:1 because every [PK] counted as unique).
    edge = next(
        e for e in bundle.join_edges
        if {e.left_table.lower(), e.right_table.lower()} == {"enrollments", "students"}
    )
    assert edge.cardinality == "N:1"


FLAG_FK_DDL = """
CREATE TABLE flights (id INTEGER PRIMARY KEY, flight_no TEXT, is_active INTEGER, flag INTEGER);
CREATE TABLE actives (id INTEGER PRIMARY KEY, note TEXT);
INSERT INTO flights VALUES (1, 'AB1', 0, 0), (2, 'AB2', 1, 1);
INSERT INTO actives VALUES (0, 'zero'), (1, 'one');
"""


def test_boolean_flag_columns_do_not_infer_false_fks(tmp_path: Path):
    db = create_sqlite(tmp_path / "flags.sqlite", FLAG_FK_DDL)
    bundle = introspect_database(f"sqlite:///{db.resolve()}", db_id="flags")
    inferred = [
        e for e in bundle.join_edges
        if e.source == "inferred_fk"
        and {e.left_table.lower(), e.right_table.lower()} == {"flights", "actives"}
    ]
    # is_active/flag values (0/1) are all present in actives.id, so the inclusion
    # probe passes -- only the boolean/flag guard keeps the false edge out.
    assert inferred == []


ZIP_FK_DDL = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, zip TEXT);
CREATE TABLE addresses (zip TEXT PRIMARY KEY, city TEXT);
INSERT INTO customers VALUES (1, '10001'), (2, '94105');
INSERT INTO addresses VALUES ('10001', 'NYC'), ('94105', 'SF');
"""


def test_zip_suffix_supports_inferred_fk(tmp_path: Path):
    db = create_sqlite(tmp_path / "zips.sqlite", ZIP_FK_DDL)
    bundle = introspect_database(f"sqlite:///{db.resolve()}", db_id="zips")
    zipped = [
        e for e in bundle.join_edges
        if {e.left_table.lower(), e.right_table.lower()} == {"customers", "addresses"}
    ]
    assert zipped, "zip-suffix foreign key was not inferred"


# --------------------------------------------------------------------------
# ValueIndex: low-cardinality columns + thread-local OR-batched probe
# --------------------------------------------------------------------------

LOWCARD_DDL_ROWS = [
    (1, 3, "DHL", "2026-01-05", 0.11),
    (2, 7, "UPS", "2026-02-10", 0.22),
]


def _lowcard_db(tmp_path: Path) -> Path:
    # 150 rows: tracking/id carry 150 distinct values (above the 100 cutoff),
    # status keeps 2 and dates keep a handful (below it).
    rows = list(LOWCARD_DDL_ROWS)
    rows += [
        (i, i % 2, "DHL" if i % 3 else "UPS", f"2026-03-{(i % 28) + 1:02d}", i + 0.5)
        for i in range(3, 151)
    ]
    p = tmp_path / "lowcard.sqlite"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE shipments (id INTEGER PRIMARY KEY, status INTEGER, courier TEXT, ship_date TEXT, tracking REAL)")
    conn.executemany("INSERT INTO shipments VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return p


def test_low_cardinality_numeric_and_date_columns_are_indexed(tmp_path: Path):
    db = _lowcard_db(tmp_path)
    url = f"sqlite:///{db.resolve()}"
    bundle = introspect_database(url, db_id="lc")
    idx = ValueIndex(tmp_path / "lc_vi.sqlite")
    total = idx.build_from_db(url, bundle, low_cardinality_max_distinct=100)
    assert total > 0

    conn = sqlite3.connect(idx.path)
    cols = {r[0] for r in conn.execute("SELECT DISTINCT column_name FROM cell_fts").fetchall()}
    conn.close()
    assert {"status", "ship_date", "courier"} <= cols  # low-card + text indexed
    assert "tracking" not in cols  # high-cardinality numeric excluded
    assert "id" not in cols  # unique ids above the cutoff excluded

    hits = idx.probe("shipments with status 3")
    assert any(h.value == "3" and h.column_name == "status" for h in hits)
    date_hits = idx.probe("shipments on 2026-01-05")
    assert any(h.value == "2026-01-05" and h.column_name == "ship_date" for h in date_hits)


def test_probe_reuses_thread_local_connection(tmp_path: Path):
    db = _lowcard_db(tmp_path)
    url = f"sqlite:///{db.resolve()}"
    bundle = introspect_database(url, db_id="tl")
    idx = ValueIndex(tmp_path / "tl_vi.sqlite")
    idx.build_from_db(url, bundle)

    calls = {"n": 0}
    real_connect = sqlite3.connect

    def counting_connect(*a, **kw):
        calls["n"] += 1
        return real_connect(*a, **kw)

    sqlite3.connect = counting_connect
    try:
        idx.probe("shipments with status 3")
        idx.probe("shipments with status 7")
        idx.probe("shipments via DHL")
    finally:
        sqlite3.connect = real_connect
    # init + build connections aside, the three probes share ONE thread-local
    # read-only connection: no per-probe open/close churn.
    assert calls["n"] <= 3, calls["n"]


# --------------------------------------------------------------------------
# Retriever: column-vector cache + bounded LRU
# --------------------------------------------------------------------------

class _StubEncoder:
    """Deterministic stub embedder: hash-bucketed vectors, no downloads."""

    def __init__(self, hidden: int = 8) -> None:
        self.hidden_size = hidden
        self.encode_calls = 0

    def encode(self, texts, batch_size: int = 32):
        self.encode_calls += 1
        import numpy as np

        return np.array([[float(len(t) % self.hidden_size)] * self.hidden_size for t in texts], dtype=np.float32)


def test_column_vectors_cached_per_bundle_with_lru_eviction(tmp_path, shop_bundle):
    enc = _StubEncoder()
    ret = HybridSchemaRetriever(enc)
    n_cols = sum(len(t.columns) for t in shop_bundle.tables)

    v1, o1 = ret.get_column_vectors(shop_bundle)
    assert v1.shape == (n_cols, 8) and len(o1) == len(shop_bundle.tables)
    calls_after_first = enc.encode_calls
    v2, _ = ret.get_column_vectors(shop_bundle)
    assert enc.encode_calls == calls_after_first  # cached, no re-encode
    assert v2.shape == v1.shape

    for i in range(_TABLE_CACHE_MAX + 4):
        b = shop_bundle.model_copy(deep=True)
        b.db_id = f"db_{i}"
        b.ddl_hash = f"h{i}"
        ret.get_column_vectors(b)
    assert len(ret._col_vec_cache) <= _TABLE_CACHE_MAX
    assert len(ret._table_cache) <= _TABLE_CACHE_MAX


# --------------------------------------------------------------------------
# StructCBR query-log miner
# --------------------------------------------------------------------------

def test_query_log_miner_extracts_join_edges_and_examples(tmp_path, shop_db, shop_bundle):
    logs = [
        "SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.amount > 1",
        "SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.amount > 2",
        "SELECT t.status FROM tickets t JOIN customers c ON c.id = t.customer_id",
    ]
    edges = mine_join_edges_from_logs(logs, shop_bundle, min_frequency=2)
    # The declared-FK customers<->orders pair is already known: only NEW pairs count.
    assert all(e.source == "query_log" for e in edges)
    known = {(e.left_table, e.left_column, e.right_table, e.right_column) for e in shop_bundle.join_edges}
    for e in edges:
        assert (e.left_table, e.left_column, e.right_table, e.right_column) not in known

    added = enrich_bundle_from_logs(shop_bundle, logs)
    if added:
        assert any(e.source == "query_log" for e in shop_bundle.join_edges)
    # Deduplicated example queries ride along for parser-prompt hints.
    assert shop_bundle.example_queries


def test_dbt_yaml_descriptions_merge_into_bundle(tmp_path, shop_bundle):
    yaml_text = """
version: 2
models:
  - name: customers
    columns:
      - name: id
        description: primary customer identifier
      - name: name
        description: full customer name
"""
    yml = tmp_path / "schema.yml"
    yml.write_text(yaml_text, encoding="utf-8")
    assert apply_dbt_descriptions(yml, shop_bundle)
    cust = shop_bundle.get_table("customers")
    assert cust.get_column("id").business_description == "primary customer identifier"
    assert cust.get_column("name").business_description == "full customer name"
    assert not apply_dbt_descriptions(tmp_path / "missing.yml", shop_bundle)


# --------------------------------------------------------------------------
# Compiler: UNION LIMIT guardrail
# --------------------------------------------------------------------------

def test_union_without_limit_gets_guardrail_limit(tmp_path, shop_bundle):
    sql = "SELECT name FROM customers UNION SELECT name FROM customers"
    res = compile_and_validate_sql(sql, shop_bundle, default_limit=5)
    assert "LIMIT" in res.sql.upper()
    res0 = compile_and_validate_sql(sql, shop_bundle, default_limit=0)
    assert "LIMIT" not in res0.sql.upper()
