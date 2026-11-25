"""Training pipeline tests: multi-stage trainer, checkpoint resume, evaluator metrics,
and full pipeline orchestration (the exact surface the notebook drives)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nl2sql.datasets.acquire import DatasetAcquisitionResult
from nl2sql.datasets.prepare import (
    ExampleRecord,
    build_db_caches,
    extract_gold_schema_targets,
)
from nl2sql.evaluation.evaluator import (
    NL2SQLEvaluator,
    execute_sqlite_readonly,
    results_equivalent,
)
from nl2sql.models.graphix_parser import GraphixCodeT5Parser
from nl2sql.pipeline import NL2SQLPipeline
from nl2sql.training.checkpoint import CheckpointManager
from nl2sql.training.trainer import NL2SQLTrainer

from conftest import SHOP_DDL, WAREHOUSE_DDL, create_sqlite, make_engine_config


def _make_records(db_paths: dict[str, Path], caches: dict) -> list[ExampleRecord]:
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
    return records


@pytest.fixture()
def training_setup(tmp_path: Path, shop_db: Path):
    db_warehouse = create_sqlite(tmp_path / "db_warehouse" / "db_warehouse.sqlite", WAREHOUSE_DDL)
    db_paths = {"shop": shop_db, "warehouse": db_warehouse}
    caches = build_db_caches(db_paths, cache_dir=tmp_path / "schema_caches")
    return _make_records(db_paths, caches), caches


def test_trainer_parser_stage_and_resume(
    tmp_path: Path, tiny_t5_dir: Path, tiny_modernbert_dir: Path, training_setup
) -> None:
    records, caches = training_setup
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=2, stage="parser")
    trainer = NL2SQLTrainer(cfg)
    res = trainer.train_parser(records, records, caches)

    assert len(res["history"]) == 2
    for rec in res["history"]:
        assert rec["stage"] == "parser"
        assert rec["train_loss"] > 0 and "val_loss" in rec and "lr" in rec
    ckpt_root = Path(cfg.training.output_dir) / "checkpoints" / "parser"
    assert (ckpt_root / "latest.txt").exists()
    artifact_dir = Path(cfg.training.output_dir) / "artifacts" / "parser"
    assert (artifact_dir / "config.json").exists()

    # Best artifact is reloadable as a full parser
    reloaded = GraphixCodeT5Parser.from_pretrained_dir(artifact_dir, map_location="cpu")
    assert reloaded.rel_bias_emb.weight.shape == (5, 4)

    # Interrupted-training resume: a new trainer with more epochs continues from the checkpoint
    cfg2 = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=3, stage="parser")
    trainer2 = NL2SQLTrainer(cfg2)
    res2 = trainer2.train_parser(records, records, caches)
    assert len(res2["history"]) == 3  # 2 prior epochs restored + 1 new
    assert res2["history"][0] == res["history"][0]
    assert res2["history"][1] == res["history"][1]

    # Checkpoint retention keeps only the last k step directories
    step_dirs = [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("step_")]
    assert len(step_dirs) <= cfg.training.keep_last_k_checkpoints


def test_trainer_embedder_stage(
    tmp_path: Path, tiny_t5_dir: Path, tiny_modernbert_dir: Path, training_setup
) -> None:
    records, caches = training_setup
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="embedder")
    res = NL2SQLTrainer(cfg).train_embedder(records, records, caches)
    assert len(res["history"]) == 1 and res["history"][0]["train_loss"] > 0
    assert (Path(cfg.training.output_dir) / "artifacts" / "embedder").exists()


def test_trainer_linker_stage(
    tmp_path: Path, tiny_t5_dir: Path, tiny_modernbert_dir: Path, training_setup
) -> None:
    records, caches = training_setup
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="linker")
    res = NL2SQLTrainer(cfg).train_linker(records, records, caches)
    assert len(res["history"]) == 1 and res["history"][0]["train_loss"] > 0
    assert (Path(cfg.training.output_dir) / "artifacts" / "linker" / "linker_weights.pt").exists()


def test_trainer_all_stages_including_verifier(
    tmp_path: Path, tiny_t5_dir: Path, tiny_modernbert_dir: Path, training_setup
) -> None:
    records, caches = training_setup
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="all")
    pipeline = NL2SQLPipeline(cfg)
    pipeline.train_records, pipeline.val_records, pipeline.caches = records, records, caches
    results = pipeline.train(stage="all")

    assert set(results) == {"embedder", "linker", "parser", "verifier"}
    for stage, res in results.items():
        assert res["history"], f"stage {stage} produced no history"
        assert (Path(cfg.training.output_dir) / "artifacts" / stage).exists()

    # Verifier was trained on execution-divergence mined negatives
    weights = Path(cfg.training.output_dir) / "artifacts" / "verifier" / "verifier_weights.pt"
    assert weights.exists() and weights.stat().st_size > 0


def test_trainer_verifier_requires_parser_artifacts(
    tmp_path: Path, tiny_t5_dir: Path, tiny_modernbert_dir: Path, training_setup
) -> None:
    records, caches = training_setup
    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="verifier")
    with pytest.raises(FileNotFoundError, match="parser"):
        NL2SQLTrainer(cfg).train_verifier(records, records, caches)


def test_execute_sqlite_readonly_and_results_equivalence(shop_db: Path) -> None:
    ok, rows, ms, msg = execute_sqlite_readonly(shop_db, "SELECT name FROM customers ORDER BY id")
    assert ok and rows == [("Alice",), ("Bob",)] and ms >= 0

    ok_err, rows_err, _, msg_err = execute_sqlite_readonly(shop_db, "SELECT * FROM missing_table")
    assert not ok_err and rows_err is None and msg_err

    # Read-only connection rejects writes outright
    ok_w, _, _, msg_w = execute_sqlite_readonly(shop_db, "DELETE FROM customers")
    assert not ok_w and "readonly" in msg_w.lower()

    # Unordered result equivalence (no ORDER BY in gold)
    assert results_equivalent("SELECT name FROM customers", [("Alice",), ("Bob",)], [("Bob",), ("Alice",)])
    # Ordered equivalence requires exact ordering
    assert not results_equivalent(
        "SELECT name FROM customers ORDER BY name", [("Alice",), ("Bob",)], [("Bob",), ("Alice",)]
    )
    # Floats are compared at 4-decimal precision
    assert results_equivalent("SELECT amount FROM orders", [(120.5,)], [(120.50001,)])


def test_evaluator_metrics(shop_db: Path, shop_bundle) -> None:
    g_tbls, g_cols = extract_gold_schema_targets(
        "SELECT COUNT(*) FROM tickets WHERE status = 'open'", shop_bundle
    )
    records = [
        ExampleRecord(id="correct", source="t", db_id="shop", db_path=shop_db,
                      question="How many open tickets for Alice?", sql="SELECT COUNT(*) FROM tickets WHERE status = 'open'",
                      gold_tables=g_tbls, gold_columns=g_cols),
        ExampleRecord(id="wrong", source="t", db_id="shop", db_path=shop_db,
                      question="Total amount", sql="SELECT SUM(amount) FROM orders",
                      gold_tables=["orders"], gold_columns=["orders.amount"]),
        ExampleRecord(id="abstain", source="t", db_id="shop", db_path=shop_db,
                      question="Ghost question", sql="SELECT COUNT(*) FROM tickets",
                      gold_tables=["tickets"], gold_columns=["tickets.id"]),
    ]

    def predict_fn(rec: ExampleRecord) -> dict:
        if rec.id == "correct":
            return {"sql": rec.sql, "abstained": False, "confidence": 0.9, "linked_tables": rec.gold_tables}
        if rec.id == "wrong":
            return {"sql": "SELECT 1", "abstained": False, "confidence": 0.6, "linked_tables": []}
        return {"sql": None, "abstained": True, "confidence": 0.2, "linked_tables": []}

    metrics = NL2SQLEvaluator().evaluate(records, predict_fn, dataset_name="unit")
    assert metrics.total_examples == 3
    assert metrics.execution_accuracy == round(1 / 3, 4)
    assert metrics.exact_match == round(1 / 3, 4)
    assert metrics.executable_rate == round(2 / 3, 4)  # wrong-but-valid SQL still executes
    assert metrics.abstention_rate == round(1 / 3, 4)
    assert metrics.selective_execution_accuracy == round(1 / 2, 4)
    assert metrics.verifier_auroc == 1.0
    assert metrics.schema_table_recall > 0.0
    assert metrics.p50_latency_ms >= 0 and metrics.p95_latency_ms >= metrics.p50_latency_ms
    assert metrics.throughput_qps > 0


def test_checkpoint_manager_atomic_prune_and_best(tmp_path: Path) -> None:
    from nl2sql.utils.repro import set_seed

    set_seed(3)
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path / "run", keep_last_k=2)

    for step in (1, 2, 3):
        with torch.no_grad():
            model.weight.add_(0.1)
        mgr.save("parser", model, opt, None, None, epoch=step, global_step=step * 5,
                 best_metric=float(step), config_dict={"a": 1}, metrics_history=[], is_best=(step == 3))

    stage_dir = tmp_path / "run" / "checkpoints" / "parser"
    remaining = sorted(d.name for d in stage_dir.iterdir() if d.name.startswith("step_"))
    assert remaining == ["step_0000010", "step_0000015"]  # oldest pruned
    assert (stage_dir / "best").exists()
    # "best" mirrors the newest checkpoint (highest best_metric)
    best_state = torch.load(stage_dir / "best" / "trainer_state.pt", map_location="cpu", weights_only=False)
    assert best_state["best_metric"] == 3.0
    # meta.json mirrors the payload
    meta = json.loads((stage_dir / "best" / "meta.json").read_text())
    assert meta["global_step"] == 15
    # explicit resume path resolution
    assert mgr.resolve_resume_path("parser", str(stage_dir / "step_0000015")) == stage_dir / "step_0000015"
    assert mgr.resolve_resume_path("parser", "auto") == stage_dir / "step_0000015"
    assert mgr.resolve_resume_path("parser", "/nonexistent/path") is None


def test_pipeline_end_to_end(
    tmp_path: Path, tiny_t5_dir: Path, tiny_modernbert_dir: Path, monkeypatch
) -> None:
    # -- synthetic acquisition result standing in for the downloaded BIRD corpus --
    data_dir = tmp_path / "data"
    # The three SQLite files only need to exist for acquisition/introspection.
    create_sqlite(data_dir / "train_dbs" / "shop" / "shop.sqlite", SHOP_DDL)
    create_sqlite(data_dir / "train_dbs" / "warehouse" / "warehouse.sqlite", WAREHOUSE_DDL)
    create_sqlite(data_dir / "dev_dbs" / "dev_shop" / "dev_shop.sqlite", SHOP_DDL)

    train_rows = [
        {"id": "t1", "source": "bird_train", "db_id": "shop", "question": "How many open tickets for Alice?", "evidence": "", "sql": "SELECT COUNT(*) FROM tickets WHERE status = 'open'"},
        {"id": "t2", "source": "bird_train", "db_id": "shop", "question": "What is the total order amount for Bob?", "evidence": "", "sql": "SELECT SUM(amount) FROM orders JOIN customers ON orders.customer_id = customers.id WHERE customers.name = 'Bob'"},
        {"id": "t3", "source": "bird_train", "db_id": "warehouse", "question": "Total revenue of all stores", "evidence": "", "sql": "SELECT SUM(revenue) FROM sales"},
        {"id": "t4", "source": "bird_train", "db_id": "warehouse", "question": "How many stores are in the East region?", "evidence": "", "sql": "SELECT COUNT(*) FROM stores JOIN regions ON stores.region_id = regions.id WHERE regions.region_name = 'East'"},
    ]
    dev_rows = [
        {"id": "d1", "source": "bird_dev", "db_id": "dev_shop", "question": "How many open tickets?", "evidence": "", "sql": "SELECT COUNT(*) FROM tickets WHERE status = 'open'"},
    ]

    def _write_jsonl(rows: list[dict], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return path

    acq = DatasetAcquisitionResult(
        bird_train_jsonl=_write_jsonl(train_rows, data_dir / "prepared" / "bird_train.jsonl"),
        bird_train_db_root=data_dir / "train_dbs",
        bird_dev_jsonl=_write_jsonl(dev_rows, data_dir / "prepared" / "bird_dev.jsonl"),
        bird_dev_db_root=data_dir / "dev_dbs",
    )

    def fake_acquire(cfg_data):
        return acq

    monkeypatch.setattr("nl2sql.pipeline.acquire_all_datasets", fake_acquire)

    cfg = make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir, epochs=1, stage="parser")
    pipeline = NL2SQLPipeline(cfg)

    summary = pipeline.prepare_data()
    assert summary["train_examples"] + summary["val_examples"] == 4
    assert summary["test_examples"] == 1
    assert summary["cached_databases"] == 3
    # Config snapshot persisted for reproducibility
    assert (Path(cfg.training.output_dir) / "config_snapshot.yaml").exists()

    results = pipeline.train(stage="parser")
    assert "parser" in results and results["parser"]["history"]

    # Evaluate with a parser oracle so the answer path exercises compile/execute/metrics
    q2sql = {r.question: r.sql for r in pipeline.val_records + pipeline.test_records}

    def fake_generate(self, source_text, bundle, tables, value_matches, **kwargs):
        for q, s in q2sql.items():
            if q in source_text:
                return [s]
        return ["SELECT 1"]

    monkeypatch.setattr(GraphixCodeT5Parser, "generate_candidates", fake_generate)

    metrics_val = pipeline.evaluate(split="val", use_oracle_evidence=False)
    assert metrics_val.total_examples == len(pipeline.val_records)
    assert metrics_val.execution_accuracy > 0.5
    assert (Path(cfg.training.output_dir) / "eval" / "val_no_evidence.json").exists()

    metrics_test = pipeline.evaluate(split="test", use_oracle_evidence=True)
    assert metrics_test.total_examples == 1

    # Hub upload is a no-op when disabled
    assert pipeline.push_artifacts() is None
