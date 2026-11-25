"""Heavy dataset-preparation and training-infrastructure tests:
schema formatting edge cases, gold-target extraction, missing-DB failure mode,
parser-dataset table budgets, resumable downloads against a Range-ignoring server,
config round-trips, and checkpoint manager edge cases."""
from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest
import torch

from nl2sql.config import DataConfig, EngineConfig, TrainingConfig, load_config, save_config
from nl2sql.datasets.acquire import (
    DatasetAcquisitionResult,
    download_file_resumable,
    _locate_sqlite_root,
)
from nl2sql.datasets.prepare import (
    EmbedderDataset,
    ExampleRecord,
    ParserDataset,
    build_db_caches,
    extract_gold_schema_targets,
    format_schema_for_parser,
    perturb_schema_bundle,
    prepare_records,
)
from nl2sql.db.value_index import ValueIndex, ValueMatch
from nl2sql.training.checkpoint import CheckpointManager
from nl2sql.utils.repro import set_seed

from conftest import SHOP_DDL, create_sqlite


class PlainHTTPHandler(SimpleHTTPRequestHandler):
    """Ignores Range headers entirely (always serves the full file, status 200)."""

    def do_GET(self) -> None:  # noqa: N802
        path = Path(self.translate_path(self.path))
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)  # range request ignored on purpose
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # silence
        pass


@pytest.fixture()
def plain_http_server(tmp_path: Path, monkeypatch):
    serve_dir = tmp_path / "serve_plain"
    serve_dir.mkdir()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    handler = partial(PlainHTTPHandler, directory=str(serve_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield serve_dir, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


class TestSchemaFormatting:
    def test_empty_tables_renders_schema_header(self, shop_bundle) -> None:
        txt = format_schema_for_parser(shop_bundle, [], [], "anything")
        assert txt.startswith("Question:")
        assert "Schema:" in txt
        assert "[TBL]" not in txt

    def test_values_and_evidence_included(self, shop_bundle) -> None:
        vm = [ValueMatch(table_name="customers", column_name="name", value="Alice", matched_phrase="Alice", score=1.0)]
        txt = format_schema_for_parser(shop_bundle, shop_bundle.tables, vm, "q", "focus on tickets")
        assert "customers.name = 'Alice'" in txt
        assert "Hint: focus on tickets" in txt

    def test_long_business_description_truncated(self, shop_bundle) -> None:
        bundle = shop_bundle.model_copy(deep=True)
        bundle.tables[0].columns[1].business_description = "d" * 200
        txt = format_schema_for_parser(bundle, bundle.tables, [], "q")
        assert "d" * 61 not in txt  # hard cap at 60 chars

    def test_fk_lines_only_for_selected_tables(self, shop_bundle) -> None:
        txt = format_schema_for_parser(shop_bundle, shop_bundle.tables[:1], [], "q")
        assert "Foreign Keys" not in txt or "orders" not in txt


class TestGoldExtraction:
    def test_cte_names_not_counted_as_tables(self, shop_bundle) -> None:
        sql = "WITH c AS (SELECT id FROM customers) SELECT id FROM c JOIN orders ON c.id = orders.customer_id"
        tbls, _ = extract_gold_schema_targets(sql, shop_bundle)
        assert set(tbls) == {"customers", "orders"}

    def test_alias_resolution(self, shop_bundle) -> None:
        sql = "SELECT c.name FROM customers c WHERE c.id IN (SELECT o.customer_id FROM orders o)"
        tbls, cols = extract_gold_schema_targets(sql, shop_bundle)
        assert set(tbls) == {"customers", "orders"}
        assert "customers.name" in cols
        assert "orders.customer_id" in cols

    def test_star_yields_no_columns(self, shop_bundle) -> None:
        tbls, cols = extract_gold_schema_targets("SELECT * FROM customers", shop_bundle)
        assert tbls == ["customers"] and cols == []

    def test_unparseable_returns_empty(self, shop_bundle) -> None:
        assert extract_gold_schema_targets("NOT SQL", shop_bundle) == ([], [])

    def test_unknown_tables_ignored(self, shop_bundle) -> None:
        tbls, _ = extract_gold_schema_targets("SELECT * FROM ghost_table", shop_bundle)
        assert tbls == []


class TestPerturbation:
    def test_permutation_preserves_content(self, shop_bundle) -> None:
        torch.manual_seed(0)  # deterministic; use random.Random below
        import random

        before_names = sorted(t.name for t in shop_bundle.tables)
        before_cols = sorted(c.name for t in shop_bundle.tables for c in t.columns)
        new_bundle, sql = perturb_schema_bundle(shop_bundle, "SELECT 1", random.Random(7))
        assert sorted(t.name for t in new_bundle.tables) == before_names
        assert sorted(c.name for t in new_bundle.tables for c in t.columns) == before_cols
        assert sql == "SELECT 1"

    def test_order_actually_changes(self, shop_bundle) -> None:
        import random

        changed = False
        for seed in range(20):
            nb, _ = perturb_schema_bundle(shop_bundle, "", random.Random(seed))
            if [t.name for t in nb.tables] != [t.name for t in shop_bundle.tables]:
                changed = True
                break
        assert changed


class TestBuildDbCaches:
    def test_missing_sqlite_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ghost_db"):
            build_db_caches({"ghost_db": tmp_path / "ghost_db" / "ghost_db.sqlite"}, tmp_path / "cache")

    def test_build_and_reuse_cache(self, tmp_path: Path) -> None:
        db = create_sqlite(tmp_path / "shop.sqlite", SHOP_DDL)
        caches = build_db_caches({"shop": db}, tmp_path / "cache")
        assert "shop" in caches
        bundle, v_idx = caches["shop"]
        assert len(bundle.tables) == 3
        # second call reuses persisted artifacts without rebuilding
        caches2 = build_db_caches({"shop": db}, tmp_path / "cache")
        assert caches2["shop"][0].ddl_hash == bundle.ddl_hash
        assert isinstance(v_idx, ValueIndex)


class TestParserDataset:
    def _records(self, shop_db: Path) -> list[ExampleRecord]:
        return [
            ExampleRecord(
                id="r1", source="t", db_id="shop", db_path=shop_db,
                question="how many open tickets", sql="SELECT COUNT(*) FROM tickets WHERE status = 'open'",
                gold_tables=["tickets"], gold_columns=["tickets.status", "tickets.id"],
            )
        ]

    def test_gold_tables_always_included(self, tmp_path: Path) -> None:
        db = create_sqlite(tmp_path / "shop.sqlite", SHOP_DDL)
        caches = build_db_caches({"shop": db}, tmp_path / "cache")
        ds = ParserDataset(self._records(db), caches)
        item = ds[0]
        names = {t.name for t in item["selected_tables"]}
        assert "tickets" in names
        assert len(item["selected_tables"]) <= 8

    def test_target_sql_canonicalized(self, tmp_path: Path) -> None:
        db = create_sqlite(tmp_path / "shop.sqlite", SHOP_DDL)
        caches = build_db_caches({"shop": db}, tmp_path / "cache")
        ds = ParserDataset(self._records(db), caches)
        item = ds[0]
        assert "status" in item["target_sql"]

    def test_embedder_dataset_with_no_gold_columns(self, tmp_path: Path) -> None:
        db = create_sqlite(tmp_path / "shop.sqlite", SHOP_DDL)
        caches = build_db_caches({"shop": db}, tmp_path / "cache")
        rec = ExampleRecord(id="r", source="t", db_id="shop", db_path=db, question="q", sql="SELECT 1")
        assert len(EmbedderDataset([rec], caches)) == 0


class TestDownloadsAndRoots:
    def test_range_ignoring_server_restart(self, plain_http_server, tmp_path: Path) -> None:
        serve_dir, url = plain_http_server
        payload = b"A" * 5000
        (serve_dir / "blob.bin").write_bytes(payload)
        dest = tmp_path / "out" / "blob.bin"
        # simulate a stale partial download from a previous server
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        part.write_bytes(b"STALE" * 100)  # 500 stale bytes
        result = download_file_resumable(f"{url}/blob.bin", dest, timeout_sec=10)
        assert result == dest
        assert dest.read_bytes() == payload  # server ignored Range -> full restart

    def test_locate_sqlite_root_filters_macosx_and_matches_db_id(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        (root / "db_shop" / "db_shop.sqlite").parent.mkdir(parents=True, exist_ok=True)
        (root / "db_shop" / "db_shop.sqlite").write_bytes(b"")
        (root / "__MACOSX" / "db_shop" / "db_shop.sqlite").parent.mkdir(parents=True, exist_ok=True)
        (root / "__MACOSX" / "db_shop" / "db_shop.sqlite").write_bytes(b"")
        found = _locate_sqlite_root(root, {"db_shop"})
        assert found.name == "root"

    def test_locate_sqlite_root_no_dbs_raises(self, tmp_path: Path) -> None:
        from nl2sql.datasets.acquire import DatasetIntegrityError

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(DatasetIntegrityError):
            _locate_sqlite_root(empty, {"x"})


class TestConfig:
    def test_yaml_roundtrip_and_overrides(self, tmp_path: Path) -> None:
        cfg = EngineConfig(project_name="roundtrip", seed=11, training=TrainingConfig(epochs=2, batch_size=4))
        p = save_config(cfg, tmp_path / "cfg.yaml")
        loaded = load_config(p)
        assert loaded.project_name == "roundtrip" and loaded.seed == 11 and loaded.training.epochs == 2
        over = load_config(p, overrides={"training": {"epochs": 9}})
        assert over.training.epochs == 9 and over.training.batch_size == 4

    def test_invalid_values_rejected(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(epochs=0)
        with pytest.raises(Exception):
            TrainingConfig(learning_rate=-1.0)
        with pytest.raises(Exception):
            DataConfig(val_ratio_by_db=0.9)
        with pytest.raises(Exception):
            TrainingConfig(stage="bogus_stage")

    def test_unknown_top_level_keys_ignored(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.yaml"
        p.write_text("project_name: x\nunknown_key: whatever\n", encoding="utf-8")
        cfg = load_config(p)
        assert cfg.project_name == "x"


class TestCheckpointManager:
    def _mgr(self, tmp_path: Path) -> CheckpointManager:
        return CheckpointManager(tmp_path / "run", keep_last_k=2)

    def test_prune_keeps_only_k(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters())
        for step in (1, 2, 3, 4):
            mgr.save("parser", model, opt, None, None, epoch=step, global_step=step * 10,
                     best_metric=float(step), config_dict={}, metrics_history=[])
        remaining = sorted(d.name for d in (tmp_path / "run" / "checkpoints" / "parser").iterdir() if d.name.startswith("step_"))
        assert len(remaining) == 2
        assert remaining == ["step_0000030", "step_0000040"]  # newest k survive

    def test_best_checkpoint_copied(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters())
        d = mgr.save("parser", model, opt, None, None, epoch=1, global_step=5,
                     best_metric=0.1, config_dict={}, metrics_history=[], is_best=True)
        best = tmp_path / "run" / "checkpoints" / "parser" / "best"
        assert best.exists() and (best / "trainer_state.pt").exists()
        assert d.name == "step_0000005"

    def test_save_same_step_twice_is_idempotent(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters())
        mgr.save("parser", model, opt, None, None, epoch=1, global_step=7, best_metric=1.0, config_dict={}, metrics_history=[])
        mgr.save("parser", model, opt, None, None, epoch=1, global_step=7, best_metric=1.0, config_dict={}, metrics_history=[])
        steps = [d for d in (tmp_path / "run" / "checkpoints" / "parser").iterdir() if d.name.startswith("step_")]
        assert len(steps) == 1

    def test_resolve_explicit_missing_path(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        assert mgr.resolve_resume_path("parser", str(tmp_path / "nope")) is None
        assert mgr.resolve_resume_path("parser", None) is None

    def test_resolve_auto_missing_latest(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        assert mgr.resolve_resume_path("parser", "auto") is None

    def test_rng_roundtrip(self, tmp_path: Path) -> None:
        set_seed(3)
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(model.parameters())
        mgr = self._mgr(tmp_path)
        mgr.save("parser", model, opt, None, None, epoch=1, global_step=1, best_metric=0.0, config_dict={}, metrics_history=[])
        expected = torch.rand(4)
        torch.rand(9)  # perturb RNG
        path = mgr.resolve_resume_path("parser", "auto")
        mgr.load(path, model=model, optimizer=opt)
        assert torch.allclose(torch.rand(4), expected)


class TestPrepareRecordsSplits:
    def test_split_by_db_and_val_ratio(self, tmp_path: Path) -> None:
        root = tmp_path / "dbs"
        db_paths = {}
        for db_id in ("db_a", "db_b", "db_c"):
            db_paths[db_id] = create_sqlite(root / db_id / f"{db_id}.sqlite", SHOP_DDL)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rows = []
        for db_id in db_paths:
            for i in range(3):
                rows.append(
                    {"id": f"{db_id}_{i}", "source": "bird_train", "db_id": db_id,
                     "question": f"q{i}", "evidence": "", "sql": "SELECT COUNT(*) FROM customers"}
                )
        jsonl = data_dir / "train.jsonl"
        jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        acq = DatasetAcquisitionResult(bird_train_jsonl=jsonl, bird_train_db_root=root)
        train, val, test, caches = prepare_records(acq, cache_dir=data_dir / "caches", val_ratio_by_db=1 / 3, seed=42)
        assert len(train) + len(val) == 9
        assert len(val) == 3  # exactly one DB held out
        assert test == []
        assert set(caches) == set(db_paths)
        # val must cover exactly one db_id, train the others
        val_dbs = {r.db_id for r in val}
        train_dbs = {r.db_id for r in train}
        assert len(val_dbs) == 1
        assert val_dbs.isdisjoint(train_dbs)
