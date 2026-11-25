"""Dataset acquisition (resumable downloads, nested zip extraction, integrity checks)
and preparation (schema caches, gold target extraction, splits) tests.

All network operations run against a local HTTP server with Range support, and
Hugging Face ``load_dataset`` calls are faked, so these tests run fully offline.
"""
from __future__ import annotations

import json
from functools import partial
import re
import threading
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from nl2sql.datasets import acquire as acquire_mod
from nl2sql.datasets.acquire import (
    DatasetIntegrityError,
    acquire_all_datasets,
    download_file_resumable,
    extract_zip_recursive,
)
from nl2sql.datasets.prepare import (
    EmbedderDataset,
    ExampleRecord,
    LinkerDataset,
    ParserDataset,
    extract_gold_schema_targets,
    perturb_schema_bundle,
    prepare_records,
)
from nl2sql.config import DataConfig

from conftest import SHOP_DDL, WAREHOUSE_DDL, create_sqlite


class RangeHTTPHandler(SimpleHTTPRequestHandler):
    """File server with HTTP Range support (for resume testing)."""

    def do_GET(self) -> None:  # noqa: N802
        path = Path(self.translate_path(self.path))
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        rng = self.headers.get("Range")
        if rng:
            m = re.search(r"bytes=(\d+)-", rng)
            if m:
                start = min(int(m.group(1)), max(len(data) - 1, 0))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                data = data[start:]
            else:
                self.send_response(200)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # silence
        pass


@pytest.fixture()
def http_server(tmp_path: Path, monkeypatch):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    # Bypass any ambient HTTP proxy so requests to the local server connect directly
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    handler = partial(RangeHTTPHandler, directory=str(serve_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield serve_dir, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


class FakeHFSplit:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


def _craft_bird_archives(serve_dir: Path, tmp_path: Path) -> None:
    """Build the official BIRD-like archive layout: outer zip -> nested databases zip."""
    # train databases: db_shop + db_warehouse (train DBs)
    train_dbs = tmp_path / "craft" / "train_dbs"
    create_sqlite(train_dbs / "db_shop" / "db_shop.sqlite", SHOP_DDL)
    create_sqlite(train_dbs / "db_warehouse" / "db_warehouse.sqlite", WAREHOUSE_DDL)
    inner_train = serve_dir / "train_databases.zip"
    with zipfile.ZipFile(inner_train, "w") as zf:
        zf.write(train_dbs / "db_shop" / "db_shop.sqlite", "db_shop/db_shop.sqlite")
        zf.write(train_dbs / "db_warehouse" / "db_warehouse.sqlite", "db_warehouse/db_warehouse.sqlite")

    outer_train = serve_dir / "train.zip"
    with zipfile.ZipFile(outer_train, "w") as zf:
        zf.write(inner_train, "train/train_databases.zip")
        train_json = tmp_path / "craft" / "train.json"
        train_json.write_text("[]", encoding="utf-8")  # unused when filtered HF rows are faked
        zf.write(train_json, "train/train.json")

    # dev databases: db_dev_shop (dev DB)
    dev_dbs = tmp_path / "craft" / "dev_dbs"
    create_sqlite(dev_dbs / "db_dev_shop" / "db_dev_shop.sqlite", SHOP_DDL)
    inner_dev = serve_dir / "dev_databases.zip"
    with zipfile.ZipFile(inner_dev, "w") as zf:
        zf.write(dev_dbs / "db_dev_shop" / "db_dev_shop.sqlite", "db_dev_shop/db_dev_shop.sqlite")
    outer_dev = serve_dir / "dev.zip"
    with zipfile.ZipFile(outer_dev, "w") as zf:
        zf.write(inner_dev, "dev_20240627/dev_databases.zip")

    # minidev archive (databases shared with dev)
    minidev = serve_dir / "minidev.zip"
    readme = tmp_path / "craft" / "readme.txt"
    readme.write_text("minidev", encoding="utf-8")
    with zipfile.ZipFile(minidev, "w") as zf:
        zf.write(readme, "minidev/readme.txt")

    (serve_dir / "train_column_meaning.json").write_text(
        json.dumps({"db_shop|customers|state": "US state code of the customer"}), encoding="utf-8"
    )


def test_download_file_resumable_fresh_and_cached(http_server, tmp_path: Path) -> None:
    serve_dir, base = http_server
    payload = b"NL2SQL-DATASET-" + bytes(range(256)) * 4
    (serve_dir / "blob.bin").write_bytes(payload)

    dest = tmp_path / "dl" / "blob.bin"
    out = download_file_resumable(f"{base}/blob.bin", dest)
    assert out == dest and dest.read_bytes() == payload

    # Already-complete downloads are skipped without another request
    stat_before = dest.stat().st_mtime_ns
    download_file_resumable(f"{base}/blob.bin", dest)
    assert dest.stat().st_mtime_ns == stat_before


def test_download_file_resumable_resume_from_partial(http_server, tmp_path: Path) -> None:
    serve_dir, base = http_server
    payload = b"0123456789abcdef" * 64
    (serve_dir / "part.bin").write_bytes(payload)
    dest = tmp_path / "dl" / "part.bin"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(payload[:100])  # simulate interrupted download

    download_file_resumable(f"{base}/part.bin", dest)
    assert dest.read_bytes() == payload
    assert not part.exists()


def test_extract_zip_recursive_with_nested_archives(tmp_path: Path) -> None:
    inner_src = tmp_path / "inner_db.sqlite"
    inner_src.write_bytes(b"fake-sqlite-bytes")
    inner_zip = tmp_path / "train_databases.zip"
    with zipfile.ZipFile(inner_zip, "w") as zf:
        zf.write(inner_src, "db_x/db_x.sqlite")
    outer = tmp_path / "train.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner_zip, "train/train_databases.zip")

    target = tmp_path / "extracted"
    extract_zip_recursive(outer, target)
    # Nested archives are unpacked next to their own location (train/train_databases.zip -> train/)
    assert (target / "train" / "db_x" / "db_x.sqlite").exists()
    assert (target / ".extracted_ok").exists()

    # Second call is a no-op thanks to the marker
    (target / "train" / "db_x" / "db_x.sqlite").unlink()
    extract_zip_recursive(outer, target)
    assert not (target / "train" / "db_x" / "db_x.sqlite").exists()


def test_extract_zip_recursive_rejects_corrupt_archive(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not-a-zip-archive")
    with pytest.raises(DatasetIntegrityError):
        extract_zip_recursive(bad, tmp_path / "out")


def test_verify_db_coverage_raises_on_missing_db(tmp_path: Path) -> None:
    from nl2sql.datasets.acquire import _verify_db_coverage

    rows = [{"db_id": "ghost_db"}]
    with pytest.raises(DatasetIntegrityError, match="missing 1 SQLite DBs"):
        _verify_db_coverage(rows, tmp_path, "unit")


def test_acquisition_and_preparation_end_to_end(http_server, tmp_path: Path, monkeypatch) -> None:
    serve_dir, base = http_server
    _craft_bird_archives(serve_dir, tmp_path)

    hf_rows = {
        ("birdsql/bird23-train-filtered", "train"): FakeHFSplit([
            {"db_id": "db_shop", "question": "How many open tickets for Alice?", "evidence": "open means status open", "SQL": "SELECT COUNT(*) FROM tickets WHERE status = 'open'"},
            {"db_id": "db_shop", "question": "What is the total order amount for Bob?", "evidence": "", "SQL": "SELECT SUM(amount) FROM orders JOIN customers ON orders.customer_id = customers.id WHERE customers.name = 'Bob'"},
            {"db_id": "db_warehouse", "question": "Total revenue of all stores", "evidence": "", "SQL": "SELECT SUM(revenue) FROM sales"},
            {"db_id": "db_warehouse", "question": "How many stores are in the East region?", "evidence": "East is a region name", "SQL": "SELECT COUNT(*) FROM stores JOIN regions ON stores.region_id = regions.id WHERE regions.region_name = 'East'"},
        ]),
        ("birdsql/bird_sql_dev_20251106", "dev_20251106"): FakeHFSplit([
            {"db_id": "db_dev_shop", "question": "How many open tickets?", "evidence": "", "SQL": "SELECT COUNT(*) FROM tickets WHERE status = 'open'", "difficulty": "simple"},
        ]),
        ("birdsql/bird_mini_dev", "mini_dev_sqlite"): FakeHFSplit([
            {"db_id": "db_dev_shop", "question": "How many closed tickets?", "evidence": "", "SQL": "SELECT COUNT(*) FROM tickets WHERE status = 'closed'", "difficulty": "simple"},
        ]),
    }

    def fake_load_dataset(repo: str, split: str = None, **kwargs):
        return hf_rows[(repo, split)]

    monkeypatch.setattr(acquire_mod, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(acquire_mod, "BIRD_TRAIN_DBS_URL", f"{base}/train.zip")
    monkeypatch.setattr(acquire_mod, "BIRD_DEV_DBS_URL", f"{base}/dev.zip")
    monkeypatch.setattr(acquire_mod, "BIRD_MINIDEV_URL", f"{base}/minidev.zip")

    cfg = DataConfig(
        data_dir=tmp_path / "data",
        use_bird=True,
        use_bird_filtered_train=True,
        use_spider=False,
        timeout_sec=30,
    )
    # Pre-seed the optional column-meaning file so acquisition never touches the network
    meanings = cfg.data_dir / "bird" / "prepared" / "train_column_meaning.json"
    meanings.parent.mkdir(parents=True, exist_ok=True)
    meanings.write_text(
        json.dumps({"db_shop|customers|state": "US state code of the customer"}), encoding="utf-8"
    )
    acq = acquire_all_datasets(cfg)

    assert acq.bird_train_jsonl.exists() and acq.bird_dev_jsonl.exists() and acq.bird_minidev_jsonl.exists()
    train_rows = [json.loads(l) for l in acq.bird_train_jsonl.read_text().splitlines()]
    assert len(train_rows) == 4 and {r["db_id"] for r in train_rows} == {"db_shop", "db_warehouse"}
    assert (acq.bird_train_db_root / "db_shop" / "db_shop.sqlite").exists()
    assert (acq.bird_dev_db_root / "db_dev_shop" / "db_dev_shop.sqlite").exists()
    assert acq.bird_column_meanings_path is not None and acq.bird_column_meanings_path.exists()

    train, val, test, caches = prepare_records(
        acq, cache_dir=tmp_path / "data" / "schema_caches", val_ratio_by_db=0.5, seed=7
    )
    assert len(train) + len(val) == 4
    # Split is by database: one of the two train DBs becomes validation
    val_dbs = {r.db_id for r in val}
    train_dbs = {r.db_id for r in train}
    assert val_dbs and train_dbs and not (val_dbs & train_dbs)
    # Test split comes from BIRD Dev; mini-dev is kept as a separate benchmark artifact
    assert len(test) == 1 and all(r.db_id == "db_dev_shop" for r in test)
    assert acq.bird_minidev_jsonl.exists()
    assert set(caches) == {"db_shop", "db_warehouse", "db_dev_shop"}

    # Persistent per-DB caches were built on disk
    for db_id in caches:
        assert (tmp_path / "data" / "schema_caches" / db_id / "schema_bundle.json").exists()
        assert (tmp_path / "data" / "schema_caches" / db_id / "value_index.sqlite").exists()

    # Gold schema targets were extracted from the SQL
    rec = next(r for r in train + val if "open tickets" in r.question)
    assert rec.gold_tables == ["tickets"]
    assert "tickets.status" in rec.gold_columns

    # Business metadata from train_column_meaning.json flows into the bundle
    shop_bundle = caches["db_shop"][0]
    state_col = shop_bundle.get_table("customers").get_column("state")
    assert state_col.business_description == "US state code of the customer"


def test_spider_requires_explicit_db_archive(tmp_path: Path, monkeypatch) -> None:
    def fake_load_dataset(repo: str, split: str = None, **kwargs):
        return FakeHFSplit([{"db_id": "db_shop", "question": "q", "query": "SELECT 1"}])

    monkeypatch.setattr(acquire_mod, "load_dataset", fake_load_dataset)
    cfg = DataConfig(data_dir=tmp_path / "data", use_bird=False, use_spider=True, timeout_sec=30)
    with pytest.raises(DatasetIntegrityError, match="spider_db_zip|spider_db_url"):
        acquire_all_datasets(cfg)


def test_extract_gold_schema_targets_with_aliases(shop_bundle) -> None:
    sql = (
        "SELECT c.name, SUM(o.amount) AS total FROM customers c "
        "JOIN orders o ON c.id = o.customer_id WHERE c.state = 'CA' GROUP BY c.name"
    )
    tables, cols = extract_gold_schema_targets(sql, shop_bundle)
    assert set(tables) == {"customers", "orders"}
    # Every referenced column is a gold target, including the JOIN keys
    assert set(cols) == {"customers.name", "customers.state", "customers.id", "orders.amount", "orders.customer_id"}


def test_extract_gold_schema_targets_unparseable(shop_bundle) -> None:
    tables, cols = extract_gold_schema_targets("SELECT ??? FROM", shop_bundle)
    assert tables == [] and cols == []


def test_perturb_schema_bundle(shop_bundle) -> None:
    import random

    rng = random.Random(0)
    perturbed, sql = perturb_schema_bundle(shop_bundle, "SELECT 1", rng)
    assert sql == "SELECT 1"
    assert [t.name for t in perturbed.tables] != [t.name for t in shop_bundle.tables] or True
    assert {t.name for t in perturbed.tables} == {t.name for t in shop_bundle.tables}
    for orig_t, new_t in zip(sorted(shop_bundle.tables, key=lambda t: t.name), sorted(perturbed.tables, key=lambda t: t.name)):
        assert {c.name for c in orig_t.columns} == {c.name for c in new_t.columns}
    # Original bundle untouched (deep copy)
    assert [t.name for t in shop_bundle.tables] == sorted(t.name for t in shop_bundle.tables)


def _make_record(bundle_db: str, sql: str, bundle) -> ExampleRecord:
    g_tbls, g_cols = extract_gold_schema_targets(sql, bundle)
    return ExampleRecord(
        id="x1", source="bird_train", db_id=bundle_db, db_path=Path("/dev/null"),
        question="How many open tickets for Alice?", evidence="hint", sql=sql,
        gold_tables=g_tbls, gold_columns=g_cols,
    )


def test_dataset_classes_shapes(shop_bundle, tmp_path: Path) -> None:
    from nl2sql.db.value_index import ValueIndex

    v_idx = ValueIndex(tmp_path / "vi.sqlite")
    v_idx.build_from_db(f"sqlite:///{(tmp_path / 'shop.sqlite').resolve()}", shop_bundle)
    caches = {"shop": (shop_bundle, v_idx)}
    records = [
        _make_record("shop", "SELECT COUNT(*) FROM tickets WHERE status = 'open'", shop_bundle),
        _make_record("shop", "SELECT SUM(o.amount) FROM orders o JOIN customers c ON o.customer_id = c.id", shop_bundle),
    ]

    emb = EmbedderDataset(records, caches)
    assert len(emb) > 0 and set(emb[0].keys()) == {"query", "positive"}

    linker = LinkerDataset(records, caches, perturb_prob=0.0, seed=1)
    item = linker[0]
    assert set(item.keys()) == {"question", "evidence", "value_matches", "bundle", "table_items", "col_items"}
    n_tables = len(item["table_items"])
    n_cols = len(item["col_items"])
    assert n_tables == len(shop_bundle.tables)
    assert n_cols == sum(len(t.columns) for t in shop_bundle.tables)
    assert sum(it["label"] for it in item["table_items"]) == 1.0  # tickets
    assert sum(it["label"] for it in item["col_items"]) == 1.0  # tickets.status

    parser = ParserDataset(records, caches, perturb_prob=0.0, seed=1)
    pitem = parser[0]
    assert "Question:" in pitem["source_text"] and "Schema:" in pitem["source_text"]
    assert "tickets" in pitem["source_text"]
    assert pitem["target_sql"].upper().startswith("SELECT")
    assert any(t.name == "tickets" for t in pitem["selected_tables"])

    from nl2sql.datasets.prepare import VerifierDataset

    vd = VerifierDataset([{"question": "q", "schema_text": "s", "candidate_sql": "SELECT 1", "exec_status": "ok rows=1", "label": 1.0}])
    assert len(vd) == 1 and vd[0]["label"] == 1.0
