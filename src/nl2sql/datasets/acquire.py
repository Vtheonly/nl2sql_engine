from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Optional

import requests
from datasets import load_dataset
from pydantic import BaseModel
from tqdm import tqdm

from nl2sql.config import DataConfig
from nl2sql.utils.logging import get_logger

logger = get_logger(__name__)

BIRD_TRAIN_DBS_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
BIRD_DEV_DBS_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BIRD_MINIDEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"


class DatasetIntegrityError(RuntimeError):
    """Raised when required datasets or SQLite database files are missing or corrupted."""


class DatasetAcquisitionResult(BaseModel):
    bird_train_jsonl: Optional[Path] = None
    bird_dev_jsonl: Optional[Path] = None
    bird_minidev_jsonl: Optional[Path] = None
    bird_column_meanings_path: Optional[Path] = None
    bird_train_db_root: Optional[Path] = None
    bird_dev_db_root: Optional[Path] = None
    spider_train_jsonl: Optional[Path] = None
    spider_dev_jsonl: Optional[Path] = None
    spider_db_root: Optional[Path] = None


def download_file_resumable(url: str, dest_path: Path, timeout_sec: int = 120) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    resume_pos = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

    with requests.get(url, headers=headers, stream=True, timeout=timeout_sec) as resp:
        if resp.status_code == 416:
            part_path.rename(dest_path)
            return dest_path
        resp.raise_for_status()
        mode = "ab" if (resume_pos > 0 and resp.status_code == 206) else "wb"
        if mode == "wb":
            resume_pos = 0
        total = int(resp.headers.get("content-length", 0)) + resume_pos
        with open(part_path, mode) as f, tqdm(
            total=total, initial=resume_pos, unit="B", unit_scale=True, desc=dest_path.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    part_path.rename(dest_path)
    return dest_path


def extract_zip_recursive(archive_path: Path, target_dir: Path, marker_name: str = ".extracted_ok") -> Path:
    marker = target_dir / marker_name
    if marker.exists():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile as exc:
        raise DatasetIntegrityError(f"Corrupted archive {archive_path}: {exc}") from exc

    for nested in list(target_dir.rglob("*.zip")):
        if nested.name in ("train_databases.zip", "dev_databases.zip", "database.zip"):
            with zipfile.ZipFile(nested, "r") as nzf:
                nzf.extractall(nested.parent)
    marker.write_text("ok", encoding="utf-8")
    return target_dir


def _locate_sqlite_root(search_dir: Path, expected_db_ids: set[str]) -> Path:
    candidates = [p for p in search_dir.rglob("*.sqlite") if "__MACOSX" not in str(p)]
    if not candidates:
        raise DatasetIntegrityError(f"No .sqlite database files found inside {search_dir}")
    # Map db_id -> parent of <db_id>/<db_id>.sqlite
    for p in candidates:
        if p.stem in expected_db_ids and p.parent.name == p.stem:
            return p.parent.parent
    return candidates[0].parent.parent


def _verify_db_coverage(records: list[dict], db_root: Path, dataset_name: str) -> None:
    missing: set[str] = set()
    for row in records:
        db_id = row["db_id"]
        db_file = db_root / db_id / f"{db_id}.sqlite"
        if not db_file.exists():
            missing.add(db_id)
    if missing:
        raise DatasetIntegrityError(
            f"{dataset_name} integrity check failed: missing {len(missing)} SQLite DBs under {db_root}. "
            f"Examples: {sorted(missing)[:5]}"
        )


def _write_jsonl(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def acquire_all_datasets(cfg: DataConfig) -> DatasetAcquisitionResult:
    root = cfg.data_dir
    root.mkdir(parents=True, exist_ok=True)
    res = DatasetAcquisitionResult()

    if cfg.use_bird:
        bird_dir = root / "bird"
        bird_archives = bird_dir / "archives"
        logger.info("Acquiring BIRD datasets & databases...")

        # 1. BIRD Train (Filtered 6,601 or full) + train_databases
        train_zip = download_file_resumable(BIRD_TRAIN_DBS_URL, bird_archives / "train.zip", cfg.timeout_sec)
        train_extracted = extract_zip_recursive(train_zip, bird_dir / "train_raw")
        if cfg.use_bird_filtered_train:
            ds_train = load_dataset("birdsql/bird23-train-filtered", split="train")
            train_rows = [
                {
                    "id": f"bird_train_{i}",
                    "source": "bird_train",
                    "db_id": r["db_id"],
                    "question": r["question"],
                    "evidence": r.get("evidence") or "",
                    "sql": r["SQL"],
                }
                for i, r in enumerate(ds_train)
            ]
        else:
            raw_train_json = next(train_extracted.rglob("train.json"))
            raw_items = json.loads(raw_train_json.read_text(encoding="utf-8"))
            train_rows = [
                {
                    "id": f"bird_train_{i}",
                    "source": "bird_train",
                    "db_id": r["db_id"],
                    "question": r["question"],
                    "evidence": r.get("evidence") or "",
                    "sql": r["SQL"],
                }
                for i, r in enumerate(raw_items)
            ]

        train_db_ids = {r["db_id"] for r in train_rows}
        train_db_root = _locate_sqlite_root(train_extracted, train_db_ids)
        _verify_db_coverage(train_rows, train_db_root, "BIRD Train")
        res.bird_train_jsonl = _write_jsonl(train_rows, bird_dir / "prepared" / "bird_train.jsonl")
        res.bird_train_db_root = train_db_root

        # Column meaning json (best-effort official metadata)
        meanings_path = bird_dir / "prepared" / "train_column_meaning.json"
        if not meanings_path.exists():
            try:
                download_file_resumable(
                    "https://huggingface.co/datasets/birdsql/bird23-train-filtered/resolve/main/train_column_meaning.json",
                    meanings_path,
                    cfg.timeout_sec,
                )
            except Exception as exc:
                logger.warning("Optional train_column_meaning.json download failed: %s", exc)
        if meanings_path.exists():
            res.bird_column_meanings_path = meanings_path

        # 2. BIRD Dev (Official 2025-11-06 revision + dev_databases)
        dev_zip = download_file_resumable(BIRD_DEV_DBS_URL, bird_archives / "dev.zip", cfg.timeout_sec)
        dev_extracted = extract_zip_recursive(dev_zip, bird_dir / "dev_raw")
        ds_dev = load_dataset("birdsql/bird_sql_dev_20251106", split="dev_20251106")
        dev_rows = [
            {
                "id": f"bird_dev_{i}",
                "source": "bird_dev",
                "db_id": r["db_id"],
                "question": r["question"],
                "evidence": r.get("evidence") or "",
                "sql": r["SQL"],
                "difficulty": r.get("difficulty", "unknown"),
            }
            for i, r in enumerate(ds_dev)
        ]
        dev_db_ids = {r["db_id"] for r in dev_rows}
        dev_db_root = _locate_sqlite_root(dev_extracted, dev_db_ids)
        _verify_db_coverage(dev_rows, dev_db_root, "BIRD Dev")
        res.bird_dev_jsonl = _write_jsonl(dev_rows, bird_dir / "prepared" / "bird_dev.jsonl")
        res.bird_dev_db_root = dev_db_root

        # 3. BIRD Mini-Dev SQLite
        minidev_zip = download_file_resumable(BIRD_MINIDEV_URL, bird_archives / "minidev.zip", cfg.timeout_sec)
        extract_zip_recursive(minidev_zip, bird_dir / "minidev_raw")
        ds_mini = load_dataset("birdsql/bird_mini_dev", split="mini_dev_sqlite")
        mini_rows = [
            {
                "id": f"bird_minidev_{i}",
                "source": "bird_minidev",
                "db_id": r["db_id"],
                "question": r["question"],
                "evidence": r.get("evidence") or "",
                "sql": r["SQL"],
                "difficulty": r.get("difficulty", "unknown"),
            }
            for i, r in enumerate(ds_mini)
        ]
        _verify_db_coverage(mini_rows, dev_db_root, "BIRD Mini-Dev")
        res.bird_minidev_jsonl = _write_jsonl(mini_rows, bird_dir / "prepared" / "bird_minidev.jsonl")
        logger.info(
            "BIRD ready: train=%d, dev=%d, minidev=%d",
            len(train_rows), len(dev_rows), len(mini_rows),
        )

    if cfg.use_spider:
        spider_dir = root / "spider"
        logger.info("Acquiring Spider question pairs from xlangai/spider...")
        ds_sp_train = load_dataset("xlangai/spider", split="train")
        ds_sp_val = load_dataset("xlangai/spider", split="validation")
        sp_train_rows = [
            {"id": f"spider_train_{i}", "source": "spider_train", "db_id": r["db_id"], "question": r["question"], "evidence": "", "sql": r["query"]}
            for i, r in enumerate(ds_sp_train)
        ]
        sp_dev_rows = [
            {"id": f"spider_dev_{i}", "source": "spider_dev", "db_id": r["db_id"], "question": r["question"], "evidence": "", "sql": r["query"]}
            for i, r in enumerate(ds_sp_val)
        ]
        zip_path = cfg.spider_db_zip
        if zip_path is None and cfg.spider_db_url:
            zip_path = download_file_resumable(cfg.spider_db_url, spider_dir / "archives" / "spider.zip", cfg.timeout_sec)
        if zip_path is None or not Path(zip_path).exists():
            raise DatasetIntegrityError(
                "use_spider=True requires SQLite database files not bundled in HuggingFace xlangai/spider. "
                "Provide data.spider_db_zip (local path) or data.spider_db_url (direct zip archive URL)."
            )
        sp_extracted = extract_zip_recursive(Path(zip_path), spider_dir / "raw")
        sp_db_root = _locate_sqlite_root(sp_extracted, {r["db_id"] for r in sp_train_rows + sp_dev_rows})
        _verify_db_coverage(sp_train_rows + sp_dev_rows, sp_db_root, "Spider")
        res.spider_train_jsonl = _write_jsonl(sp_train_rows, spider_dir / "prepared" / "spider_train.jsonl")
        res.spider_dev_jsonl = _write_jsonl(sp_dev_rows, spider_dir / "prepared" / "spider_dev.jsonl")
        res.spider_db_root = sp_db_root
        logger.info("Spider ready: train=%d, dev=%d", len(sp_train_rows), len(sp_dev_rows))

    return res
