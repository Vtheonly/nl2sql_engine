from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Optional

import sqlglot
from sqlglot import exp
from pydantic import BaseModel, Field
from torch.utils.data import Dataset

from nl2sql.db.introspect import SchemaBundle, TableInfo, compute_ddl_hash, introspect_database
from nl2sql.db.value_index import ValueIndex, ValueMatch
from nl2sql.datasets.acquire import DatasetAcquisitionResult
from nl2sql.utils.logging import get_logger

logger = get_logger(__name__)


class ExampleRecord(BaseModel):
    id: str
    source: str
    db_id: str
    db_path: Path
    question: str
    evidence: str = ""
    sql: str
    difficulty: str = "unknown"
    gold_tables: list[str] = Field(default_factory=list)
    gold_columns: list[str] = Field(default_factory=list)  # table.column
    # ValueIndex hits for the question, precomputed once during preparation (records
    # used to re-probe FTS5 on every __getitem__ across every epoch).
    value_matches: list[ValueMatch] = Field(default_factory=list)


def extract_gold_schema_targets(sql: str, bundle: SchemaBundle) -> tuple[list[str], list[str]]:
    """Parse SQL via SQLGlot to identify ground-truth tables and table.column pairs.

    Unqualified column names are resolved only when they exist in exactly one gold
    table; genuinely ambiguous references (``SELECT id FROM a JOIN b``) and output
    aliases shadowing a column name are skipped instead of poisoning labels across
    every joined table.
    """
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return [], []

    alias_to_table: dict[str, str] = {}
    gold_tables_set: set[str] = set()
    for tbl in parsed.find_all(exp.Table):
        t_obj = bundle.get_table(tbl.name)
        if t_obj:
            gold_tables_set.add(t_obj.name)
            if tbl.alias:
                alias_to_table[tbl.alias.lower()] = t_obj.name
            alias_to_table[t_obj.name.lower()] = t_obj.name

    select_aliases = {
        e.alias.lower()
        for e in parsed.expressions
        if isinstance(e, exp.Alias) and e.alias
    } if isinstance(parsed, exp.Select) else set()

    gold_cols_set: set[str] = set()
    for col in parsed.find_all(exp.Column):
        cname = col.name
        if cname == "*":
            continue
        if col.table and col.table.lower() in alias_to_table:
            tname = alias_to_table[col.table.lower()]
            t_obj = bundle.get_table(tname)
            if t_obj and t_obj.get_column(cname):
                gold_cols_set.add(f"{t_obj.name}.{t_obj.get_column(cname).name}")
        elif not col.table:
            # Output aliases (ORDER BY total) are not schema columns.
            if cname.lower() in select_aliases:
                continue
            # Unqualified: only resolve when exactly one gold table has this column.
            matches = [
                t_obj for tname in gold_tables_set
                if (t_obj := bundle.get_table(tname)) is not None and t_obj.get_column(cname) is not None
            ]
            if len(matches) == 1:
                gold_cols_set.add(f"{matches[0].name}.{matches[0].get_column(cname).name}")
    return sorted(gold_tables_set), sorted(gold_cols_set)


def perturb_schema_bundle(bundle: SchemaBundle, sql: str, rng: random.Random) -> tuple[SchemaBundle, str]:
    """Shuffle table and column order (structure-preserving permutation that blocks
    position memorization). Identifier names and the SQL target are intentionally kept
    identical so gold SQL stays aligned with the schema."""
    new_bundle = copy.deepcopy(bundle)
    rng.shuffle(new_bundle.tables)
    for tbl in new_bundle.tables:
        rng.shuffle(tbl.columns)
    return new_bundle, sql


def build_db_caches(
    db_paths: dict[str, Path],
    cache_dir: Path,
    column_meanings: Optional[dict[str, str]] = None,
    max_distinct_per_col: int = 10000,
) -> dict[str, tuple[SchemaBundle, ValueIndex]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    caches: dict[str, tuple[SchemaBundle, ValueIndex]] = {}
    for db_id, sqlite_path in sorted(db_paths.items()):
        # A missing SQLite file must fail loudly: SQLAlchemy would otherwise silently
        # create an EMPTY database and yield a zero-table schema bundle downstream.
        if not sqlite_path.exists():
            raise FileNotFoundError(
                f"SQLite database for db_id '{db_id}' not found at '{sqlite_path}'. "
                "Re-run dataset acquisition or fix data paths."
            )
        db_cache = cache_dir / db_id
        bundle_path = db_cache / "schema_bundle.json"
        val_path = db_cache / "value_index.sqlite"
        conn_url = f"sqlite:///{sqlite_path.resolve()}"
        bundle: Optional[SchemaBundle] = None
        if bundle_path.exists():
            cached = SchemaBundle.load(bundle_path)
            # Never serve a stale bundle: verify the DDL hash against the live DB and
            # re-introspect when tables/columns changed since the cache was written.
            live_hash = compute_ddl_hash(conn_url)
            if live_hash == cached.ddl_hash:
                bundle = cached
            else:
                logger.warning("Cached schema for db_id '%s' is stale (DDL changed); re-introspecting.", db_id)
        if bundle is None:
            bundle = introspect_database(conn_url, db_id=db_id, column_meanings=column_meanings)
            bundle.save(bundle_path)
        v_index = ValueIndex(val_path)
        if not v_index.is_built_for(bundle.ddl_hash):
            v_index.build_from_db(conn_url, bundle, max_distinct_per_col=max_distinct_per_col)
        caches[db_id] = (bundle, v_index)
    return caches


def prepare_records(
    acq: DatasetAcquisitionResult,
    cache_dir: Path,
    val_ratio_by_db: float = 0.15,
    seed: int = 42,
    max_distinct_per_col: int = 10000,
) -> tuple[list[ExampleRecord], list[ExampleRecord], list[ExampleRecord], dict[str, tuple[SchemaBundle, ValueIndex]]]:
    """Load datasets, build persistent per-DB SchemaBundles + ValueIndexes, split train/val by db_id."""
    col_meanings: dict[str, str] = {}
    if acq.bird_column_meanings_path and acq.bird_column_meanings_path.exists():
        col_meanings = json.loads(acq.bird_column_meanings_path.read_text(encoding="utf-8"))

    raw_train_items: list[tuple[dict, Path]] = []
    raw_test_items: list[tuple[dict, Path]] = []

    def _read_jsonl(p: Path, db_root: Path, bucket: list[tuple[dict, Path]]) -> None:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            db_path = db_root / row["db_id"] / f"{row['db_id']}.sqlite"
            bucket.append((row, db_path))

    if acq.bird_train_jsonl and acq.bird_train_db_root:
        _read_jsonl(acq.bird_train_jsonl, acq.bird_train_db_root, raw_train_items)
    if acq.spider_train_jsonl and acq.spider_db_root:
        _read_jsonl(acq.spider_train_jsonl, acq.spider_db_root, raw_train_items)
    if acq.bird_dev_jsonl and acq.bird_dev_db_root:
        _read_jsonl(acq.bird_dev_jsonl, acq.bird_dev_db_root, raw_test_items)
    if acq.spider_dev_jsonl and acq.spider_db_root:
        _read_jsonl(acq.spider_dev_jsonl, acq.spider_db_root, raw_test_items)

    all_db_paths: dict[str, Path] = {}
    for row, db_p in raw_train_items + raw_test_items:
        db_id = row["db_id"]
        prior = all_db_paths.get(db_id)
        if prior is not None and prior.resolve() != db_p.resolve():
            # BIRD and Spider can legitimately share db_id strings; overwriting one
            # with the other would silently train and evaluate against the wrong DB.
            raise ValueError(
                f"db_id '{db_id}' is provided by two different databases: '{prior}' and '{db_p}'. "
                "Disable one dataset or namespace its db_ids before combining BIRD + Spider."
            )
        all_db_paths[db_id] = db_p
    caches = build_db_caches(
        all_db_paths,
        cache_dir=cache_dir,
        column_meanings=col_meanings,
        max_distinct_per_col=max_distinct_per_col,
    )

    train_db_ids = sorted({row["db_id"] for row, _ in raw_train_items})
    rng = random.Random(seed)
    rng.shuffle(train_db_ids)
    n_val_dbs = max(1, int(round(len(train_db_ids) * val_ratio_by_db))) if len(train_db_ids) > 1 else 0
    val_db_set = set(train_db_ids[:n_val_dbs])

    def _to_records(items: list[tuple[dict, Path]]) -> list[ExampleRecord]:
        out: list[ExampleRecord] = []
        for row, db_p in items:
            bundle, v_idx = caches[row["db_id"]]
            g_tbls, g_cols = extract_gold_schema_targets(row["sql"], bundle)
            # Precompute FTS5 value hits once: they only depend on (question, db),
            # never on the epoch or the training stage.
            val_hits = v_idx.probe(row["question"], top_k=6)
            out.append(
                ExampleRecord(
                    id=row["id"],
                    source=row["source"],
                    db_id=row["db_id"],
                    db_path=db_p,
                    question=row["question"],
                    evidence=row.get("evidence", ""),
                    sql=row["sql"],
                    difficulty=row.get("difficulty", "unknown"),
                    gold_tables=g_tbls,
                    gold_columns=g_cols,
                    value_matches=val_hits,
                )
            )
        return out

    def _drop_unlinkable(records: list[ExampleRecord], split: str) -> list[ExampleRecord]:
        """Records whose SQL yields no gold tables would train the parser to produce
        SQL from random distractor tables alone; drop them (and say how many)."""
        kept = [r for r in records if r.gold_tables]
        dropped = len(records) - len(kept)
        if dropped:
            logger.warning(
                "Dropped %d/%d '%s' records with no resolvable gold tables (unparseable SQL or unknown tables).",
                dropped, len(records), split,
            )
        return kept

    full_train = _to_records(raw_train_items)
    train_split = _drop_unlinkable([r for r in full_train if r.db_id not in val_db_set], "train")
    val_split = _drop_unlinkable([r for r in full_train if r.db_id in val_db_set], "val")
    test_split = _to_records(raw_test_items)
    return train_split, val_split, test_split, caches


def load_minidev_records(
    acq: DatasetAcquisitionResult,
    caches: dict[str, tuple[SchemaBundle, ValueIndex]],
) -> list[ExampleRecord]:
    """Load the BIRD mini-dev benchmark rows as evaluation records.

    ``acquire_all_datasets`` downloads and verifies this split, but it was previously
    never exposed to the pipeline. Mini-dev shares the dev database set, so the
    already-built caches are reused.
    """
    if not acq.bird_minidev_jsonl or not acq.bird_minidev_jsonl.exists():
        return []
    if not acq.bird_dev_db_root:
        return []
    out: list[ExampleRecord] = []
    for line in acq.bird_minidev_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["db_id"] not in caches:
            continue
        bundle, v_idx = caches[row["db_id"]]
        g_tbls, g_cols = extract_gold_schema_targets(row["sql"], bundle)
        out.append(
            ExampleRecord(
                id=row["id"],
                source=row.get("source", "bird_minidev"),
                db_id=row["db_id"],
                db_path=acq.bird_dev_db_root / row["db_id"] / f"{row['db_id']}.sqlite",
                question=row["question"],
                evidence=row.get("evidence", ""),
                sql=row["sql"],
                difficulty=row.get("difficulty", "unknown"),
                gold_tables=g_tbls,
                gold_columns=g_cols,
                value_matches=v_idx.probe(row["question"], top_k=6),
            )
        )
    return out


def format_schema_for_parser(
    bundle: SchemaBundle,
    tables: list[TableInfo],
    value_matches: list[ValueMatch],
    question: str,
    evidence: str = "",
    example_hints: Optional[list[dict[str, str]]] = None,
) -> str:
    tbl_names = {t.name for t in tables}
    parts = [f"Question: {question}"]
    if evidence.strip():
        parts.append(f"Hint: {evidence.strip()}")
    if example_hints:
        # Registered example queries ground the parser on unseen enterprise databases
        # (the StructCBR-style adaptation signal at inference time).
        lines = [f"Q: {h.get('question', '')} SQL: {h.get('sql', h.get('query', ''))}" for h in example_hints[:3]]
        parts.append("Example Queries:\n" + "\n".join(lines))
    if value_matches:
        vals = "; ".join(f"{v.table_name}.{v.column_name} = '{v.value}'" for v in value_matches[:6])
        parts.append(f"Matched Values: {vals}")
    schema_lines = []
    for t in tables:
        cols_desc = []
        for c in t.columns:
            tag = "[PK]" if c.is_pk else ""
            samp = f" ex=({', '.join(c.sample_values[:2])})" if c.sample_values else ""
            desc = f" -- {c.business_description[:60]}" if c.business_description else ""
            cols_desc.append(f"[COL] {c.name} {c.data_type}{tag}{samp}{desc}")
        schema_lines.append(f"[TBL] {t.name}: " + " | ".join(cols_desc))
    fk_lines = [
        f"{e.left_table}.{e.left_column} -> {e.right_table}.{e.right_column} ({e.cardinality})"
        for e in bundle.join_edges
        if e.left_table in tbl_names and e.right_table in tbl_names
    ]
    parts.append("[SCHEMA] Schema:\n" + "\n".join(schema_lines))
    if fk_lines:
        parts.append("Foreign Keys: " + "; ".join(fk_lines))
    return "\n".join(parts)


class EmbedderDataset(Dataset):
    """Positive (question, table.column description) pairs for MNRL bi-encoder fine-tuning."""

    def __init__(self, records: list[ExampleRecord], caches: dict[str, tuple[SchemaBundle, ValueIndex]]) -> None:
        self.pairs: list[tuple[str, str]] = []
        for rec in records:
            bundle, _ = caches[rec.db_id]
            for col_ref in rec.gold_columns:
                if "." not in col_ref:
                    continue
                tname, cname = col_ref.split(".", 1)
                t_obj = bundle.get_table(tname)
                c_obj = t_obj.get_column(cname) if t_obj else None
                if t_obj and c_obj:
                    card = (
                        f"table {t_obj.name} column {c_obj.name} type {c_obj.data_type} "
                        f"samples {' '.join(c_obj.sample_values[:3])} "
                        f"{c_obj.business_description or c_obj.comment or ''}"
                    ).strip()
                    self.pairs.append((rec.question, card))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, str]:
        q, c = self.pairs[idx]
        return {"query": q, "positive": c}


class LinkerDataset(Dataset):
    """Dataset for ModernBERT table+column multi-head schema linking."""

    def __init__(
        self,
        records: list[ExampleRecord],
        caches: dict[str, tuple[SchemaBundle, ValueIndex]],
        perturb_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.caches = caches
        self.perturb_prob = perturb_prob
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        bundle, v_index = self.caches[rec.db_id]
        if self.perturb_prob > 0 and self.rng.random() < self.perturb_prob:
            bundle, _ = perturb_schema_bundle(bundle, rec.sql, self.rng)
        # Hits precomputed once in prepare_records(); probe only as a legacy fallback
        # for records constructed ad hoc (tests, external callers).
        val_hits = rec.value_matches or v_index.probe(rec.question, top_k=5)
        gold_tbl_set = {t.lower() for t in rec.gold_tables}
        gold_col_set = {c.lower() for c in rec.gold_columns}

        table_items = []
        col_items = []
        for t in bundle.tables:
            table_items.append({"name": t.name, "label": 1.0 if t.name.lower() in gold_tbl_set else 0.0})
            for c in t.columns:
                full = f"{t.name}.{c.name}".lower()
                samp = ", ".join(c.sample_values[:2])
                text_repr = f"{t.name}.{c.name} ({c.data_type})" + (f" [{samp}]" if samp else "")
                col_items.append({"name": f"{t.name}.{c.name}", "repr": text_repr, "label": 1.0 if full in gold_col_set else 0.0})

        return {
            "question": rec.question,
            "evidence": rec.evidence,
            "value_matches": val_hits,
            "bundle": bundle,
            "table_items": table_items,
            "col_items": col_items,
        }


class ParserDataset(Dataset):
    """Dataset for Graphix-CodeT5+-220M outputting normalized canonical SQL."""

    def __init__(
        self,
        records: list[ExampleRecord],
        caches: dict[str, tuple[SchemaBundle, ValueIndex]],
        perturb_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.caches = caches
        self.perturb_prob = perturb_prob
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        bundle, v_index = self.caches[rec.db_id]
        if self.perturb_prob > 0 and self.rng.random() < self.perturb_prob:
            bundle, _ = perturb_schema_bundle(bundle, rec.sql, self.rng)
        # Hits precomputed once in prepare_records(); probe only as a legacy fallback.
        val_hits = rec.value_matches or v_index.probe(rec.question, top_k=6)

        # Records without gold tables were filtered out during preparation; guard
        # anyway so a distractor-only prompt can never be trained on.

        # During training, retain all gold tables + up to 4 distractor tables so model handles noise
        gold_set = {t.lower() for t in rec.gold_tables}
        gold_tables = [t for t in bundle.tables if t.name.lower() in gold_set]
        distractors = [t for t in bundle.tables if t.name.lower() not in gold_set]
        self.rng.shuffle(distractors)
        selected_tables = gold_tables + distractors[: max(0, min(4, 8 - len(gold_tables)))]
        self.rng.shuffle(selected_tables)

        source_text = format_schema_for_parser(
            bundle=bundle,
            tables=selected_tables,
            value_matches=val_hits,
            question=rec.question,
            evidence=rec.evidence,
        )
        try:
            canonical_sql = sqlglot.transpile(rec.sql, read="sqlite", write="sqlite")[0]
        except Exception:
            canonical_sql = rec.sql.strip()

        return {
            "id": rec.id,
            "db_id": rec.db_id,
            "source_text": source_text,
            "target_sql": canonical_sql,
            "bundle": bundle,
            "selected_tables": selected_tables,
            "value_matches": val_hits,
        }


class VerifierDataset(Dataset):
    """Execution-divergence dataset for ModernBERT-base selective correctness scoring."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        # Each sample: {"question", "schema_text", "candidate_sql", "exec_status", "label": 0.0|1.0}
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]
