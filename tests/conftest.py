"""Shared fixtures: synthetic SQLite databases, tiny offline models, and configs.

The tiny models (a ~215K-param T5 and a ~22K-param ModernBERT) let the full
training/inference stack run end-to-end in CI without downloading any weights.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    ModernBertConfig,
    ModernBertModel,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)

from nl2sql.config import DataConfig, EngineConfig, HubConfig, ModelConfig, TrainingConfig
from nl2sql.db.introspect import SchemaBundle, introspect_database

SHOP_DDL = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, state TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
CREATE TABLE tickets (id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT);
INSERT INTO customers VALUES (1, 'Alice', 'CA'), (2, 'Bob', 'NY');
INSERT INTO orders VALUES (10, 1, 120.5), (11, 1, 80.0), (12, 2, 50.0);
INSERT INTO tickets VALUES (100, 1, 'open'), (101, 2, 'closed');
"""

WAREHOUSE_DDL = """
CREATE TABLE regions (id INTEGER PRIMARY KEY, region_name TEXT);
CREATE TABLE stores (id INTEGER PRIMARY KEY, region_id INTEGER, store_name TEXT);
CREATE TABLE sales (id INTEGER PRIMARY KEY, store_id INTEGER, revenue REAL, sale_date TEXT);
INSERT INTO regions VALUES (1, 'West'), (2, 'East');
INSERT INTO stores VALUES (10, 1, 'Alpha'), (11, 2, 'Beta');
INSERT INTO sales VALUES (100, 10, 500.0, '2026-01-05'), (101, 10, 250.0, '2026-02-10'), (102, 11, 90.0, '2026-03-01');
"""


def create_sqlite(db_path: Path, ddl: str) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(ddl)
    conn.close()
    return db_path


def build_wordlevel_tokenizer(words: list[str], extra_specials: list[str]) -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<unk>": 1, "</s>": 2}
    for w in words:
        vocab.setdefault(w, len(vocab))
    for s in extra_specials:
        vocab.setdefault(s, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        additional_special_tokens=[s for s in extra_specials if s not in ("<pad>", "<unk>", "</s>")],
    )


VOCAB_WORDS = [
    "Question", "Schema", "customers", "orders", "tickets", "regions", "stores", "sales",
    "id", "name", "state", "customer_id", "amount", "status", "region_id", "store_id",
    "revenue", "sale_date", "region_name", "store_name", "INTEGER", "TEXT", "REAL",
    "SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "BY", "ORDER", "LIMIT", "AS",
    "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT", "AND", "OR", "How", "many", "open",
    "for", "Alice", "Bob", "total", "revenue", "the", "of", "in", "is", "what", "all",
    "Matched", "Values", "Hint", "ex", "PK", "a", "b", "c", "the", "of", "in", "and",
]


@pytest.fixture(scope="session")
def tiny_t5_dir(tmp_path_factory: Path) -> Path:
    out = tmp_path_factory.mktemp("tiny_t5")
    tok = build_wordlevel_tokenizer(VOCAB_WORDS, ["[TBL]", "[COL]", "[PK]", "[FK]", "[VAL]"])
    cfg = T5Config(
        vocab_size=len(tok),
        d_model=32,
        d_ff=64,
        num_heads=4,
        num_layers=2,
        num_decoder_layers=2,
        pad_token_id=0,
        decoder_start_token_id=0,
        eos_token_id=2,
    )
    model = T5ForConditionalGeneration(cfg)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def tiny_modernbert_dir(tmp_path_factory: Path) -> Path:
    out = tmp_path_factory.mktemp("tiny_modernbert")
    tok = build_wordlevel_tokenizer(VOCAB_WORDS, ["[TBL]", "[COL]"])
    cfg = ModernBertConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        cls_token_id=None,
        sep_token_id=None,
    )
    model = ModernBertModel(cfg)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


@pytest.fixture()
def shop_db(tmp_path: Path) -> Path:
    return create_sqlite(tmp_path / "shop.sqlite", SHOP_DDL)


@pytest.fixture()
def shop_url(shop_db: Path) -> str:
    return f"sqlite:///{shop_db.resolve()}"


@pytest.fixture()
def shop_bundle(shop_url: str) -> SchemaBundle:
    return introspect_database(shop_url, db_id="shop", infer_missing_fks=True)


def make_engine_config(
    tmp_dir: Path,
    tiny_t5: Path,
    tiny_modernbert: Path,
    epochs: int = 1,
    stage: str = "parser",
    val_ratio_by_db: float = 0.5,
) -> EngineConfig:
    return EngineConfig(
        project_name="nl2sql-test",
        work_dir=tmp_dir / "workspace",
        seed=7,
        data=DataConfig(
            data_dir=tmp_dir / "data",
            use_bird=True,
            use_bird_filtered_train=True,
            use_spider=False,
            val_ratio_by_db=val_ratio_by_db,
            schema_perturbation_prob=0.0,
            max_distinct_values_per_col=1000,
            timeout_sec=30,
        ),
        models=ModelConfig(
            embedder_model_id=str(tiny_modernbert),
            linker_model_id=str(tiny_modernbert),
            parser_model_id=str(tiny_t5),
            verifier_model_id=str(tiny_modernbert),
            linker_max_length=256,
            parser_max_source_length=256,
            parser_max_target_length=32,
            verifier_max_length=256,
            num_return_candidates=2,
            use_xgrammar=False,
        ),
        training=TrainingConfig(
            stage=stage,
            output_dir=tmp_dir / "run",
            epochs=epochs,
            batch_size=2,
            grad_accum_steps=1,
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            mixed_precision="no",
            eval_every_epochs=1,
            save_every_steps=100,
            keep_last_k_checkpoints=2,
            early_stopping_patience=3,
            resume_from="auto",
        ),
        hub=HubConfig(enabled=False),
    )


@pytest.fixture()
def engine_config(tmp_path: Path, tiny_t5_dir: Path, tiny_modernbert_dir: Path) -> EngineConfig:
    return make_engine_config(tmp_path, tiny_t5_dir, tiny_modernbert_dir)
