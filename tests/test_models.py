"""Neural component tests using tiny offline models: Graphix parser, schema linker,
verifier, and the hybrid retriever."""
from __future__ import annotations

from pathlib import Path

import torch

from nl2sql.db.value_index import ValueIndex
from nl2sql.models import (
    ModernBertSchemaLinker,
    ModernBertVerifier,
    build_sqlite_schema_ebnf,
)
from nl2sql.models.graphix_parser import (
    NUM_REL_TYPES,
    REL_SAME_TABLE,
    GraphixCodeT5Parser,
)
from nl2sql.retrieval.schema_retriever import DenseTableEncoder, HybridSchemaRetriever
from nl2sql.db.introspect import TableInfo


def _build_value_index(shop_db: Path, shop_bundle) -> ValueIndex:
    idx = ValueIndex(shop_db.parent / "vi_models.sqlite")
    idx.build_from_db(f"sqlite:///{shop_db.resolve()}", shop_bundle)
    return idx


def test_graphix_parser_forward_and_relation_matrix(tiny_t5_dir: Path, shop_db: Path, shop_bundle) -> None:
    parser = GraphixCodeT5Parser(model_id_or_path=str(tiny_t5_dir))
    tables = shop_bundle.tables
    v_idx = _build_value_index(shop_db, shop_bundle)
    val_hits = v_idx.probe("How many open tickets for Alice?", top_k=4)

    from nl2sql.datasets.prepare import format_schema_for_parser

    src = format_schema_for_parser(shop_bundle, tables, val_hits, "How many open tickets for Alice?", "hint")
    enc = parser.tokenizer(src, return_tensors="pt", truncation=True, max_length=256)
    rel = GraphixCodeT5Parser.build_relation_matrix(enc["input_ids"], parser.tokenizer, [shop_bundle], [tables], [val_hits])
    bsz, seq_len = enc["input_ids"].shape
    assert tuple(rel.shape) == (bsz, seq_len, seq_len)
    assert rel.dtype == torch.long and int(rel.min()) >= 0 and int(rel.max()) < NUM_REL_TYPES
    # With the "Schema:" anchor detected, some same-table relations must be marked
    assert (rel == REL_SAME_TABLE).sum() > 0

    labels = parser.tokenizer("SELECT COUNT(*) FROM tickets WHERE status = 'open'", return_tensors="pt")["input_ids"]
    out = parser(enc["input_ids"], enc["attention_mask"], labels, rel)
    assert torch.isfinite(out.loss)


def test_graphix_parser_generate_candidates_and_roundtrip(tiny_t5_dir: Path, shop_db: Path, shop_bundle) -> None:
    parser = GraphixCodeT5Parser(model_id_or_path=str(tiny_t5_dir))
    v_idx = _build_value_index(shop_db, shop_bundle)
    val_hits = v_idx.probe("How many open tickets for Alice?", top_k=4)
    tables = shop_bundle.tables[:2]

    from nl2sql.datasets.prepare import format_schema_for_parser

    src = format_schema_for_parser(shop_bundle, tables, val_hits, "How many open tickets for Alice?")
    cands = parser.generate_candidates(
        src, shop_bundle, tables, val_hits, num_candidates=2,
        max_source_len=256, max_target_len=16, use_xgrammar=False,
    )
    assert isinstance(cands, list) and 1 <= len(cands) <= 2
    assert all(isinstance(c, str) and c for c in cands)

    # Save / load round-trip preserves the Graphix relation-bias parameters
    out_dir = shop_db.parent / "parser_artifact"
    before = parser.rel_bias_emb.weight.detach().clone()
    with torch.no_grad():
        parser.rel_bias_emb.weight.add_(1.0)
    parser.save_pretrained(out_dir)
    reloaded = GraphixCodeT5Parser.from_pretrained_dir(out_dir, map_location="cpu")
    assert torch.allclose(reloaded.rel_bias_emb.weight, parser.rel_bias_emb.weight)
    assert not torch.allclose(reloaded.rel_bias_emb.weight, before)


def test_build_sqlite_schema_ebnf(shop_bundle, shop_db: Path) -> None:
    v_idx = _build_value_index(shop_db, shop_bundle)
    val_hits = v_idx.probe("How many open tickets for Alice?", top_k=4)
    ebnf = build_sqlite_schema_ebnf(shop_bundle.tables, val_hits)
    for t in shop_bundle.tables:
        assert f'"{t.name}"' in ebnf
    assert "SELECT" in ebnf and "root ::=" in ebnf
    for v in val_hits:
        if "'" not in v.value:
            assert f"'{v.value}'" in ebnf


def test_schema_linker_forward_prune_and_roundtrip(tiny_modernbert_dir: Path, shop_db: Path, shop_bundle) -> None:
    linker = ModernBertSchemaLinker(model_id_or_path=str(tiny_modernbert_dir), pos_weight=2.0)
    v_idx = _build_value_index(shop_db, shop_bundle)
    val_hits = v_idx.probe("How many open tickets for Alice?", top_k=4)

    text, tbl_names, col_names = linker.format_inputs(
        "How many open tickets for Alice?", shop_bundle.tables, val_hits, "status open"
    )
    assert "[TBL] customers" in text and "[COL]" in text
    assert tbl_names == [t.name for t in shop_bundle.tables]
    assert len(col_names) == sum(len(t.columns) for t in shop_bundle.tables)

    enc = linker.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    table_labels = [torch.tensor([1.0, 0.0, 0.0])]
    col_labels = [torch.tensor([0.0] * len(col_names))]
    col_labels[0][col_names.index("tickets.status")] = 1.0
    out = linker(enc["input_ids"], enc["attention_mask"], table_labels, col_labels)
    assert torch.isfinite(out["loss"])
    assert len(out["table_logits"][0]) == len(tbl_names)
    assert len(out["column_logits"][0]) == len(col_names)

    # Prune keeps between min_tables and max_tables, respecting the candidate pool
    pruned = linker.prune_schema(
        "How many open tickets for Alice?", shop_bundle, shop_bundle.tables, val_hits,
        threshold=0.5, min_tables=2, max_tables=2, max_length=256,
    )
    assert 1 <= len(pruned) <= 2
    assert {t.name for t in pruned} <= {t.name for t in shop_bundle.tables}

    out_dir = shop_db.parent / "linker_artifact"
    with torch.no_grad():
        linker.table_head[-1].weight.add_(0.5)
    linker.save_pretrained(out_dir)
    reloaded = ModernBertSchemaLinker.from_pretrained_dir(out_dir, map_location="cpu")
    assert torch.allclose(reloaded.table_head[-1].weight, linker.table_head[-1].weight)


def test_verifier_scoring_and_roundtrip(tiny_modernbert_dir: Path, shop_db: Path, shop_bundle) -> None:
    verifier = ModernBertVerifier(model_id_or_path=str(tiny_modernbert_dir))
    text = verifier.format_pair("q?", "schema text", "SELECT 1", "ok rows=1")
    assert "Question: q?" in text and "Candidate SQL: SELECT 1" in text

    enc = verifier.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    out = verifier(enc["input_ids"], enc["attention_mask"], torch.tensor([1.0]))
    assert torch.isfinite(out["loss"])

    scores = verifier.score_candidates(
        "q?", "schema text", [("SELECT 1", "ok rows=1"), ("SELECT bad", "err=boom")], max_length=256
    )
    assert len(scores) == 2 and all(0.0 <= s <= 1.0 for s in scores)

    out_dir = shop_db.parent / "verifier_artifact"
    with torch.no_grad():
        verifier.classifier[-1].weight.add_(0.25)
    verifier.save_pretrained(out_dir)
    reloaded = ModernBertVerifier.from_pretrained_dir(out_dir, map_location="cpu")
    assert torch.allclose(reloaded.classifier[-1].weight, verifier.classifier[-1].weight)
    # Temperature metadata survives the round-trip
    reloaded.temperature.fill_(0.7)
    reloaded.save_pretrained(out_dir)
    again = ModernBertVerifier.from_pretrained_dir(out_dir, map_location="cpu")
    assert abs(float(again.temperature.item()) - 0.7) < 1e-6


def test_dense_encoder_and_hybrid_retriever(tiny_modernbert_dir: Path, shop_db: Path, shop_bundle) -> None:
    enc = DenseTableEncoder(str(tiny_modernbert_dir), device="cpu")
    vecs = enc.encode(["table customers columns id name", "table orders columns id amount"])
    assert vecs.shape == (2, 32)  # tiny hidden size
    assert abs(float((vecs**2).sum(axis=1).mean()) - 1.0) < 1e-3  # L2-normalized

    retriever = HybridSchemaRetriever(enc)
    v_idx = _build_value_index(shop_db, shop_bundle)
    res = retriever.retrieve("How many open tickets for Alice?", shop_bundle, v_idx, max_candidate_tables=24)
    assert len(res.candidate_tables) == len(shop_bundle.tables)  # small DB: all tables
    assert any(v.column_name == "status" and v.value == "open" for v in res.value_matches)

    # Large-schema path: >24 tables triggers BM25 + dense hybrid ranking and FK bridge expansion
    big_bundle = shop_bundle.model_copy(deep=True)
    filler_tables = [
        TableInfo(name=f"audit_{i}", row_count=0, columns=[])
        for i in range(30)
    ]
    big_bundle.tables = big_bundle.tables + filler_tables
    res_big = retriever.retrieve("How many open tickets for Alice?", big_bundle, v_idx, max_candidate_tables=8)
    assert 0 < len(res_big.candidate_tables) <= 8
    assert {t.name for t in res_big.candidate_tables} <= {t.name for t in big_bundle.tables}
