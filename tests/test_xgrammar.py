"""XGrammar-constrained decoding tests (skipped when the optional extra is missing)."""
from __future__ import annotations

import pytest
import torch

xgrammar = pytest.importorskip("xgrammar", reason="optional extra 'constrained' not installed")

from nl2sql.datasets.prepare import format_schema_for_parser
from nl2sql.models.graphix_parser import (
    GraphixCodeT5Parser,
    Seq2SeqXGrammarLogitsProcessor,
    build_sqlite_schema_ebnf,
)


def test_schema_ebnf_compiles_with_tokenizer(shop_bundle, tiny_t5_dir) -> None:
    parser = GraphixCodeT5Parser(model_id_or_path=str(tiny_t5_dir))
    assert parser._xgr_available
    ebnf = build_sqlite_schema_ebnf(shop_bundle.tables, [])
    compiled = parser._xgr_compiler.compile_grammar(ebnf)
    assert compiled is not None

    matcher = xgrammar.GrammarMatcher(compiled)
    bitmask = xgrammar.allocate_token_bitmask(1, parser.model.config.vocab_size)
    matcher.fill_next_token_bitmask(bitmask, 0)
    allowed = int((bitmask[0] != 0).sum())
    assert allowed >= 1  # at least one token starts a valid SQL query


def test_constrained_generation_never_crashes_or_nans(
    tiny_t5_dir, shop_db, shop_bundle
) -> None:
    from nl2sql.db.value_index import ValueIndex

    parser = GraphixCodeT5Parser(model_id_or_path=str(tiny_t5_dir))
    v_idx = ValueIndex(shop_db.parent / "vi_xgr.sqlite")
    v_idx.build_from_db(f"sqlite:///{shop_db.resolve()}", shop_bundle)
    val_hits = v_idx.probe("How many open tickets for Alice?", top_k=4)

    src = format_schema_for_parser(shop_bundle, shop_bundle.tables, val_hits, "How many open tickets for Alice?")
    cands = parser.generate_candidates(
        src, shop_bundle, shop_bundle.tables, val_hits,
        num_candidates=2, max_source_len=256, max_target_len=24, use_xgrammar=True,
    )
    # A tiny WordLevel vocab cannot walk the full grammar; the point is that constrained
    # sampling stays finite and terminates cleanly via the EOS fallback instead of NaN-crashing.
    assert isinstance(cands, list)
    assert all(isinstance(c, str) for c in cands)


def test_logits_processor_masks_disallowed_tokens(shop_bundle, tiny_t5_dir) -> None:
    parser = GraphixCodeT5Parser(model_id_or_path=str(tiny_t5_dir))
    ebnf = build_sqlite_schema_ebnf(shop_bundle.tables[:1], [])
    compiled = parser._xgr_compiler.compile_grammar(ebnf)
    proc = Seq2SeqXGrammarLogitsProcessor(
        compiled,
        vocab_size=parser.model.config.vocab_size,
        start_len=1,
        eos_token_id=parser.tokenizer.eos_token_id,
    )

    input_ids = torch.full((1, 1), parser.model.config.decoder_start_token_id, dtype=torch.long)
    scores = torch.zeros((1, parser.model.config.vocab_size))
    out = proc(input_ids, scores)
    assert not torch.isnan(out).any()  # EOS fallback prevents NaN from fully-masked rows
    assert torch.isfinite(out).any()  # at least one grammar-allowed token survives
    masked = int((out[0] == float("-inf")).sum())
    assert masked > 0  # grammar actually constrains most of the vocab
