"""Heavy model-layer tests: parser thread-safety, determinism, EBNF robustness,
relation-matrix invariants, linker/verifier edge cases, and retriever regressions."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

from nl2sql.db.introspect import ColumnInfo, JoinEdge, SchemaBundle, TableInfo
from nl2sql.db.value_index import ValueMatch
from nl2sql.models.graphix_parser import GraphixCodeT5Parser, build_sqlite_schema_ebnf
from nl2sql.models.schema_linker import ModernBertSchemaLinker
from nl2sql.models.verifier import ModernBertVerifier
from nl2sql.retrieval.schema_retriever import DenseTableEncoder, HybridSchemaRetriever


def _encode_with_rel(parser: GraphixCodeT5Parser, text: str, rel: torch.Tensor):
    """Run the parser's bias-critical encoder path (same pattern as forward/generate)."""
    enc = parser.tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    # T5 requires labels (or generation mode) for a direct model forward call.
    labels = torch.full((1, 4), parser.model.config.eos_token_id or 2, dtype=torch.long)
    with torch.no_grad():
        out = parser.forward(enc["input_ids"], enc["attention_mask"], labels=labels, rel_matrix=rel)
    return out.encoder_last_hidden_state.clone()


class TestParserThreadSafety:
    def test_concurrent_bias_sections_do_not_cross_contaminate(self, tiny_t5_dir: Path, shop_bundle) -> None:
        """`_active_rel_bias` is shared instance state; concurrent forwards used to
        overwrite/clear each other's bias tensor mid-encoder-pass. With non-zero
        rel-bias weights and same-length inputs (so the shape guard cannot mask the
        race), concurrent encoder outputs must equal the sequential baselines."""
        parser = GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).eval()
        # Randomize relation-type embeddings so a misapplied bias visibly changes hidden states.
        with torch.no_grad():
            parser.rel_bias_emb.weight.copy_(torch.randn_like(parser.rel_bias_emb.weight) * 3.0)

        # Same token length, different relation structure.
        text_a = "Question: how many open tickets\nSchema:\n[TBL] customers: [COL] id INTEGER [COL] name TEXT [COL] state TEXT"
        text_b = "Question: how many closed tickets\nSchema:\n[TBL] orders: [COL] id INTEGER [COL] amount REAL [COL] status TEXT"
        enc_a = parser.tokenizer(text_a)["input_ids"]
        enc_b = parser.tokenizer(text_b)["input_ids"]
        assert len(enc_a) == len(enc_b), "test requires equal sequence lengths"

        L = len(enc_a)
        rel_a = torch.zeros((1, L, L), dtype=torch.long)  # REL_NONE
        rel_b = torch.full((1, L, L), 1, dtype=torch.long)  # REL_SAME_TABLE
        assert not torch.equal(parser._compute_bias_tensor(rel_a), parser._compute_bias_tensor(rel_b))

        base_a = _encode_with_rel(parser, text_a, rel_a)
        base_b = _encode_with_rel(parser, text_b, rel_b)
        assert not torch.allclose(base_a, base_b), "rel bias must influence the encoder output"

        mismatches: list[str] = []
        errors: list[Exception] = []
        n_rounds = 8
        for _ in range(n_rounds):
            results: dict[str, torch.Tensor] = {}
            barrier = threading.Barrier(4)

            def worker(key: str, text: str, rel: torch.Tensor) -> None:
                try:
                    barrier.wait(timeout=30)
                    results[key] = _encode_with_rel(parser, text, rel)
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [
                threading.Thread(target=worker, args=("a", text_a, rel_a)),
                threading.Thread(target=worker, args=("b", text_b, rel_b)),
                threading.Thread(target=worker, args=("a2", text_a, rel_a)),
                threading.Thread(target=worker, args=("b2", text_b, rel_b)),
            ]
            [t.start() for t in threads]
            [t.join(timeout=60) for t in threads]

            assert not errors, f"concurrent forwards raised: {errors}"
            for key, base in (("a", base_a), ("a2", base_a), ("b", base_b), ("b2", base_b)):
                if not torch.allclose(results[key], base):
                    mismatches.append(key)

        assert not mismatches, (
            f"attention-bias cross-contamination detected in {len(mismatches)}/{4 * n_rounds} "
            f"concurrent forwards (keys: {sorted(set(mismatches))})"
        )

    def test_bias_lock_is_released_after_use(self, tiny_t5_dir: Path, shop_bundle) -> None:
        parser = GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).eval()
        assert parser._bias_lock.acquire(blocking=False), "lock leaked after construction"
        parser._bias_lock.release()
        text = "Question: q\nSchema:\n[TBL] customers: [COL] id INTEGER"
        enc = parser.tokenizer(text, return_tensors="pt")
        rel = torch.zeros((1, enc["input_ids"].shape[1], enc["input_ids"].shape[1]), dtype=torch.long)
        _encode_with_rel(parser, text, rel)
        assert parser._bias_lock.acquire(blocking=False), "lock leaked after forward"
        parser._bias_lock.release()
        assert parser._active_rel_bias is None


class TestParserGeneration:
    def test_greedy_generation_is_deterministic(self, tiny_t5_dir: Path, shop_bundle) -> None:
        parser = GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).eval()
        text = "Question: how many open tickets\nSchema:\n[TBL] tickets: [COL] status TEXT"
        kwargs = dict(max_source_len=128, max_target_len=16, use_xgrammar=False)
        out1 = parser.generate_candidates(text, shop_bundle, shop_bundle.tables, [], num_candidates=1, **kwargs)
        out2 = parser.generate_candidates(text, shop_bundle, shop_bundle.tables, [], num_candidates=1, **kwargs)
        assert out1 == out2

    def test_candidate_count_and_dedup(self, tiny_t5_dir: Path, shop_bundle) -> None:
        parser = GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).eval()
        text = "Question: how many open tickets\nSchema:\n[TBL] tickets: [COL] status TEXT"
        outs = parser.generate_candidates(
            text, shop_bundle, shop_bundle.tables, [], num_candidates=4,
            max_source_len=128, max_target_len=16, use_xgrammar=False,
        )
        assert 0 <= len(outs) <= 4
        assert len(outs) == len(set(outs))  # deduplicated
        assert all(isinstance(s, str) and s == s.strip() for s in outs)

    def test_generation_with_empty_table_list_does_not_crash(self, tiny_t5_dir: Path, shop_bundle) -> None:
        parser = GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).eval()
        text = "Question: anything\nSchema:\n"
        outs = parser.generate_candidates(
            text, shop_bundle, [], [], num_candidates=2,
            max_source_len=128, max_target_len=16, use_xgrammar=False,
        )
        assert isinstance(outs, list)


class TestEBNF:
    def test_empty_schema_falls_back_to_dual(self) -> None:
        ebnf = build_sqlite_schema_ebnf([], [])
        assert '"dual"' in ebnf
        assert "root ::=" in ebnf

    def test_tables_and_columns_present(self, shop_bundle) -> None:
        ebnf = build_sqlite_schema_ebnf(shop_bundle.tables, [])
        for t in shop_bundle.tables:
            assert f'"{t.name}"' in ebnf
        for c in shop_bundle.tables[0].columns:
            assert f'"{c.name}"' in ebnf

    def test_values_with_single_quotes_are_excluded(self, shop_bundle) -> None:
        vm = [
            ValueMatch(table_name="customers", column_name="name", value="O'Brien", matched_phrase="obrien", score=1.0),
            ValueMatch(table_name="customers", column_name="state", value="CA", matched_phrase="ca", score=1.0),
        ]
        ebnf = build_sqlite_schema_ebnf(shop_bundle.tables, vm)
        assert "O'Brien" not in ebnf
        assert "'CA'" in ebnf

    def test_value_with_double_quote_is_escaped(self, shop_bundle) -> None:
        vm = [ValueMatch(table_name="t", column_name="c", value='He said "hi"', matched_phrase="x", score=1.0)]
        ebnf = build_sqlite_schema_ebnf(shop_bundle.tables, vm)
        assert 'He said \\"hi\\"' in ebnf


class TestRelationMatrix:
    def test_matrix_values_are_valid_relation_types(self, tiny_t5_dir: Path, shop_bundle) -> None:
        parser = GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).eval()
        text = "Question: how many tickets\nSchema:\n[TBL] tickets: [COL] id INTEGER [COL] status TEXT"
        enc = parser.tokenizer(text, return_tensors="pt")
        rel = parser.build_relation_matrix(enc["input_ids"], parser.tokenizer, [shop_bundle], [shop_bundle.tables], [[]])
        assert rel.shape == enc["input_ids"].shape + enc["input_ids"].shape[1:]
        assert int(rel.min()) >= 0 and int(rel.max()) < 5

    def test_matrix_with_empty_tables_is_zero(self, tiny_t5_dir: Path, shop_bundle) -> None:
        parser = GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir).eval()
        enc = parser.tokenizer("Question: nothing here", return_tensors="pt")
        rel = parser.build_relation_matrix(enc["input_ids"], parser.tokenizer, [shop_bundle], [[]], [[]])
        assert rel.shape == (1, enc["input_ids"].shape[1], enc["input_ids"].shape[1])
        assert int(rel.sum()) == 0

    def test_forward_loss_is_finite_with_and_without_rel(self, tiny_t5_dir: Path, shop_bundle) -> None:
        parser = GraphixCodeT5Parser.from_pretrained_dir(tiny_t5_dir)
        enc = parser.tokenizer("Question: how many tickets\nSchema:\n[TBL] tickets: [COL] id INTEGER", return_tensors="pt")
        labels = parser.tokenizer("SELECT COUNT(*) FROM tickets", return_tensors="pt")["input_ids"]
        L = enc["input_ids"].shape[1]
        rel = torch.zeros((1, L, L), dtype=torch.long)
        out_rel = parser(enc["input_ids"], enc["attention_mask"], labels=labels, rel_matrix=rel)
        out_plain = parser(enc["input_ids"], enc["attention_mask"], labels=labels)
        for out in (out_rel, out_plain):
            assert torch.isfinite(out.loss).item()


class TestSchemaLinker:
    def test_prune_with_empty_candidates(self, tiny_modernbert_dir: Path, shop_bundle) -> None:
        linker = ModernBertSchemaLinker(tiny_modernbert_dir).eval()
        assert linker.prune_schema("q", shop_bundle, [], []) == []

    def test_value_match_boost_selects_table(self, tiny_modernbert_dir: Path, shop_bundle, monkeypatch) -> None:
        """Tables carrying a grounded value match must survive pruning even when the
        linker head scores everything at zero (boost floor 0.75 > threshold 0.30)."""
        linker = ModernBertSchemaLinker(tiny_modernbert_dir).eval()
        monkeypatch.setattr(
            ModernBertSchemaLinker, "forward",
            lambda self, input_ids, attention_mask, table_labels=None, column_labels=None: {
                "loss": torch.tensor(0.0),
                "table_logits": [torch.zeros(0)],
                "column_logits": [torch.zeros(0)],
            },
        )
        vm = [ValueMatch(table_name="tickets", column_name="status", value="open", matched_phrase="open", score=1.0)]
        selected = linker.prune_schema(
            "open tickets", shop_bundle, shop_bundle.tables, vm,
            threshold=0.30, min_tables=1, max_tables=8, max_length=256,
        )
        names = {t.name for t in selected}
        assert "tickets" in names

    def test_min_tables_fallback(self, tiny_modernbert_dir: Path, shop_bundle, monkeypatch) -> None:
        linker = ModernBertSchemaLinker(tiny_modernbert_dir).eval()
        monkeypatch.setattr(
            ModernBertSchemaLinker, "forward",
            lambda self, input_ids, attention_mask, table_labels=None, column_labels=None: {
                "loss": torch.tensor(0.0),
                "table_logits": [torch.zeros(len(shop_bundle.tables))],
                "column_logits": [torch.zeros(0)],
            },
        )
        selected = linker.prune_schema(
            "q", shop_bundle, shop_bundle.tables, [],
            threshold=0.30, min_tables=2, max_tables=8, max_length=256,
        )
        # all-zero scores still keep the min-tables floor (FK bridging may add one more)
        assert 2 <= len(selected) <= 3

    def test_save_load_roundtrip_preserves_weights(self, tiny_modernbert_dir: Path, tmp_path: Path, shop_db) -> None:
        linker = ModernBertSchemaLinker(tiny_modernbert_dir)
        with torch.no_grad():
            linker.table_head[1].weight.add_(0.5)
        out_dir = tmp_path / "linker"
        linker.save_pretrained(out_dir)
        loaded = ModernBertSchemaLinker.from_pretrained_dir(out_dir)
        assert torch.allclose(loaded.table_head[1].weight, linker.table_head[1].weight)


class TestVerifier:
    def test_empty_candidates_returns_empty(self, tiny_modernbert_dir: Path) -> None:
        verifier = ModernBertVerifier(tiny_modernbert_dir).eval()
        assert verifier.score_candidates("q", "schema", []) == []

    def test_zero_temperature_is_clamped(self, tiny_modernbert_dir: Path, monkeypatch) -> None:
        verifier = ModernBertVerifier(tiny_modernbert_dir).eval()
        with torch.no_grad():
            verifier.temperature.fill_(0.0)
        monkeypatch.setattr(
            ModernBertVerifier, "forward",
            lambda self, input_ids, attention_mask, labels=None: {"logits": torch.zeros(input_ids.shape[0])},
        )
        scores = verifier.score_candidates("q", "schema", [("SELECT 1", "ok rows=1"), ("SELECT 2", "ok rows=2")])
        assert len(scores) == 2 and all(s == 0.5 for s in scores)  # sigmoid(0/0.05)

    def test_scores_are_bounded_probabilities(self, tiny_modernbert_dir: Path, shop_bundle) -> None:
        verifier = ModernBertVerifier(tiny_modernbert_dir).eval()
        scores = verifier.score_candidates(
            "how many open tickets", "Schema: [TBL] tickets", [("SELECT 1", "ok rows=1")]
        )
        assert len(scores) == 1 and 0.0 <= scores[0] <= 1.0


class TestDenseEncoderAndRetriever:
    def test_empty_encode_matches_model_dimension(self, tiny_modernbert_dir: Path) -> None:
        enc = DenseTableEncoder(str(tiny_modernbert_dir), device="cpu")
        empty = enc.encode([])
        one = enc.encode(["hello world"])
        assert empty.shape == (0, one.shape[1])
        assert float(np.linalg.norm(one[0]) - 1.0) < 1e-4  # L2-normalized

    def test_retriever_respects_small_table_budget(self) -> None:
        """max_candidate_tables=3 previously produced 5 tables via a negative slice."""
        tables = [
            TableInfo(
                name=f"t{i}",
                row_count=10,
                columns=[ColumnInfo(name="id", data_type="INTEGER", is_pk=True), ColumnInfo(name=f"c{i}", data_type="TEXT")],
            )
            for i in range(6)
        ]
        bundle = SchemaBundle(
            db_id="big", dialect="sqlite", ddl_hash="x1", tables=tables,
            join_edges=[JoinEdge(left_table=f"t{i}", left_column="id", right_table=f"t{(i + 1) % 6}", right_column="id") for i in range(5)],
        )
        retriever = HybridSchemaRetriever(embedder=None)
        for budget in (1, 2, 3, 5, 6):
            res = retriever.retrieve("t0 c0", bundle, value_index=None, max_candidate_tables=budget)
            assert 0 < len(res.candidate_tables) <= budget, f"budget {budget} violated: {len(res.candidate_tables)}"

    def test_retriever_cap_on_many_tables(self) -> None:
        tables = [
            TableInfo(
                name=f"big_{i}",
                row_count=5,
                columns=[ColumnInfo(name="id", data_type="INTEGER", is_pk=True), ColumnInfo(name="note", data_type="TEXT")],
            )
            for i in range(40)
        ]
        edges = [JoinEdge(left_table="big_0", left_column="id", right_table=f"big_{i}", right_column="id") for i in range(1, 5)]
        bundle = SchemaBundle(db_id="huge", dialect="sqlite", ddl_hash="h1", tables=tables, join_edges=edges)
        retriever = HybridSchemaRetriever(embedder=None)
        res = retriever.retrieve("big_0 note", bundle, value_index=None, max_candidate_tables=24)
        assert len(res.candidate_tables) <= 24
        assert any(t.name == "big_0" for t in res.candidate_tables)

    def test_retrieval_is_deterministic(self) -> None:
        tables = [
            TableInfo(name=f"tab{i}", row_count=5, columns=[ColumnInfo(name="id", data_type="INTEGER", is_pk=True)])
            for i in range(8)
        ]
        bundle = SchemaBundle(db_id="det", dialect="sqlite", ddl_hash="d1", tables=tables)
        retriever = HybridSchemaRetriever(embedder=None)
        r1 = retriever.retrieve("tab", bundle, value_index=None, max_candidate_tables=4)
        r2 = retriever.retrieve("tab", bundle, value_index=None, max_candidate_tables=4)
        assert [t.name for t in r1.candidate_tables] == [t.name for t in r2.candidate_tables]

    def test_small_schema_returns_all_tables(self, shop_bundle) -> None:
        retriever = HybridSchemaRetriever(embedder=None)
        res = retriever.retrieve("anything", shop_bundle, value_index=None, max_candidate_tables=24)
        assert len(res.candidate_tables) == len(shop_bundle.tables)
