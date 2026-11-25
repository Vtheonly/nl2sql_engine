from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from transformers import AutoModel

from nl2sql.utils.hf_compat import load_tokenizer

from nl2sql.db.introspect import SchemaBundle, TableInfo
from nl2sql.db.value_index import ValueIndex, ValueMatch


class RetrievalResult(BaseModel):
    candidate_tables: list[TableInfo]
    value_matches: list[ValueMatch] = Field(default_factory=list)
    example_hints: list[dict[str, str]] = Field(default_factory=list)


def _tokenize_ident(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]


class DenseTableEncoder:
    """Compact BGE-small bi-encoder for column/table semantic recall."""

    def __init__(self, model_id_or_path: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(model_id_or_path)
        self.model = AutoModel.from_pretrained(model_id_or_path).to(self.device).eval()
        # Read the real hidden size so the empty-batch placeholder matches the model's
        # output dimension even when a non-384-dim encoder is configured.
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 384))

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            out = self.model(**enc).last_hidden_state[:, 0]
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
            vecs.append(out.cpu().numpy())
        return np.vstack(vecs).astype(np.float32)


class HybridSchemaRetriever:
    """Combines FTS5 cell probing, BM25, BGE-small dense vectors, and 1-hop Hypergraph expansion."""

    def __init__(self, embedder: Optional[DenseTableEncoder] = None) -> None:
        self.embedder = embedder
        self._table_cache: dict[str, tuple[str, BM25Okapi, np.ndarray]] = {}

    def _build_table_docs(self, bundle: SchemaBundle) -> tuple[list[list[str]], list[str], list[list[str]]]:
        """Return (BM25 token docs, per-column docs, table-level fallback docs).

        Dense retrieval encodes every column separately and max-pools per table:
        concatenating all columns into one 256-token string made columns 25-60 of a
        wide table invisible to the embedder.
        """
        tok_docs: list[list[str]] = []
        col_docs: list[list[str]] = []
        str_docs: list[str] = []
        for t in bundle.tables:
            cols = [f"{c.name} ({c.business_description or c.comment or ''})" for c in t.columns]
            full = f"table {t.name} {t.comment or ''} columns: " + ", ".join(cols)
            str_docs.append(full)
            tok_docs.append(_tokenize_ident(full))
            col_docs.append(cols)
        return tok_docs, str_docs, col_docs

    def _get_indexes(self, bundle: SchemaBundle) -> tuple[BM25Okapi, np.ndarray]:
        cached = self._table_cache.get(bundle.db_id)
        if cached and cached[0] == bundle.ddl_hash:
            return cached[1], cached[2]
        tok_docs, str_docs, col_docs = self._build_table_docs(bundle)
        bm25 = BM25Okapi(tok_docs)
        if self.embedder is not None:
            flat_cols = [doc for cols in col_docs for doc in cols]
            col_vecs = self.embedder.encode(flat_cols) if flat_cols else np.zeros((0, self.embedder.hidden_size), dtype=np.float32)
            # Max-pool column vectors per table: one strong column match is enough to
            # retrieve the table, regardless of how many columns it has.
            tbl_vecs = np.zeros((len(col_docs), col_vecs.shape[1]), dtype=np.float32)
            offset = 0
            for i, cols in enumerate(col_docs):
                if cols:
                    tbl_vecs[i] = col_vecs[offset : offset + len(cols)].max(axis=0)
                offset += len(cols)
            # Blend the pooled column vector with the table-level doc vector so the
            # table name/comment signal is not lost for tables with weak columns.
            doc_vecs = self.embedder.encode(str_docs)
            norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            tbl_vecs = tbl_vecs + doc_vecs / np.maximum(norms, 1e-8)
            vecs = tbl_vecs
        else:
            vecs = np.zeros((len(str_docs), 1), dtype=np.float32)
        self._table_cache[bundle.db_id] = (bundle.ddl_hash, bm25, vecs)
        return bm25, vecs

    def retrieve(
        self,
        question: str,
        bundle: SchemaBundle,
        value_index: Optional[ValueIndex] = None,
        max_candidate_tables: int = 24,
    ) -> RetrievalResult:
        val_hits = value_index.probe(question, top_k=8) if value_index else []
        # Registered example queries (per-database adaptation signal) ride along with
        # the retrieval result; the engine injects them into the parser prompt.
        hints = [dict(h) for h in (bundle.example_queries or [])[:3]]
        if len(bundle.tables) <= max_candidate_tables:
            return RetrievalResult(candidate_tables=bundle.tables, value_matches=val_hits, example_hints=hints)

        bm25, tbl_vecs = self._get_indexes(bundle)
        q_toks = _tokenize_ident(question)
        bm25_scores = np.array(bm25.get_scores(q_toks), dtype=np.float32)
        # BM25Okapi emits negative scores for terms present in > half the documents;
        # clipping keeps the blend with cosine similarity monotonic.
        bm25_scores = np.maximum(bm25_scores, 0.0)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        if self.embedder is not None and tbl_vecs.shape[1] > 1:
            q_vec = self.embedder.encode([question])[0]
            dense_scores = tbl_vecs @ q_vec
        else:
            dense_scores = np.zeros_like(bm25_scores)

        val_tables = {v.table_name for v in val_hits}
        combined = []
        for idx, t in enumerate(bundle.tables):
            bonus = 0.35 if t.name in val_tables else 0.0
            score = 0.45 * float(bm25_scores[idx]) + 0.55 * float(dense_scores[idx]) + bonus
            combined.append((score, t))

        combined.sort(key=lambda x: x[0], reverse=True)
        # `max_candidate_tables - 4` must never go negative: a negative slice bound
        # (e.g. max=3 -> combined[:-1]) silently returned MORE tables than requested.
        n_seeds = max(1, max_candidate_tables - 4)
        picked = {t.name: t for _, t in combined[:n_seeds]}

        # 1-hop FK bridge expansion from Hypergraph; hard-capped at the caller's budget
        for edge in bundle.join_edges:
            if len(picked) >= max_candidate_tables:
                break
            if edge.left_table in picked and edge.right_table not in picked:
                rt = bundle.get_table(edge.right_table)
                if rt:
                    picked[rt.name] = rt
            elif edge.right_table in picked and edge.left_table not in picked:
                lt = bundle.get_table(edge.left_table)
                if lt:
                    picked[lt.name] = lt

        # Final guard: retrieval must never exceed the caller's table budget.
        ordered_names = [t.name for _, t in combined if t.name in picked]
        picked_set = set(ordered_names)
        extras = [n for n in picked if n not in picked_set]
        return RetrievalResult(candidate_tables=[picked[n] for n in (ordered_names + extras)[:max_candidate_tables]], value_matches=val_hits, example_hints=hints)
