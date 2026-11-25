from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import sqlglot
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nl2sql.compiler.sql_compiler import SQLSecurityError, compile_and_validate_sql
from nl2sql.config import EngineConfig
from nl2sql.datasets.prepare import format_schema_for_parser
from nl2sql.db.introspect import SchemaBundle, introspect_database
from nl2sql.db.value_index import ValueIndex
from nl2sql.evaluation.evaluator import execute_sqlite_readonly
from nl2sql.models.graphix_parser import GraphixCodeT5Parser
from nl2sql.models.schema_linker import ModernBertSchemaLinker
from nl2sql.models.verifier import ModernBertVerifier
from nl2sql.retrieval.schema_retriever import DenseTableEncoder, HybridSchemaRetriever


# Value-bearing patterns that make a question unsafe for template caching. This must
# stay in sync with the substitution set used by NL2SQLEngine._skeleton_key, otherwise
# two questions with different literals can collapse onto the same cache entry.
_CACHE_UNSAFE_RE = re.compile(r"\d|'[^']*'|\"[^\"]*\"")

# Upper bound on cached SQL templates so long-lived services cannot grow unbounded.
_TEMPLATE_CACHE_MAX = 1024


class QueryRequest(BaseModel):
    db_id: str
    question: str
    evidence: str = ""
    target_dialect: str = "sqlite"
    allowed_tables: Optional[list[str]] = None
    execute: bool = False


class QueryResponse(BaseModel):
    db_id: str
    sql: Optional[str] = None
    abstained: bool = False
    reason: str = "ok"
    confidence: float = 0.0
    linked_tables: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    cache_hit: bool = False
    rows: Optional[list[list[Any]]] = None


class NL2SQLEngine:
    """Complete runtime NL2SQL engine implementing Layers 0 through 7."""

    def __init__(
        self,
        cfg: EngineConfig,
        artifacts_dir: str | Path,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(device, cfg.training.device)
        art = Path(artifacts_dir)

        emb_path = art / "embedder" if (art / "embedder").exists() else cfg.models.embedder_model_id
        self.retriever = HybridSchemaRetriever(DenseTableEncoder(str(emb_path), device=self.device))

        self.linker: Optional[ModernBertSchemaLinker] = (
            ModernBertSchemaLinker.from_pretrained_dir(art / "linker", map_location=self.device).to(self.device)
            if (art / "linker").exists()
            else None
        )
        parser_path = art / "parser" if (art / "parser").exists() else cfg.models.parser_model_id
        self.parser = GraphixCodeT5Parser.from_pretrained_dir(parser_path, map_location=self.device).to(self.device)
        self.verifier: Optional[ModernBertVerifier] = (
            ModernBertVerifier.from_pretrained_dir(art / "verifier", map_location=self.device).to(self.device)
            if (art / "verifier").exists()
            else None
        )

        self.db_registry: dict[str, tuple[SchemaBundle, ValueIndex, Optional[Path]]] = {}
        # (db_id, question skeleton, evidence, dialect, allowed-tables signature) -> SQL.
        # Bounded LRU: entries are evicted oldest-first once the cap is reached.
        self._template_cache: "OrderedDict[tuple[str, str, str, str, Optional[tuple[str, ...]]], str]" = OrderedDict()
        self._cache_lock = threading.Lock()

    def register_database(self, db_id: str, connection_url: str, cache_dir: str | Path, sqlite_path: Optional[Path] = None) -> SchemaBundle:
        cdir = Path(cache_dir) / db_id
        bpath = cdir / "schema_bundle.json"
        bundle = SchemaBundle.load(bpath) if bpath.exists() else introspect_database(connection_url, db_id=db_id)
        if not bpath.exists():
            bundle.save(bpath)
        v_idx = ValueIndex(cdir / "value_index.sqlite")
        if not v_idx.is_built_for(bundle.ddl_hash):
            v_idx.build_from_db(connection_url, bundle, max_distinct_per_col=self.cfg.data.max_distinct_values_per_col)
        self.db_registry[db_id] = (bundle, v_idx, sqlite_path)
        return bundle

    @staticmethod
    def _resolve_device(device: Optional[str], cfg_device: Optional[str] = None) -> str:
        dev = device or cfg_device or "cpu"
        if isinstance(dev, str) and dev.startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"
        return dev

    @staticmethod
    def _normalize_question(text: str) -> str:
        # Lightweight deterministic normalization for relative dates / whitespace.
        # dateparser is imported lazily: it is a heavy import and only needed when a
        # relative date actually appears in the question.
        cleaned = " ".join(text.strip().split())
        if re.search(r"\byesterday\b", cleaned, flags=re.IGNORECASE):
            try:
                import dateparser

                dt = dateparser.parse("yesterday")
            except Exception:
                dt = None
            if dt:
                cleaned = re.sub(r"\byesterday\b", dt.strftime("%Y-%m-%d"), cleaned, flags=re.IGNORECASE)
        return cleaned

    @staticmethod
    def _skeleton_key(question: str) -> str:
        q = re.sub(r"\b\d+(?:\.\d+)?\b", "{NUM}", question.lower())
        q = re.sub(r"'[^']*'|\"[^\"]*\"", "{STR}", q)
        return " ".join(q.split())

    def answer(
        self,
        db_id: str,
        question: str,
        evidence: str = "",
        target_dialect: str = "sqlite",
        allowed_tables: Optional[list[str]] = None,
        execute: bool = False,
    ) -> QueryResponse:
        t0 = time.perf_counter()
        if db_id not in self.db_registry:
            raise KeyError(f"Database '{db_id}' is not registered.")
        bundle, v_idx, sqlite_path = self.db_registry[db_id]
        norm_q = self._normalize_question(question)
        # RBAC allowlists are matched case-insensitively, consistent with the compiler's
        # own table resolution (bundle.get_table lowercases the lookup).
        allowed_set = {t.lower() for t in allowed_tables} if allowed_tables is not None else None

        # Layer 2: Retrieval & Adaptive Schema Pruning
        ret = self.retriever.retrieve(norm_q, bundle, v_idx)
        cand_tables = [t for t in ret.candidate_tables if allowed_set is None or t.name.lower() in allowed_set]
        if self.linker is not None and len(cand_tables) > self.cfg.models.linker_max_tables:
            linked_tables = self.linker.prune_schema(
                norm_q, bundle, cand_tables, ret.value_matches, evidence=evidence,
                threshold=self.cfg.models.linker_threshold,
                min_tables=self.cfg.models.linker_min_tables,
                max_tables=self.cfg.models.linker_max_tables,
                max_length=self.cfg.models.linker_max_length,
            )
        else:
            linked_tables = cand_tables[: self.cfg.models.linker_max_tables]

        linked_names = [t.name for t in linked_tables]
        schema_txt = format_schema_for_parser(bundle, linked_tables, ret.value_matches, norm_q, evidence)

        # Layer 3: Verified Exact/Normalized Cache Fast Path.
        # The cache key MUST include everything that can change the grounded SQL:
        # evidence text, target dialect, and the caller's RBAC allowlist. Otherwise a
        # cached answer can silently leak tables the caller is not allowed to query,
        # or return SQL compiled for a different dialect / evidence context.
        cache_key = (
            db_id,
            self._skeleton_key(norm_q),
            evidence.strip(),
            target_dialect.strip().lower(),
            tuple(sorted(allowed_set)) if allowed_set is not None else None,
        )
        cached_sql: Optional[str] = None
        if _CACHE_UNSAFE_RE.search(norm_q) is None:
            with self._cache_lock:
                if cache_key in self._template_cache:
                    cached_sql = self._template_cache[cache_key]
                    self._template_cache.move_to_end(cache_key)
        if cached_sql is not None:
            out_rows = None
            if execute and sqlite_path is not None:
                ok_c, rows_c, _, _ = execute_sqlite_readonly(sqlite_path, cached_sql)
                out_rows = [list(r) for r in rows_c] if (ok_c and rows_c is not None) else None
            return QueryResponse(
                db_id=db_id, sql=cached_sql, confidence=0.99, linked_tables=linked_names,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2), cache_hit=True, rows=out_rows,
            )

        # Layer 4: Graphix-CodeT5+-220M + XGrammar Candidate Generation
        raw_cands = self.parser.generate_candidates(
            schema_txt, bundle, linked_tables, ret.value_matches,
            num_candidates=self.cfg.models.num_return_candidates,
            max_source_len=self.cfg.models.parser_max_source_length,
            max_target_len=self.cfg.models.parser_max_target_length,
            use_xgrammar=self.cfg.models.use_xgrammar,
        )

        # Layer 5 & 7: Compile, Repair Joins, Check Fan-Out & Security Policies + Layer 6 Dry-Run
        compiled_cands = []
        exec_inputs = []
        for raw_sql in raw_cands:
            try:
                comp = compile_and_validate_sql(raw_sql, bundle, target_dialect=target_dialect, allowed_tables=allowed_set)
            except SQLSecurityError:
                continue
            if sqlite_path is not None:
                ok, rows, _, msg = execute_sqlite_readonly(sqlite_path, comp.sql)
                summary = f"ok rows={len(rows)}" if ok and rows is not None else f"err={msg[:60]}"
            else:
                ok, rows, summary = True, None, "ast_valid"
            compiled_cands.append((comp, ok, rows, summary))
            exec_inputs.append((comp.sql, summary))

        if not compiled_cands:
            return QueryResponse(
                db_id=db_id, abstained=True, reason="No safe/valid SQL candidate compiled.",
                confidence=0.0, linked_tables=linked_names,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # Layer 6: Verifier Ranking + Calibration Gate
        if self.verifier is not None:
            scores = self.verifier.score_candidates(norm_q, schema_txt, exec_inputs, max_length=self.cfg.models.verifier_max_length)
        else:
            scores = [0.85 if ok else 0.15 for _, ok, _, _ in compiled_cands]

        ranked = sorted(zip(scores, compiled_cands), key=lambda x: x[0], reverse=True)
        best_score, (best_comp, best_ok, best_rows, _) = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0

        if not best_ok or best_score < self.cfg.models.abstention_threshold:
            return QueryResponse(
                db_id=db_id, abstained=True, reason="Low verifier confidence or failed execution.",
                confidence=round(best_score, 4), linked_tables=linked_names,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
        if best_comp.ambiguity_flags:
            return QueryResponse(
                db_id=db_id, abstained=True,
                reason=f"Clarification needed: {'; '.join(best_comp.ambiguity_flags)}",
                confidence=round(best_score, 4), linked_tables=linked_names,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
        if self.verifier is not None and len(ranked) > 1 and (best_score - second_score) < self.cfg.models.abstention_margin:
            # Two near-identical verifier scores mean the ranking carries no signal:
            # abstain instead of guessing. (Previously guarded by `and not best_ok`,
            # which was unreachable because failed candidates already returned above.)
            return QueryResponse(
                db_id=db_id, abstained=True, reason="Ambiguous top candidates with insufficient margin.",
                confidence=round(best_score, 4), linked_tables=linked_names,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        if _CACHE_UNSAFE_RE.search(norm_q) is None:
            with self._cache_lock:
                self._template_cache[cache_key] = best_comp.sql
                while len(self._template_cache) > _TEMPLATE_CACHE_MAX:
                    self._template_cache.popitem(last=False)

        out_rows = [list(r) for r in best_rows] if (execute and best_rows is not None) else None
        return QueryResponse(
            db_id=db_id, sql=best_comp.sql, abstained=False, reason="ok",
            confidence=round(best_score, 4), linked_tables=linked_names,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2), rows=out_rows,
        )


def create_app(engine: NL2SQLEngine) -> FastAPI:
    app = FastAPI(title="NL2SQL Engine API", version="0.1.0")

    @app.post("/query", response_model=QueryResponse)
    def query_sql(req: QueryRequest) -> QueryResponse:
        try:
            return engine.answer(
                db_id=req.db_id,
                question=req.question,
                evidence=req.evidence,
                target_dialect=req.target_dialect,
                allowed_tables=req.allowed_tables,
                execute=req.execute,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return app
