from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import sqlglot
from sqlglot import exp
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score

from nl2sql.datasets.prepare import ExampleRecord
from nl2sql.utils.logging import get_resource_stats


class EvalMetrics(BaseModel):
    dataset_name: str
    total_examples: int
    executable_rate: float
    execution_accuracy: float
    exact_match: float
    r_ves: float
    schema_table_recall: float = 0.0
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_qps: float
    abstention_rate: float = 0.0
    selective_execution_accuracy: float = 0.0
    verifier_auroc: Optional[float] = None
    by_difficulty: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_source: dict[str, dict[str, float]] = Field(default_factory=dict)
    resource_stats: dict[str, Any] = Field(default_factory=dict)


_READONLY_FIRST_TOKENS = {"SELECT", "WITH"}


def _first_keyword(sql: str) -> str:
    """First meaningful keyword of a statement, skipping leading comments/whitespace."""
    s = (sql or "").lstrip()
    while True:
        if s.startswith("--"):
            nl = s.find("\n")
            s = "" if nl < 0 else s[nl + 1 :].lstrip()
        elif s.startswith("/*"):
            end = s.find("*/")
            s = "" if end < 0 else s[end + 2 :].lstrip()
        else:
            break
    parts = s.split(None, 1)
    return parts[0].upper() if parts else ""


def execute_sqlite_readonly(
    db_path: Path,
    sql: str,
    timeout_sec: float = 3.0,
    max_rows: int = 50000,
) -> tuple[bool, Optional[list[tuple[Any, ...]]], float, str]:
    start = time.perf_counter()
    # Guard 1 (statement gate): SQLite's `mode=ro` URI only protects the MAIN database;
    # an `ATTACH DATABASE` on a read-only connection can still create and write files
    # elsewhere on disk. Only statements that begin with SELECT/WITH are ever executed,
    # so ATTACH/PRAGMA/WRITE statements are rejected before reaching the engine.
    first_kw = _first_keyword(sql)
    if first_kw not in _READONLY_FIRST_TOKENS:
        return False, None, 0.0, f"Blocked non-readonly statement (first keyword: {first_kw or '<empty>'})"
    uri = f"file:{db_path.resolve()}?mode=ro"
    deadline = start + timeout_sec
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_sec)
        # Guard 2 (connection gate): query_only makes the whole connection -- including
        # any ATTACHed database -- reject writes, in depth behind the statement gate.
        conn.execute("PRAGMA query_only = 1")

        def _progress() -> int:
            return 1 if time.perf_counter() > deadline else 0

        conn.set_progress_handler(_progress, 10000)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return True, [tuple(r) for r in rows], elapsed_ms, "ok"
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return False, None, elapsed_ms, str(exc)
    finally:
        if conn is not None:
            conn.close()


# Integers beyond 2**53 lose low bits through float():.4f formatting (18-digit
# account IDs would compare equal after rounding). Keep them exact as strings.
_SAFE_FLOAT_INT_BOUND = 2 ** 53


def _normalize_row(row: tuple[Any, ...]) -> tuple[str, ...]:
    out = []
    for v in row:
        # Ints and floats must canonicalize to the same string: SQL engines freely
        # retype COUNT(*) between INTEGER and REAL, and an int-vs-float difference
        # must not fail an otherwise-identical result set.
        if isinstance(v, bool):
            out.append("1" if v else "0")
        elif isinstance(v, int):
            if abs(v) <= _SAFE_FLOAT_INT_BOUND:
                out.append(f"{float(v):.4f}")
            else:
                out.append(str(v))
        elif isinstance(v, float):
            if math.isfinite(v) and abs(v) <= _SAFE_FLOAT_INT_BOUND:
                out.append(f"{v:.4f}")
            else:
                out.append(repr(v))
        elif v is None:
            out.append("NULL")
        else:
            out.append(str(v).strip())
    return tuple(out)


def _gold_is_ordered(gold_sql: str) -> bool:
    """True only when the OUTERMOST query carries ORDER BY.

    A text search for `order by` also fires on subquery/CTE-internal ordering, which
    does not define the final row order and forced unnecessarily strict comparisons.
    """
    try:
        ast = sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return "order by" in gold_sql.lower()
    return isinstance(ast, (exp.Select, exp.Union)) and ast.args.get("order") is not None


def results_equivalent(gold_sql: str, gold_rows: list[tuple], pred_rows: list[tuple]) -> bool:
    g_norm = [_normalize_row(r) for r in gold_rows]
    p_norm = [_normalize_row(r) for r in pred_rows]
    ordered = _gold_is_ordered(gold_sql)
    if ordered:
        return g_norm == p_norm
    return sorted(g_norm) == sorted(p_norm)


def normalize_sql_em(sql: str) -> str:
    try:
        return sqlglot.transpile(sql, read="sqlite", write="sqlite")[0].lower().strip()
    except Exception:
        return " ".join(sql.lower().split())


class NL2SQLEvaluator:
    """Comprehensive execution, structural, latency, selective-prediction, and R-VES evaluator."""

    def evaluate(
        self,
        records: list[ExampleRecord],
        predict_fn: Callable[[ExampleRecord], dict[str, Any]],
        dataset_name: str = "eval",
    ) -> EvalMetrics:
        latencies: list[float] = []
        exec_ok = 0
        ex_hits = 0
        em_hits = 0
        rves_sum = 0.0
        tbl_recalls: list[float] = []
        abstained = 0
        non_abstained_hits = 0
        verifier_scores: list[float] = []
        verifier_labels: list[int] = []
        # Per-slice accounting (BIRD difficulty: simple/moderate/challenging; source:
        # bird vs spider) so regressions on a slice are visible, not averaged away.
        slice_totals: dict[str, int] = {}
        slice_hits: dict[str, int] = {}

        def _bump(group: str, key: str, hit: bool) -> None:
            totals = slice_totals.setdefault(f"{group}:{key}", 0)
            slice_totals[f"{group}:{key}"] = totals + 1
            if hit:
                slice_hits[f"{group}:{key}"] = slice_hits.get(f"{group}:{key}", 0) + 1

        t_wall_start = time.perf_counter()
        for rec in records:
            t0 = time.perf_counter()
            pred = predict_fn(rec)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            pred_sql = pred.get("sql")
            is_abstained = bool(pred.get("abstained", False)) or not pred_sql
            conf = float(pred.get("confidence", 0.0))

            linked_tbls = {t.lower() for t in pred.get("linked_tables", [])}
            if rec.gold_tables:
                g_tbls = {t.lower() for t in rec.gold_tables}
                tbl_recalls.append(len(linked_tbls & g_tbls) / len(g_tbls))

            if is_abstained:
                abstained += 1
                verifier_scores.append(conf)
                verifier_labels.append(0)
                continue

            if normalize_sql_em(pred_sql) == normalize_sql_em(rec.sql):
                em_hits += 1

            g_ok, g_rows, g_ms, _ = execute_sqlite_readonly(rec.db_path, rec.sql)
            p_ok, p_rows, p_ms, _ = execute_sqlite_readonly(rec.db_path, pred_sql)
            if p_ok:
                exec_ok += 1

            correct = bool(g_ok and p_ok and g_rows is not None and p_rows is not None and results_equivalent(rec.sql, g_rows, p_rows))
            _bump("difficulty", rec.difficulty or "unknown", correct)
            _bump("source", rec.source or "unknown", correct)
            verifier_scores.append(conf)
            verifier_labels.append(1 if correct else 0)
            if correct:
                ex_hits += 1
                non_abstained_hits += 1
                ratio = math.sqrt(max(g_ms, 0.05) / max(p_ms, 0.05))
                if ratio >= 2.0:
                    rves_sum += 1.25
                elif ratio >= 1.0:
                    rves_sum += 1.0
                elif ratio >= 0.5:
                    rves_sum += 0.75
                else:
                    rves_sum += 0.5

        total = max(len(records), 1)
        answered = max(total - abstained, 1)
        wall_sec = max(time.perf_counter() - t_wall_start, 1e-6)
        auroc: Optional[float] = None
        if len(set(verifier_labels)) == 2:
            auroc = round(float(roc_auc_score(verifier_labels, verifier_scores)), 4)

        def _group_breakdown(group: str) -> dict[str, dict[str, float]]:
            out: dict[str, dict[str, float]] = {}
            for key in sorted({k.split(":", 1)[1] for k in slice_totals if k.startswith(f"{group}:")}):
                total = slice_totals.get(f"{group}:{key}", 0)
                hits = slice_hits.get(f"{group}:{key}", 0)
                out[key] = {
                    "total": total,
                    "execution_accuracy": round(hits / max(total, 1), 4),
                }
            return out

        return EvalMetrics(
            dataset_name=dataset_name,
            total_examples=len(records),
            executable_rate=round(exec_ok / total, 4),
            execution_accuracy=round(ex_hits / total, 4),
            exact_match=round(em_hits / total, 4),
            r_ves=round(rves_sum / total, 4),
            schema_table_recall=round(float(np.mean(tbl_recalls)) if tbl_recalls else 0.0, 4),
            p50_latency_ms=round(float(np.percentile(latencies, 50)) if latencies else 0.0, 2),
            p95_latency_ms=round(float(np.percentile(latencies, 95)) if latencies else 0.0, 2),
            throughput_qps=round(len(records) / wall_sec, 2),
            abstention_rate=round(abstained / total, 4),
            selective_execution_accuracy=round(non_abstained_hits / answered, 4),
            verifier_auroc=auroc,
            by_difficulty=_group_breakdown("difficulty"),
            by_source=_group_breakdown("source"),
            resource_stats=get_resource_stats(),
        )
