"""StructCBR semantic layer: mine historical SQL logs and dbt-style YAML into a bundle.

Human-written artifacts -- a warehouse's query log, the analytics team's dbt
schema descriptions -- already encode most of the knowledge the engine has to
guess at introspection time: which tables join to which, what a column means,
and which query shapes are typical. This module extracts that structure with
sqlglot (deterministic AST walks, no models) and merges it into a cached
``SchemaBundle``:

* join edges observed repeatedly in the log but absent from declared/inferred
  FKs become ``JoinEdge(source="query_log")`` entries;
* dbt YAML column descriptions become ``business_description`` values;
* deduplicated representative queries populate ``example_queries`` so the
  retriever can inject them as few-shot hints into parser prompts.
"""
from __future__ import annotations

from pathlib import Path

import sqlglot
import yaml
from sqlglot import exp

from nl2sql.db.introspect import JoinEdge, SchemaBundle


def _join_on_pairs(sql: str) -> list[tuple[str, str, str, str]]:
    """Extract (left_table, left_col, right_table, right_col) pairs from JOIN ONs.

    Only equality conditions between two qualified columns of joined tables are
    considered; anything else (USING, expressions, aliases we cannot resolve)
    is skipped rather than guessed.
    """
    try:
        ast = sqlglot.parse_one(sql, read=None)
    except Exception:
        return []
    if ast is None:
        return []
    pairs: list[tuple[str, str, str, str]] = []
    for join in ast.find_all(exp.Join):
        on = join.args.get("on")
        if on is None:
            continue
        for eq in [n for n in on.find_all(exp.EQ)]:
            l, r = eq.this, eq.expression
            if not (isinstance(l, exp.Column) and isinstance(r, exp.Column)):
                continue
            if not (l.table and r.table and l.name and r.name):
                continue
            pairs.append((l.table, l.name, r.table, r.name))
    return pairs


def mine_join_edges_from_logs(
    sql_texts: list[str],
    bundle: SchemaBundle,
    min_frequency: int = 2,
) -> list[JoinEdge]:
    """Join edges observed at least ``min_frequency`` times that the bundle lacks.

    Frequency gating (default 2+) keeps one-off exploratory or mistaken queries
    from entering the hypergraph; canonical signatures are deduplicated against
    edges the bundle already carries in either direction.
    """
    known: set[tuple[str, str, str, str]] = set()
    for e in bundle.join_edges:
        for sig in (
            (e.left_table.lower(), e.left_column.lower(), e.right_table.lower(), e.right_column.lower()),
            (e.right_table.lower(), e.right_column.lower(), e.left_table.lower(), e.left_column.lower()),
        ):
            known.add(sig)

    counts: dict[tuple[str, str, str, str], int] = {}
    for sql in sql_texts:
        for lt, lc, rt, rc in _join_on_pairs(sql):
            t_l, t_r = bundle.get_table(lt), bundle.get_table(rt)
            if t_l is None or t_r is None:
                continue  # aliased or unknown reference: never guess
            c_l, c_r = t_l.get_column(lc), t_r.get_column(rc)
            if c_l is None or c_r is None:
                continue
            sig = (t_l.name.lower(), c_l.name.lower(), t_r.name.lower(), c_r.name.lower())
            counts[sig] = counts.get(sig, 0) + 1

    edges: list[JoinEdge] = []
    seen: set[tuple[str, str, str, str]] = set()
    for (lt, lc, rt, rc), freq in sorted(counts.items()):
        if freq < min_frequency or (lt, lc, rt, rc) in known or (rt, rc, lt, lc) in known:
            continue
        if (lt, lc, rt, rc) in seen or (rt, rc, lt, lc) in seen:
            continue
        seen.add((lt, lc, rt, rc))
        edges.append(
            JoinEdge(
                left_table=bundle.get_table(lt).name,
                left_column=bundle.get_table(lt).get_column(lc).name,
                right_table=bundle.get_table(rt).name,
                right_column=bundle.get_table(rt).get_column(rc).name,
                cardinality="N:1",
                source="query_log",
                confidence=round(min(0.95, 0.5 + 0.1 * freq), 3),
            )
        )
    return edges


def apply_dbt_descriptions(yaml_path: str | Path, bundle: SchemaBundle) -> bool:
    """Merge dbt schema-YAML descriptions into a bundle's column meanings.

    Accepts the standard dbt structure::

        version: 2
        models:
          - name: customers
            columns:
              - name: id
                description: primary key
    """
    p = Path(yaml_path)
    if not p.exists():
        return False
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return False
    changed = False
    for model in doc.get("models") or []:
        table = bundle.get_table(str(model.get("name") or ""))
        if table is None:
            continue
        for col in model.get("columns") or []:
            cname = str(col.get("name") or "")
            cinfo = table.get_column(cname) if cname else None
            desc = col.get("description")
            if cinfo is not None and desc and desc != cinfo.business_description:
                cinfo.business_description = str(desc)
                changed = True
    return changed


def mine_example_queries(
    sql_texts: list[str],
    bundle: SchemaBundle,
    max_examples: int = 3,
) -> list[dict[str, str]]:
    """Deduplicated representative queries usable as retriever few-shot hints.

    Dedup key: the sorted set of referenced tables + aggregates, so the same
    shape written with different literals collapses to one example.
    """
    seen_keys: set[tuple] = set()
    examples: list[dict[str, str]] = []
    for sql in sql_texts:
        try:
            ast = sqlglot.parse_one(sql, read=None)
        except Exception:
            continue
        if ast is None or not isinstance(ast, (exp.Select, exp.Union)):
            continue
        tables = tuple(sorted({t.name.lower() for t in ast.find_all(exp.Table) if t.name}))
        if not tables:
            continue
        aggs = tuple(sorted(type(a).__name__ for a in ast.find_all(exp.AggFunc)))
        key = (tables, aggs)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        examples.append({"question": "", "sql": " ".join(sql.split())})
        if len(examples) >= max_examples:
            break
    return examples


def enrich_bundle_from_logs(
    bundle: SchemaBundle,
    sql_texts: list[str],
    min_frequency: int = 2,
    max_examples: int = 3,
) -> int:
    """Populate join_edges and example_queries of ``bundle`` from query logs.

    Returns the number of newly added edges. The bundle is mutated in place;
    callers persist it via ``bundle.save()``.
    """
    edges = mine_join_edges_from_logs(sql_texts, bundle, min_frequency=min_frequency)
    if edges:
        bundle.join_edges.extend(edges)
    examples = mine_example_queries(sql_texts, bundle, max_examples=max_examples)
    if examples:
        existing = {(e.get("sql") or "").strip() for e in bundle.example_queries}
        for ex in examples:
            if ex["sql"] not in existing:
                bundle.example_queries.append(ex)
                existing.add(ex["sql"])
    return len(edges)
