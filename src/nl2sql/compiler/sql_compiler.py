from __future__ import annotations

from collections import deque
from typing import Optional

import sqlglot
from sqlglot import exp
from pydantic import BaseModel, Field

from nl2sql.db.introspect import JoinEdge, SchemaBundle

FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)
FORBIDDEN_FUNCTIONS = {"pg_sleep", "lo_export", "lo_import", "load_extension", "sqlite_load_extension"}


class SQLSecurityError(ValueError):
    """Raised when generated SQL violates read-only AST or RBAC security policies."""


class CompiledSQL(BaseModel):
    sql: str
    used_tables: list[str] = Field(default_factory=list)
    join_repairs: list[str] = Field(default_factory=list)
    fanout_cte_rewrites: list[str] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)


def _find_join_path(edges: list[JoinEdge], src: str, dst: str) -> list[list[JoinEdge]]:
    adj: dict[str, list[tuple[str, JoinEdge]]] = {}
    for e in edges:
        adj.setdefault(e.left_table.lower(), []).append((e.right_table.lower(), e))
        adj.setdefault(e.right_table.lower(), []).append((e.left_table.lower(), e))

    q = deque([(src.lower(), [])])
    best_paths: list[list[JoinEdge]] = []
    best_len: Optional[int] = None
    visited_depth: dict[str, int] = {src.lower(): 0}

    while q:
        curr, path = q.popleft()
        if best_len is not None and len(path) > best_len:
            continue
        if curr == dst.lower():
            best_len = len(path)
            best_paths.append(path)
            continue
        for nxt, edge in adj.get(curr, []):
            depth = len(path) + 1
            if nxt not in visited_depth or visited_depth[nxt] >= depth:
                visited_depth[nxt] = depth
                q.append((nxt, path + [edge]))
    return best_paths


def _is_trivial_on(on: Optional[exp.Expression]) -> bool:
    """True when a JOIN has no real ON condition (sqlglot renders missing ON as ON TRUE)."""
    return on is None or (isinstance(on, exp.Boolean) and on.this is True)


def _detect_fanout_tables(ast: exp.Expression, bundle: SchemaBundle) -> list[str]:
    """Identify multiple 1:N child tables aggregated simultaneously around a shared parent."""
    aggs = list(ast.find_all(exp.AggFunc))
    if len(aggs) < 2:
        return []
    agg_tables: set[str] = set()
    for agg in aggs:
        for col in agg.find_all(exp.Column):
            if col.table:
                agg_tables.add(col.table.lower())
    if len(agg_tables) < 2:
        return []

    fanout_candidates: list[str] = []
    for e in bundle.join_edges:
        # N:1 means left_table is the 'N' (many) side pointing to right_table '1' side
        if e.cardinality == "N:1" and e.left_table.lower() in agg_tables:
            fanout_candidates.append(e.left_table)
        elif e.cardinality == "1:N" and e.right_table.lower() in agg_tables:
            fanout_candidates.append(e.right_table)
    return sorted(set(fanout_candidates))


def compile_and_validate_sql(
    raw_sql: str,
    bundle: SchemaBundle,
    target_dialect: str = "sqlite",
    allowed_tables: Optional[set[str]] = None,
    default_limit: int = 200,
) -> CompiledSQL:
    try:
        ast = sqlglot.parse_one(raw_sql, read="sqlite")
    except Exception as exc:
        raise SQLSecurityError(f"Unparseable SQL: {exc}") from exc

    if not isinstance(ast, (exp.Select, exp.Union)):
        raise SQLSecurityError(f"Only SELECT/UNION read queries are allowed, got {type(ast).__name__}")
    for forbidden in ast.find_all(*FORBIDDEN_NODES):
        raise SQLSecurityError(f"Forbidden SQL AST node detected: {type(forbidden).__name__}")
    for fn in ast.find_all(exp.Anonymous, exp.Func):
        fn_name = (fn.name if hasattr(fn, "name") else str(fn.sql_name())).lower()
        if fn_name in FORBIDDEN_FUNCTIONS:
            raise SQLSecurityError(f"Forbidden SQL function detected: {fn_name}")

    cte_names = {c.alias_or_name.lower() for c in ast.find_all(exp.CTE) if c.alias_or_name}
    referenced_tables: list[str] = []
    allowed_low = {t.lower() for t in allowed_tables} if allowed_tables is not None else None

    for tbl in ast.find_all(exp.Table):
        tname = tbl.name
        if not tname or tname.lower() in cte_names:
            continue
        t_obj = bundle.get_table(tname)
        if t_obj is None:
            raise SQLSecurityError(f"Referenced unknown table '{tname}' not in schema '{bundle.db_id}'")
        if allowed_low is not None and t_obj.name.lower() not in allowed_low:
            raise SQLSecurityError(f"RBAC violation: table '{t_obj.name}' is not allowed for caller")
        if t_obj.name not in referenced_tables:
            referenced_tables.append(t_obj.name)

    join_repairs: list[str] = []
    ambiguity_flags: list[str] = []

    # Check top-level Select joins: repair Cartesian JOINs lacking ON when Hypergraph has a clear FK path
    if isinstance(ast, exp.Select):
        # sqlglot renamed the FROM arg key across versions ("from" -> "from_"); read both.
        from_expr = ast.args.get("from_") or ast.args.get("from")
        joins = ast.args.get("joins") or []
        if from_expr and joins:
            base_tbl = from_expr.this.name if isinstance(from_expr.this, exp.Table) else None
            seen_tbls = {base_tbl} if base_tbl else set()
            for j in joins:
                j_tbl = j.this.name if isinstance(j.this, exp.Table) else None
                if not j_tbl:
                    continue
                if _is_trivial_on(j.args.get("on")) and j.args.get("using") is None and seen_tbls:
                    # Try single-hop or shortest path from any already-joined table
                    chosen_path: Optional[list[JoinEdge]] = None
                    for prev in sorted(seen_tbls):
                        paths = _find_join_path(bundle.join_edges, prev, j_tbl)
                        if len(paths) > 1:
                            ambiguity_flags.append(f"Multiple join paths between {prev} and {j_tbl}")
                        if paths:
                            chosen_path = max(paths, key=lambda p: sum(e.confidence for e in p))
                            break
                    if chosen_path and len(chosen_path) == 1:
                        e = chosen_path[0]
                        on_cond = exp.EQ(
                            this=exp.Column(this=exp.to_identifier(e.left_column), table=exp.to_identifier(e.left_table)),
                            expression=exp.Column(this=exp.to_identifier(e.right_column), table=exp.to_identifier(e.right_table)),
                        )
                        j.set("on", on_cond)
                        join_repairs.append(f"Synthesized ON {e.left_table}.{e.left_column} = {e.right_table}.{e.right_column}")
                    else:
                        ambiguity_flags.append(f"Unconstrained join on table {j_tbl}")
                seen_tbls.add(j_tbl)

    # Detect diverging 1:N multi-fact fan-out traps
    fanout_tables = _detect_fanout_tables(ast, bundle)
    fanout_rewrites: list[str] = []
    if len(fanout_tables) >= 2:
        fanout_rewrites.append(
            f"Fan-out risk detected across multiple 1:N tables ({', '.join(fanout_tables)}); review CTE pre-aggregation"
        )
        ambiguity_flags.append(f"Potential M:N fan-out aggregation across {', '.join(fanout_tables)}")

    # Enforce safe LIMIT on unaggregated top-level selects lacking LIMIT
    if isinstance(ast, exp.Select) and ast.args.get("limit") is None:
        has_top_agg = any(True for _ in ast.expressions if _ and _.find(exp.AggFunc))
        if not has_top_agg and default_limit > 0:
            ast = ast.limit(default_limit)

    final_sql = ast.sql(dialect=target_dialect)
    return CompiledSQL(
        sql=final_sql,
        used_tables=referenced_tables,
        join_repairs=join_repairs,
        fanout_cte_rewrites=fanout_rewrites,
        ambiguity_flags=ambiguity_flags,
    )
