from __future__ import annotations

import re
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

# Aggregates whose value is invariant under row duplication caused by 1:N fan-out.
# MIN/MAX ignore duplicates, and DISTINCT-inside-aggregate collapses them, so only
# plain SUM/AVG/COUNT can silently multiply when two N:1 children are joined.
_FANOUT_INVARIANT_FUNCS = {"MIN", "MAX"}

# Aggregates the automatic CTE pre-aggregation rewrite knows how to lower: the CTE
# pre-computes SUM(x)/COUNT(x) per parent key and the outer query re-aggregates with
# SUM (SUM of pre-summed per-parent values under any grouping granularity; COUNT of
# per-parent counts likewise, COALESCEd to 0 for parents with no child rows).
_REWRITABLE_FANOUT_FUNCS = (exp.Sum, exp.Count)


def _select_arg_key(name: str) -> str:
    """Resolve a logical Select arg name to this sqlglot version's actual key.

    sqlglot renamed ``from`` -> ``from_`` and ``with`` -> ``with_`` across versions;
    arg_types is the authoritative registry, so writing through the right key keeps
    the compiler compatible with both generations.
    """
    if name in exp.Select.arg_types:
        return name
    if f"{name}_" in exp.Select.arg_types:
        return f"{name}_"
    return name


class SQLSecurityError(ValueError):
    """Raised when generated SQL violates read-only AST or RBAC security policies."""


class CompiledSQL(BaseModel):
    sql: str
    used_tables: list[str] = Field(default_factory=list)
    join_repairs: list[str] = Field(default_factory=list)
    fanout_warnings: list[str] = Field(default_factory=list)
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


def _nearest_select(node: exp.Expression) -> Optional[exp.Select]:
    """Closest enclosing Select of a node (aggregates resolve in their own scope)."""
    parent = getattr(node, "parent", None)
    while parent is not None and not isinstance(parent, exp.Select):
        parent = getattr(parent, "parent", None)
    return parent if isinstance(parent, exp.Select) else None


def _build_select_scope(select: exp.Select) -> dict[str, str]:
    """Map the tables (and their aliases) of *this* Select's FROM/JOINs to canonical names.

    Scoped to a single Select so aliases declared inside subqueries or CTEs cannot
    leak into (or shadow) the outer scope used for fan-out resolution and join
    synthesis. Join-synthesis must emit ``o.customer_id`` when the query aliased
    orders as ``o``; emitting the raw table name produces invalid SQL on aliased
    queries.
    """
    scope: dict[str, str] = {}

    def _add_source_table(tbl: exp.Table) -> None:
        tname = tbl.name
        if not tname:
            return
        alias_or_name = tbl.alias_or_name
        scope.setdefault(tname.lower(), tname)
        if alias_or_name:
            scope.setdefault(alias_or_name.lower(), tname)

    from_expr = select.args.get("from_") or select.args.get("from")
    from_table = getattr(from_expr, "this", None) if from_expr is not None else None
    if isinstance(from_table, exp.Table):
        _add_source_table(from_table)
    for j in select.args.get("joins") or []:
        if isinstance(j.this, exp.Table):
            _add_source_table(j.this)
    return scope


def _select_from_table(select: exp.Select) -> Optional[exp.Table]:
    from_expr = select.args.get("from_") or select.args.get("from")
    from_table = getattr(from_expr, "this", None) if from_expr is not None else None
    return from_table if isinstance(from_table, exp.Table) else None


def _agg_is_fanout_sensitive(agg: exp.AggFunc) -> bool:
    name = (getattr(agg, "sql_name", lambda: "")() or "").upper()
    if not name:
        return False
    if name in _FANOUT_INVARIANT_FUNCS:
        return False
    if agg.find(exp.Distinct) is not None:
        # COUNT/SUM/AVG(DISTINCT x) collapses duplicated rows.
        return False
    return True


def _is_inside_cte(node: exp.Expression) -> bool:
    """True when the node sits inside a CTE (i.e. its result is pre-aggregated)."""
    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, exp.CTE):
            return True
        parent = getattr(parent, "parent", None)
    return False


def _detect_fanout_tables(ast: exp.Expression, bundle: SchemaBundle) -> list[str]:
    """Identify multiple 1:N child tables whose *fan-out-sensitive* aggregates are
    computed simultaneously around a shared parent table.

    Fan-out-invariant aggregates (MIN/MAX, DISTINCT variants), aggregates already
    pre-aggregated inside CTEs, and children without a shared parent are excluded so
    that safe queries are not flagged (which would force the engine to abstain).
    """
    aggs = [a for a in ast.find_all(exp.AggFunc) if _agg_is_fanout_sensitive(a) and not _is_inside_cte(a)]
    if len(aggs) < 2:
        return []

    # Alias resolution is per-Select: each column resolves against the FROM/JOINs of
    # the Select that actually encloses it, never against aliases declared in some
    # sibling subquery.
    scope_cache: dict[int, dict[str, str]] = {}

    def _scope_of(col: exp.Column) -> dict[str, str]:
        sel = _nearest_select(col)
        key = id(sel) if sel is not None else 0
        if key not in scope_cache:
            scope_cache[key] = _build_select_scope(sel) if sel is not None else {}
        return scope_cache[key]

    agg_tables: set[str] = set()
    for agg in aggs:
        for col in agg.find_all(exp.Column):
            if col.table:
                # Canonical names must be lowercased: mixed-case table names ("Orders",
                # common in BIRD) would otherwise miss the lowercase join-edge keys
                # below and silently skip fan-out detection.
                agg_tables.add(_scope_of(col).get(col.table.lower(), col.table.lower()).lower())
    if len(agg_tables) < 2:
        return []

    # parents[t] = set of 1-side tables that t joins to through an N:1 edge.
    parents: dict[str, set[str]] = {}
    for e in bundle.join_edges:
        if e.cardinality == "N:1":
            parents.setdefault(e.left_table.lower(), set()).add(e.right_table.lower())
        elif e.cardinality == "1:N":
            parents.setdefault(e.right_table.lower(), set()).add(e.left_table.lower())

    candidates = [t for t in agg_tables if parents.get(t)]
    if len(candidates) < 2:
        return []
    shared = set.intersection(*(parents[t] for t in candidates))
    if not shared:
        # Children aggregate against unrelated parents: no cross-fanout between them.
        return []
    return sorted(candidates)


def _agg_encloses(col: exp.Column, agg: exp.Expression) -> bool:
    node: Optional[exp.Expression] = col
    while node is not None:
        if node is agg:
            return True
        node = getattr(node, "parent", None)
    return False


def _fanout_parent_edge(bundle: SchemaBundle, child: str, shared: set[str]) -> Optional[JoinEdge]:
    """The single unambiguous N:1 edge from ``child`` to a shared parent table."""
    hits: list[JoinEdge] = []
    for e in bundle.join_edges:
        left, right = e.left_table.lower(), e.right_table.lower()
        if left == child and e.cardinality == "N:1" and right in shared:
            hits.append(e)
        elif right == child and e.cardinality == "1:N" and left in shared:
            hits.append(e)
    if len(hits) != 1:
        # No edge, or several candidate parents for this child: rewriting would
        # require guessing the join key -- fall back to the warning path.
        return None
    return hits[0]


def _cte_preaggregate_fanout(ast: exp.Select, bundle: SchemaBundle, fanout_tables: list[str]) -> list[str]:
    """Rewrite diverging 1:N child-table aggregates into per-parent CTE pre-aggregation.

    The canonical fan-out trap joins two N:1 children of one parent and computes
    plain SUM/COUNT over both; every child row is duplicated once per sibling-child
    row, silently multiplying the aggregates. Instead of abstaining, each child
    table is lowered into its own CTE that pre-aggregates per parent key::

        WITH __preagg_orders AS (
            SELECT customer_id AS _pk, SUM(amount) AS _v0
            FROM orders GROUP BY customer_id
        )
        SELECT c.name, SUM(__preagg_orders._v0), COALESCE(SUM(__preagg_tickets._v1), 0)
        FROM customers c
        LEFT JOIN __preagg_orders ON __preagg_orders._pk = c.id
        LEFT JOIN __preagg_tickets ON __preagg_tickets._pk = c.id
        GROUP BY c.name

    The rewrite is only applied when it is provably faithful: every flagged child
    column must live inside a rewritable SUM/COUNT aggregate of the top Select's
    projections (or in the child's own join ON, which is removed with the join),
    the child must sit on the JOIN side (not be the FROM base), each child needs a
    single unambiguous edge to the shared parent, and no CTEs may pre-exist.
    Returns the rewritten child table names, or an empty list when any check fails
    (the caller then keeps the warning/abstention path).

    SUM pre-aggregates re-aggregate with SUM (NULL for childless parents, matching
    SQL's SUM-over-empty semantics); COUNT pre-aggregates re-aggregate with
    COALESCE(SUM(...), 0) so childless parents report 0, matching COUNT semantics.
    """
    if not isinstance(ast, exp.Select):
        return []
    # find_all returns a generator (always truthy) -- force evaluation.
    if ast.args.get(_select_arg_key("with")) is not None or any(True for _ in ast.find_all(exp.CTE)):
        return []

    scope = _build_select_scope(ast)
    from_table = _select_from_table(ast)
    from_canon = from_table.name.lower() if from_table is not None else None
    joins = [j for j in (ast.args.get("joins") or []) if isinstance(j.this, exp.Table)]
    join_by_table: dict[str, exp.Join] = {}
    for j in joins:
        tname = scope.get(j.this.name.lower(), j.this.name)
        join_by_table.setdefault(tname.lower(), j)

    # Shared parents of the flagged children (same computation as detection).
    parents: dict[str, set[str]] = {}
    for e in bundle.join_edges:
        if e.cardinality == "N:1":
            parents.setdefault(e.left_table.lower(), set()).add(e.right_table.lower())
        elif e.cardinality == "1:N":
            parents.setdefault(e.right_table.lower(), set()).add(e.left_table.lower())
    shared = set.intersection(*(parents[t] for t in fanout_tables))

    # Every fanout-sensitive aggregate of the top Select's projections must be a
    # plain SUM/COUNT over one flagged table's qualified column.
    target_aggs: dict[str, list[exp.AggFunc]] = {t: [] for t in fanout_tables}
    for proj in ast.expressions:
        for agg in proj.find_all(exp.AggFunc):
            if not _agg_is_fanout_sensitive(agg):
                continue
            cols = [c for c in agg.find_all(exp.Column)]
            tables = {scope.get(c.table.lower(), c.table.lower()).lower() for c in cols if c.table}
            if len(tables) != 1:
                return []  # COUNT(*) or mixed/unqualified columns: not attributable
            tbl = tables.pop()
            if not (isinstance(agg, _REWRITABLE_FANOUT_FUNCS) and tbl in target_aggs):
                return []  # AVG or aggregates over non-flagged tables: bail out
            if _nearest_select(agg) is not ast:
                return []  # aggregate lives in a subquery projection: cannot lower
            target_aggs[tbl].append(agg)

    for t in fanout_tables:
        if not target_aggs[t]:
            return []  # flagged without a rewritable aggregate: keep warning path
        if t == from_canon or t not in join_by_table:
            return []  # child is the FROM base or missing from the JOIN side

    # No flagged-table column may appear anywhere except its own aggregates and
    # its own join's ON (removed together with the join).
    forbidden_join_nodes = {t: join_by_table[t] for t in fanout_tables}
    for col in ast.find_all(exp.Column):
        if not col.table:
            continue
        canon = scope.get(col.table.lower(), col.table.lower()).lower()
        if canon not in target_aggs or _nearest_select(col) is not ast:
            continue
        allowed = any(_agg_encloses(col, agg) for agg in target_aggs[canon])
        if not allowed:
            node: Optional[exp.Expression] = col
            while node is not None:
                if isinstance(node, exp.Join) and node is forbidden_join_nodes.get(canon):
                    allowed = True
                    break
                node = getattr(node, "parent", None)
        if not allowed:
            return []

    cte_names = {c.alias_or_name.lower() for c in ast.find_all(exp.CTE) if c.alias_or_name}
    edges: dict[str, JoinEdge] = {}
    parent_refs: dict[str, str] = {}
    for t in fanout_tables:
        edge = _fanout_parent_edge(bundle, t, shared)
        if edge is None:
            return []
        edges[t] = edge
        t_obj = bundle.get_table(t)
        p_obj = bundle.get_table(edge.right_table if edge.left_table.lower() == t else edge.left_table)
        if t_obj is None or p_obj is None:
            return []
        # Parent reference spelling: the parent's alias in THIS query if present.
        parent_canon = (edge.right_table if edge.left_table.lower() == t else edge.left_table).lower()
        parent_tbl_node = _table_node_for(scope, ast, parent_canon)
        parent_refs[t] = parent_tbl_node.alias_or_name if parent_tbl_node is not None else p_obj.name
        cte_name = f"__preagg_{t_obj.name.lower()}"
        if cte_name in cte_names or not _is_safe_ident(cte_name):
            return []
        cte_names.add(cte_name)

    # Build one CTE per flagged child: pre-aggregate each rewritable aggregate per parent key.
    cte_nodes: list[exp.CTE] = []
    new_join_nodes: list[exp.Join] = []
    for t in fanout_tables:
        t_obj = bundle.get_table(t)
        edge = edges[t]
        fk_col = edge.left_column if edge.left_table.lower() == t else edge.right_column
        pk_col = edge.right_column if edge.left_table.lower() == t else edge.left_column
        cte_name = f"__preagg_{t_obj.name.lower()}"
        select_parts = [f"{_q(fk_col)} AS _pk"]
        for i, agg in enumerate(target_aggs[t]):
            arg_cols = [c for c in agg.find_all(exp.Column)]
            arg = _q(arg_cols[0].name) if arg_cols else "*"
            inner = ("SUM" if isinstance(agg, exp.Sum) else "COUNT") + f"({arg})"
            select_parts.append(f"{inner} AS _v{i}")
        cte_body = sqlglot.parse_one(
            f"SELECT {', '.join(select_parts)} FROM {_q(t_obj.name)} GROUP BY {_q(fk_col)}",
            read="sqlite",
        )
        cte_nodes.append(exp.CTE(this=cte_body, alias=exp.to_identifier(cte_name)))
        on_cond = exp.EQ(
            this=exp.Column(this=exp.to_identifier("_pk"), table=exp.to_identifier(cte_name)),
            expression=exp.Column(this=exp.to_identifier(pk_col), table=exp.to_identifier(parent_refs[t])),
        )
        j = exp.Join(this=exp.Table(this=exp.to_identifier(cte_name)))
        j.set("side", "LEFT")
        j.set("on", on_cond)
        new_join_nodes.append(j)
        # Replace the flagged aggregates with outer re-aggregation over the CTE columns.
        for i, agg in enumerate(target_aggs[t]):
            outer = exp.Sum(this=exp.Column(this=exp.to_identifier(f"_v{i}"), table=exp.to_identifier(cte_name)))
            if isinstance(agg, exp.Count):
                # COUNT semantics: childless parents must yield 0, not NULL.
                outer = exp.Coalesce(this=outer, expressions=[exp.Literal.number(0)])
            agg.replace(outer)

    # Drop the flagged children's joins (their ON clauses vanish with them) and
    # attach the pre-aggregation CTEs via LEFT JOINs. Identity filtering: sqlglot
    # Expression equality is not guaranteed to be identity, and removing a
    # deep-equal sibling join would corrupt the query.
    flagged_join_ids = {id(join_by_table[t]) for t in fanout_tables}
    remaining_joins = [j for j in (ast.args.get("joins") or []) if id(j) not in flagged_join_ids]
    ast.set("joins", remaining_joins + new_join_nodes)
    with_key = _select_arg_key("with")
    ast.set(with_key, exp.With(expressions=cte_nodes))
    return fanout_tables


def _table_node_for(scope: dict[str, str], ast: exp.Select, canon: str) -> Optional[exp.Table]:
    ft = _select_from_table(ast)
    if ft is not None and scope.get(ft.name.lower(), ft.name).lower() == canon:
        return ft
    for j in ast.args.get("joins") or []:
        if isinstance(j.this, exp.Table) and scope.get(j.this.name.lower(), j.this.name).lower() == canon:
            return j.this
    return None


def _q(ident: str) -> str:
    """Quote an identifier only when it is not already a simple safe name."""
    if _is_safe_ident(ident):
        return ident
    return '"' + ident.replace('"', '""') + '"'


def _is_safe_ident(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


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
    # Function-name gating only applies to Anonymous calls (the node type that
    # actually carries the function's literal name); typed nodes like Count expose
    # .name as their FIRST ARGUMENT, so the old `hasattr(fn, 'name')` fallback was
    # dead code that compared argument text against function names.
    for fn in ast.find_all(exp.Anonymous):
        fn_name = (fn.name or "").lower()
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
            scope = _build_select_scope(ast)
            base_name = from_expr.this.name if isinstance(from_expr.this, exp.Table) else None
            base_tbl = scope.get(base_name.lower()) if base_name else None
            # table name (lower) -> reference spelling (alias or name) to use in ON clauses
            seen_refs: dict[str, str] = {}
            seen_tbls: set[str] = set()
            if base_tbl:
                base_ref = from_expr.this.alias_or_name if isinstance(from_expr.this, exp.Table) else base_tbl
                seen_refs[base_tbl.lower()] = base_ref or base_tbl
                seen_tbls.add(base_tbl)
            for j in joins:
                j_tbl = j.this.name if isinstance(j.this, exp.Table) else None
                if not j_tbl:
                    continue
                j_canon = scope.get(j_tbl.lower(), j_tbl)
                j_ref = j.this.alias_or_name if isinstance(j.this, exp.Table) else j_canon
                if _is_trivial_on(j.args.get("on")) and j.args.get("using") is None and seen_tbls:
                    # Try single-hop or shortest path from any already-joined table
                    chosen_path: Optional[list[JoinEdge]] = None
                    for prev in sorted(seen_tbls):
                        paths = _find_join_path(bundle.join_edges, prev, j_canon)
                        if len(paths) > 1:
                            ambiguity_flags.append(f"Multiple join paths between {prev} and {j_canon}")
                        if paths:
                            chosen_path = max(paths, key=lambda p: sum(e.confidence for e in p))
                            break
                    if chosen_path and len(chosen_path) == 1:
                        e = chosen_path[0]
                        # Resolve each endpoint to the alias actually used in this query:
                        # the new table uses the join's own alias, the other side uses
                        # an alias already visible in the FROM scope.
                        left_ref = j_ref if e.left_table.lower() == j_canon.lower() else seen_refs.get(e.left_table.lower(), e.left_table)
                        right_ref = j_ref if e.right_table.lower() == j_canon.lower() else seen_refs.get(e.right_table.lower(), e.right_table)
                        on_cond = exp.EQ(
                            this=exp.Column(this=exp.to_identifier(e.left_column), table=exp.to_identifier(left_ref)),
                            expression=exp.Column(this=exp.to_identifier(e.right_column), table=exp.to_identifier(right_ref)),
                        )
                        j.set("on", on_cond)
                        join_repairs.append(f"Synthesized ON {e.left_table}.{e.left_column} = {e.right_table}.{e.right_column}")
                    else:
                        ambiguity_flags.append(f"Unconstrained join on table {j_canon}")
                seen_refs.setdefault(j_canon.lower(), j_ref or j_canon)
                seen_tbls.add(j_canon)

    # Diverging 1:N multi-fact fan-out traps: rewrite into per-parent CTE
    # pre-aggregation whenever provably faithful; only when the rewrite cannot
    # be applied safely does the compiler fall back to warning + abstention.
    fanout_tables = _detect_fanout_tables(ast, bundle)
    fanout_warnings: list[str] = []
    if len(fanout_tables) >= 2:
        rewritten = _cte_preaggregate_fanout(ast, bundle, fanout_tables) if isinstance(ast, exp.Select) else []
        if rewritten:
            fanout_warnings.append(
                f"Auto CTE pre-aggregation applied for diverging 1:N tables ({', '.join(rewritten)}): "
                "aggregates are computed per parent key before joining"
            )
        else:
            fanout_warnings.append(
                f"Fan-out risk detected across multiple 1:N tables ({', '.join(fanout_tables)}); review CTE pre-aggregation"
            )
            ambiguity_flags.append(f"Potential M:N fan-out aggregation across {', '.join(fanout_tables)}")

    # Enforce safe LIMIT on unaggregated top-level selects/unions lacking LIMIT.
    # A top-level UNION without LIMIT previously escaped the guardrail entirely:
    # only the executor's hard row cap protected production callers.
    if isinstance(ast, (exp.Select, exp.Union)) and ast.args.get("limit") is None:
        has_top_agg = any(True for _ in ast.expressions if _ and _.find(exp.AggFunc))
        if not has_top_agg and default_limit > 0:
            ast = ast.limit(default_limit)

    final_sql = ast.sql(dialect=target_dialect)
    return CompiledSQL(
        sql=final_sql,
        used_tables=referenced_tables,
        join_repairs=join_repairs,
        fanout_warnings=fanout_warnings,
        ambiguity_flags=ambiguity_flags,
    )
