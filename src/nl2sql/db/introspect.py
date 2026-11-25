from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    is_pk: bool = False
    is_unique: bool = False
    comment: Optional[str] = None
    business_description: Optional[str] = None
    sample_values: list[str] = Field(default_factory=list)


class TableInfo(BaseModel):
    name: str
    row_count: int = 0
    comment: Optional[str] = None
    columns: list[ColumnInfo] = Field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def get_column(self, col_name: str) -> Optional[ColumnInfo]:
        low = col_name.lower()
        for c in self.columns:
            if c.name.lower() == low:
                return c
        return None


class JoinEdge(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    cardinality: Literal["1:1", "N:1", "1:N", "M:N"] = "N:1"
    source: Literal["declared_fk", "inferred_fk", "query_log"] = "declared_fk"
    confidence: float = 1.0


class SchemaBundle(BaseModel):
    db_id: str
    dialect: str
    ddl_hash: str
    tables: list[TableInfo]
    join_edges: list[JoinEdge] = Field(default_factory=list)
    example_queries: list[dict[str, str]] = Field(default_factory=list)

    def get_table(self, name: str) -> Optional[TableInfo]:
        low = name.lower()
        for t in self.tables:
            if t.name.lower() == low:
                return t
        return None

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "SchemaBundle":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _quote_ident(engine: Engine, ident: str) -> str:
    return engine.dialect.identifier_preparer.quote(ident)


def _is_type_compatible(t1: str, t2: str) -> bool:
    s1, s2 = t1.lower(), t2.lower()
    int_t = ("int", "bigint", "smallint", "serial", "numeric")
    str_t = ("char", "text", "varchar", "string", "uuid")
    if any(k in s1 for k in int_t) and any(k in s2 for k in int_t):
        return True
    if any(k in s1 for k in str_t) and any(k in s2 for k in str_t):
        return True
    return s1.split("(")[0].strip() == s2.split("(")[0].strip()


def _infer_edge_cardinality(left_col: ColumnInfo, right_col: ColumnInfo) -> Literal["1:1", "N:1", "1:N", "M:N"]:
    left_unique = left_col.is_pk or left_col.is_unique
    right_unique = right_col.is_pk or right_col.is_unique
    if left_unique and right_unique:
        return "1:1"
    if not left_unique and right_unique:
        return "N:1"
    if left_unique and not right_unique:
        return "1:N"
    return "M:N"


def _normalize_stem(name: str) -> str:
    stem = re.sub(r"(_id|id)$", "", name.lower()).strip("_")
    return stem[:-1] if stem.endswith("s") and len(stem) > 3 else stem


def _probe_inclusion(engine: Engine, lt: str, lc: str, rt: str, rc: str, sample_limit: int = 300) -> float:
    ql, qlc = _quote_ident(engine, lt), _quote_ident(engine, lc)
    qr, qrc = _quote_ident(engine, rt), _quote_ident(engine, rc)
    sql = text(
        f"SELECT COUNT(*) AS total, "
        f"SUM(CASE WHEN s.v IN (SELECT {qrc} FROM {qr} WHERE {qrc} IS NOT NULL) THEN 1 ELSE 0 END) AS hits "
        f"FROM (SELECT DISTINCT {qlc} AS v FROM {ql} WHERE {qlc} IS NOT NULL LIMIT :lim) s"
    )
    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {"lim": sample_limit}).fetchone()
            if not row or not row[0] or row[0] == 0:
                return 0.0
            return float(row[1] or 0) / float(row[0])
    except Exception:
        return 0.0


def introspect_database(
    connection_url: str,
    db_id: str,
    column_meanings: Optional[dict[str, str]] = None,
    sample_values_per_col: int = 4,
    infer_missing_fks: bool = True,
) -> SchemaBundle:
    engine = create_engine(connection_url)
    insp = inspect(engine)
    dialect = engine.dialect.name
    tables: list[TableInfo] = []
    raw_fks: list[tuple[str, str, str, str]] = []
    column_meanings = column_meanings or {}

    with engine.connect() as conn:
        for table_name in sorted(insp.get_table_names()):
            if table_name.startswith("sqlite_"):
                continue
            pk_info = insp.get_pk_constraint(table_name) or {}
            pk_cols = set(pk_info.get("constrained_columns") or [])
            unique_cols: set[str] = set()
            for uq in insp.get_unique_constraints(table_name) or []:
                cols = uq.get("column_names") or []
                if len(cols) == 1:
                    unique_cols.add(cols[0])
            for idx in insp.get_indexes(table_name) or []:
                cols = idx.get("column_names") or []
                if idx.get("unique") and len(cols) == 1 and cols[0]:
                    unique_cols.add(cols[0])

            t_comment = None
            try:
                t_comment = (insp.get_table_comment(table_name) or {}).get("text")
            except NotImplementedError:
                pass

            qt = _quote_ident(engine, table_name)
            try:
                row_count = int(conn.execute(text(f"SELECT COUNT(*) FROM {qt}")).scalar() or 0)
            except Exception:
                row_count = 0

            col_infos: list[ColumnInfo] = []
            for col in insp.get_columns(table_name):
                cname = col["name"]
                qc = _quote_ident(engine, cname)
                samples: list[str] = []
                if sample_values_per_col > 0 and row_count > 0:
                    try:
                        rows = conn.execute(
                            text(
                                f"SELECT DISTINCT {qc} FROM {qt} WHERE {qc} IS NOT NULL LIMIT :lim"
                            ),
                            {"lim": sample_values_per_col},
                        ).fetchall()
                        samples = [str(r[0])[:64] for r in rows if r[0] is not None]
                    except Exception:
                        samples = []

                key_exact = f"{db_id}|{table_name}|{cname}"
                key_low = key_exact.lower()
                biz_desc = column_meanings.get(key_exact) or column_meanings.get(key_low)
                col_infos.append(
                    ColumnInfo(
                        name=cname,
                        data_type=str(col["type"]),
                        nullable=bool(col.get("nullable", True)),
                        is_pk=(cname in pk_cols and len(pk_cols) == 1),
                        is_unique=(cname in unique_cols),
                        comment=col.get("comment"),
                        business_description=biz_desc,
                        sample_values=samples,
                    )
                )

            for fk in insp.get_foreign_keys(table_name) or []:
                ref_table = fk.get("referred_table")
                lcols = fk.get("constrained_columns") or []
                rcols = fk.get("referred_columns") or []
                if ref_table and len(lcols) == len(rcols):
                    for lc, rc in zip(lcols, rcols):
                        if lc and rc:
                            raw_fks.append((table_name, lc, ref_table, rc))

            tables.append(TableInfo(name=table_name, row_count=row_count, comment=t_comment, columns=col_infos))

    table_map = {t.name.lower(): t for t in tables}
    edges: list[JoinEdge] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    for lt, lc, rt, rc in raw_fks:
        t_l, t_r = table_map.get(lt.lower()), table_map.get(rt.lower())
        if not t_l or not t_r:
            continue
        c_l, c_r = t_l.get_column(lc), t_r.get_column(rc)
        if not c_l or not c_r:
            continue
        card = _infer_edge_cardinality(c_l, c_r)
        sig = (t_l.name, c_l.name, t_r.name, c_r.name)
        if sig not in seen_pairs:
            seen_pairs.add(sig)
            edges.append(
                JoinEdge(
                    left_table=t_l.name,
                    left_column=c_l.name,
                    right_table=t_r.name,
                    right_column=c_r.name,
                    cardinality=card,
                    source="declared_fk",
                    confidence=1.0,
                )
            )

    if infer_missing_fks and len(tables) > 1:
        pk_targets: list[tuple[TableInfo, ColumnInfo]] = [
            (t, c) for t in tables for c in t.columns if c.is_pk or c.is_unique
        ]
        for t_l in tables:
            if t_l.row_count == 0:
                continue
            for c_l in t_l.columns:
                if not (c_l.name.lower().endswith("id") or c_l.is_pk):
                    continue
                l_stem = _normalize_stem(c_l.name)
                for t_r, c_r in pk_targets:
                    if t_l.name == t_r.name or t_r.row_count == 0:
                        continue
                    sig = (t_l.name, c_l.name, t_r.name, c_r.name)
                    rev_sig = (t_r.name, c_r.name, t_l.name, c_l.name)
                    if sig in seen_pairs or rev_sig in seen_pairs:
                        continue
                    if not _is_type_compatible(c_l.data_type, c_r.data_type):
                        continue
                    r_tbl_stem = _normalize_stem(t_r.name)
                    name_match = (c_l.name.lower() == c_r.name.lower() and c_l.name.lower() != "id") or (
                        l_stem and l_stem == r_tbl_stem
                    )
                    if not name_match:
                        continue
                    ratio = _probe_inclusion(engine, t_l.name, c_l.name, t_r.name, c_r.name)
                    if ratio >= 0.85:
                        seen_pairs.add(sig)
                        edges.append(
                            JoinEdge(
                                left_table=t_l.name,
                                left_column=c_l.name,
                                right_table=t_r.name,
                                right_column=c_r.name,
                                cardinality=_infer_edge_cardinality(c_l, c_r),
                                source="inferred_fk",
                                confidence=round(0.6 + 0.35 * ratio, 3),
                            )
                        )

    ddl_repr = json.dumps(
        [{"t": t.name, "c": [(c.name, c.data_type, c.is_pk) for c in t.columns]} for t in tables],
        sort_keys=True,
    )
    ddl_hash = hashlib.sha256(ddl_repr.encode("utf-8")).hexdigest()[:16]
    engine.dispose()
    return SchemaBundle(db_id=db_id, dialect=dialect, ddl_hash=ddl_hash, tables=tables, join_edges=edges)
