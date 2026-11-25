from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import create_engine, text

from nl2sql.db.introspect import SchemaBundle


class ValueMatch(BaseModel):
    table_name: str
    column_name: str
    value: str
    matched_phrase: str
    score: float


def _is_text_like(dtype: str) -> bool:
    d = dtype.lower()
    return any(k in d for k in ("char", "text", "varchar", "string", "clob", "enum"))


def _clean_fts_token(token: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_\s.-]", " ", token).strip()


class ValueIndex:
    """SQLite FTS5 persistent distinct cell-value index for literal grounding."""

    def __init__(self, index_db_path: str | Path) -> None:
        self.path = Path(index_db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS cell_fts USING fts5("
                "table_name UNINDEXED, column_name UNINDEXED, raw_value, norm_value, "
                "tokenize='unicode61')"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)"
            )

    def is_built_for(self, ddl_hash: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT value FROM index_meta WHERE key='ddl_hash'").fetchone()
            return bool(row and row[0] == ddl_hash)

    def build_from_db(
        self,
        connection_url: str,
        bundle: SchemaBundle,
        max_distinct_per_col: int = 10000,
    ) -> int:
        if self.is_built_for(bundle.ddl_hash):
            return 0
        engine = create_engine(connection_url)
        prep = engine.dialect.identifier_preparer
        total_rows = 0
        with sqlite3.connect(self.path) as idx_conn:
            idx_conn.execute("DELETE FROM cell_fts")
            with engine.connect() as db_conn:
                for table in bundle.tables:
                    qt = prep.quote(table.name)
                    for col in table.columns:
                        if not _is_text_like(col.data_type):
                            continue
                        qc = prep.quote(col.name)
                        try:
                            rows = db_conn.execute(
                                text(
                                    f"SELECT DISTINCT {qc} FROM {qt} "
                                    f"WHERE {qc} IS NOT NULL LIMIT :lim"
                                ),
                                {"lim": max_distinct_per_col},
                            ).fetchall()
                        except Exception:
                            continue
                        batch = []
                        for (val,) in rows:
                            s = str(val).strip()
                            if not s or len(s) > 120:
                                continue
                            norm = re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z]+", " ", s.lower())).strip()
                            batch.append((table.name, col.name, s, norm))
                        if batch:
                            idx_conn.executemany(
                                "INSERT INTO cell_fts(table_name, column_name, raw_value, norm_value) VALUES (?, ?, ?, ?)",
                                batch,
                            )
                            total_rows += len(batch)
            idx_conn.execute(
                "INSERT OR REPLACE INTO index_meta(key, value) VALUES ('ddl_hash', ?)",
                (bundle.ddl_hash,),
            )
            idx_conn.commit()
        engine.dispose()
        return total_rows

    def probe(self, question: str, top_k: int = 8) -> list[ValueMatch]:
        words = [w for w in re.findall(r"[A-Za-z0-9_.-]+", question) if len(w) >= 2]
        ngrams: list[str] = []
        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if n == 1 and phrase.lower() in {
                    "the", "and", "for", "with", "from", "what", "which", "how",
                    "many", "much", "show", "list", "find", "give", "where", "who",
                    "average", "total", "count", "sum", "max", "min", "top", "most",
                }:
                    continue
                ngrams.append(phrase)

        seen_hits: dict[tuple[str, str, str], ValueMatch] = {}
        with sqlite3.connect(self.path) as conn:
            # Multi-word phrases carry the most grounding signal, but unigrams must never
            # be squeezed out by the phrase budget: on long questions the 4/3/2-grams alone
            # exceeded the old 36-phrase cap, so single-token values (e.g. a customer name
            # appearing late in the question) silently stopped matching.
            multi_grams = [g for g in ngrams if " " in g][:24]
            single_grams = [g for g in ngrams if " " not in g][:24]
            probe_phrases = multi_grams + single_grams
            for phrase in probe_phrases:
                cleaned = _clean_fts_token(phrase)
                tokens = [t for t in cleaned.split() if len(t) >= 2]
                if not tokens:
                    continue
                fts_query = " AND ".join(f'"{t}"' for t in tokens)
                try:
                    rows = conn.execute(
                        "SELECT table_name, column_name, raw_value, bm25(cell_fts) AS rank "
                        "FROM cell_fts WHERE cell_fts MATCH ? ORDER BY rank LIMIT 4",
                        (fts_query,),
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                for tbl, col, raw_val, bm25_rank in rows:
                    exact_bonus = 2.0 if raw_val.lower() == phrase.lower() else 0.0
                    phrase_bonus = 0.25 * len(tokens)
                    score = exact_bonus + phrase_bonus - float(bm25_rank)
                    key = (tbl, col, raw_val)
                    if key not in seen_hits or score > seen_hits[key].score:
                        seen_hits[key] = ValueMatch(
                            table_name=tbl,
                            column_name=col,
                            value=raw_val,
                            matched_phrase=phrase,
                            score=round(score, 4),
                        )

        ranked = sorted(seen_hits.values(), key=lambda m: m.score, reverse=True)
        return ranked[:top_k]
