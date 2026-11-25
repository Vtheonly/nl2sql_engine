from __future__ import annotations

import re
import sqlite3
import threading
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


def _is_low_cardinality_candidate(dtype: str) -> bool:
    """Non-text columns that carry grounding signal once cardinality is low.

    Integer status codes, dates and booleans appear verbatim in questions
    ("status 3", "on 2026-01-05"); high-cardinality numerics (ids, amounts)
    never match question text and only bloat the index.
    """
    d = dtype.lower()
    return any(k in d for k in ("int", "real", "float", "double", "decimal", "numeric", "bool", "date", "time"))


# Stopword set for unigram probing (matched case-insensitively). Single-letter
# English words are excluded so genuine enum codes ('A', 'F', 'M', 'Y') survive.
_UNIGRAM_STOPWORDS = {
    "the", "and", "for", "with", "from", "what", "which", "how",
    "many", "much", "show", "list", "find", "give", "where", "who",
    "average", "total", "count", "sum", "max", "min", "top", "most",
    "a", "i", "of", "in", "is", "on", "to", "by", "or", "as", "at",
}


def _clean_fts_token(token: str) -> str:
    # Unicode-aware: keep letters/digits/underscore (any script) so accented and CJK
    # values survive normalization instead of being stripped to nothing.
    return re.sub(r"[^\w\s.-]", " ", token, flags=re.UNICODE).strip()


def _norm_cell_value(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", s.lower(), flags=re.UNICODE)).strip()


class ValueIndex:
    """SQLite FTS5 persistent distinct cell-value index for literal grounding."""

    def __init__(self, index_db_path: str | Path) -> None:
        self.path = Path(index_db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS cell_fts USING fts5("
                "table_name UNINDEXED, column_name UNINDEXED, raw_value, norm_value, "
                "tokenize='unicode61')"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.commit()
        finally:
            conn.close()

    def _ro_conn(self) -> sqlite3.Connection:
        """Thread-local read-only connection for probe queries.

        Opening a fresh connection per probe call (the historical behavior) cost
        dozens of open/close cycles per question; mode=ro additionally guarantees
        the probe path can never write.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            self._local.conn = conn
        return conn

    def is_built_for(self, ddl_hash: str) -> bool:
        conn = sqlite3.connect(self.path)
        try:
            row = conn.execute("SELECT value FROM index_meta WHERE key='ddl_hash'").fetchone()
            return bool(row and row[0] == ddl_hash)
        finally:
            conn.close()

    def build_from_db(
        self,
        connection_url: str,
        bundle: SchemaBundle,
        max_distinct_per_col: int = 10000,
        low_cardinality_max_distinct: int = 100,
    ) -> int:
        if self.is_built_for(bundle.ddl_hash):
            return 0
        engine = create_engine(connection_url)
        prep = engine.dialect.identifier_preparer
        total_rows = 0
        idx_conn = sqlite3.connect(self.path)
        try:
            idx_conn.execute("DELETE FROM cell_fts")
            with engine.connect() as db_conn:
                for table in bundle.tables:
                    qt = prep.quote(table.name)
                    for col in table.columns:
                        text_like = _is_text_like(col.data_type)
                        if not text_like and not _is_low_cardinality_candidate(col.data_type):
                            continue
                        qc = prep.quote(col.name)
                        # Text columns: cap at max_distinct_per_col. Low-cardinality
                        # non-text columns (status codes, dates, booleans): only
                        # indexed when they carry few distinct values.
                        limit = (
                            max_distinct_per_col
                            if text_like
                            else low_cardinality_max_distinct + 1
                        )
                        try:
                            rows = db_conn.execute(
                                text(
                                    f"SELECT DISTINCT {qc} FROM {qt} "
                                    f"WHERE {qc} IS NOT NULL LIMIT :lim"
                                ),
                                {"lim": limit},
                            ).fetchall()
                        except Exception:
                            continue
                        if not text_like and len(rows) > low_cardinality_max_distinct:
                            # High-cardinality numeric/date column: no grounding
                            # signal, only index bloat.
                            continue
                        batch = []
                        for (val,) in rows:
                            s = str(val).strip()
                            if not s or len(s) > 120:
                                continue
                            norm = _norm_cell_value(s)
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
        finally:
            idx_conn.close()
        engine.dispose()
        return total_rows

    def probe(self, question: str, top_k: int = 8) -> list[ValueMatch]:
        # Unicode word extraction: \w keeps accented letters and CJK, unlike the old
        # ASCII-only pattern which turned "São Paulo" into "S", "o", "Paulo".
        words = [w for w in re.findall(r"\w+", question, flags=re.UNICODE) if w]
        # Single-letter enum codes quoted in the question ('A', 'F') are kept even
        # though bare single letters are usually noise.
        quoted_singles = {
            q.strip()
            for q in re.findall(r"['\"]([^'\"]{1,2})['\"]", question)
            if q.strip()
        }
        ngrams: list[str] = []
        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if n == 1 and phrase.lower() in _UNIGRAM_STOPWORDS and phrase not in quoted_singles:
                    continue
                if n == 1 and len(phrase) == 1 and phrase.lower() in {"a", "i"} and phrase not in quoted_singles:
                    continue
                ngrams.append(phrase)

        # Multi-word phrases carry the most grounding signal, but unigrams must never
        # be squeezed out by the phrase budget: on long questions the 4/3/2-grams alone
        # exceeded the old 36-phrase cap, so single-token values (e.g. a customer name
        # appearing late in the question) silently stopped matching.
        multi_grams = [g for g in ngrams if " " in g][:24]
        single_grams = [g for g in ngrams if " " not in g][:24]
        probe_phrases = multi_grams + single_grams

        # Phrases ready for FTS: tokenize each phrase and AND its tokens.
        phrase_tokens: list[tuple[str, list[str]]] = []
        for phrase in probe_phrases:
            cleaned = _clean_fts_token(phrase)
            tokens = [t for t in cleaned.split() if t]
            if tokens:
                phrase_tokens.append((phrase, tokens))
        if not phrase_tokens:
            return []

        seen_hits: dict[tuple[str, str, str], ValueMatch] = {}
        conn = self._ro_conn()
        # OR-batched FTS rounds: one MATCH per batch of phrases instead of one
        # round-trip per phrase (previously up to 48 serial queries per question).
        batch_size = 8
        for start in range(0, len(phrase_tokens), batch_size):
            batch = phrase_tokens[start : start + batch_size]
            or_query = " OR ".join(
                "(" + " AND ".join(f'"{t}"' for t in toks) + ")" for _, toks in batch
            )
            try:
                rows = conn.execute(
                    "SELECT table_name, column_name, raw_value, bm25(cell_fts) AS rank "
                    "FROM cell_fts WHERE cell_fts MATCH ? ORDER BY rank LIMIT ?",
                    (or_query, 4 * len(batch)),
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for tbl, col, raw_val, bm25_rank in rows:
                row_tokens = set(_norm_cell_value(str(raw_val)).split())
                for phrase, tokens in batch:
                    if not all(t.lower() in row_tokens for t in tokens):
                        # The OR round returned this row for another phrase in the
                        # batch; only count it for phrases it actually matches.
                        continue
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
