from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, LogitsProcessor, T5ForConditionalGeneration

from nl2sql.db.introspect import SchemaBundle, TableInfo
from nl2sql.db.value_index import ValueMatch
from nl2sql.utils.logging import get_logger

logger = get_logger(__name__)

# Graphix relational edge types for schema-aware encoder self-attention
REL_NONE = 0
REL_SAME_TABLE = 1
REL_FK_EDGE = 2
REL_VALUE_MATCH = 3
REL_QUESTION_EXACT = 4
NUM_REL_TYPES = 5

# Unambiguous schema-section marker. Prompt text is free-form (and may legitimately
# contain the word "schema"), so the question/schema boundary must never be inferred
# from a plain English word.
SCHEMA_TOKEN = "[SCHEMA]"
TBL_TOKEN = "[TBL]"
COL_TOKEN = "[COL]"

# Tokens that terminate a column-name span in the serialized schema prompt. The
# prompt renders columns as ``[COL] <name> <TYPE> [PK] ex=(...) -- description``.
_TYPE_TOKENS = {
    "integer", "int", "text", "real", "blob", "numeric", "varchar", "char",
    "string", "boolean", "bool", "date", "datetime", "timestamp", "float",
    "double", "decimal", "primary", "key", "not", "null", "references",
}


def _clean_token(tok: Any) -> str:
    """Normalize a sentencepiece/BPE surface form for structural comparisons."""
    return str(tok).replace("Ġ", "").replace("▁", "").lower()


def _norm_ident(text: str) -> str:
    """Case/space/punctuation-insensitive identifier key (handles quoted names)."""
    return re.sub(r"[^0-9a-z_]", "", str(text).lower())


def _norm_word(text: str) -> str:
    """Unicode-aware word key used for question/value token matching."""
    return re.sub(r"[^\w]", "", str(text).lower(), flags=re.UNICODE)


def _ebnf_literal(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ident_alternatives(name: str) -> list[str]:
    """Unquoted, double-quoted and backtick-quoted spellings of one identifier."""
    return [
        _ebnf_literal(name),
        _ebnf_literal(f'"{name}"'),
        _ebnf_literal(f"`{name}`"),
    ]


def build_sqlite_schema_ebnf(tables: list[TableInfo], value_matches: list[ValueMatch]) -> str:
    """Construct compact EBNF constraining SQL tokens to valid syntax and linked schema symbols.

    The grammar is deliberately grounded: every table/column reference must resolve to
    a symbol that actually exists in the linked schema. Alias-qualified references are
    allowed, but only when the column part is a real column (``c.name``), never an
    arbitrary ``foo.bar`` pair.
    """
    tbl_alts = " | ".join(alt for t in tables for alt in _ident_alternatives(t.name)) or '"dual"'
    col_names = sorted({c.name for t in tables for c in t.columns})
    col_only_alts = " | ".join(alt for c in col_names for alt in _ident_alternatives(c)) or '"*"'

    qual_alts: list[str] = []
    for t in tables:
        for c in t.columns:
            # Unquoted, per-identifier quoted, and backtick spellings of a qualified ref.
            qual_alts.append(_ebnf_literal(f"{t.name}.{c.name}"))
            qual_alts.append(_ebnf_literal(f'"{t.name}"."{c.name}"'))
            qual_alts.append(_ebnf_literal(f"`{t.name}`.`{c.name}`"))
            qual_alts.append(_ebnf_literal(f"{t.name}.\"{c.name}\""))
            qual_alts.append(_ebnf_literal(f'"{t.name}".{c.name}'))
    all_col_alts: list[str] = list(qual_alts)
    for c in col_names:
        all_col_alts.extend(_ident_alternatives(c))
    col_alts = " | ".join(dict.fromkeys(all_col_alts)) or '"*"'

    lit_vals = sorted({v.value.replace('"', '\\"') for v in value_matches if "'" not in v.value})
    extra_strs = " | ".join(f'"\'{v}\'"' for v in lit_vals)

    # Body-only string rule (inlined into `literal` below); extra grounded literals are allowed verbatim
    string_rule = r"""("'" [^'\n\r]* "'")"""
    if extra_strs:
        string_rule += f" | {extra_strs}"

    return rf"""
root ::= ws? query ws?
query ::= with_clause? select_core (ws set_op ws select_core)* order_clause? limit_clause?
with_clause ::= "WITH" ws ("RECURSIVE" ws)? cte_item ("," ws cte_item)* ws
cte_item ::= ident ws "AS" ws "(" ws? select_core ws? ")"
set_op ::= "UNION" (ws "ALL")? | "INTERSECT" | "EXCEPT"
select_core ::= "SELECT" ws ("DISTINCT" ws)? select_list (ws from_clause)? where_clause? group_clause? having_clause?
select_list ::= select_item ("," ws select_item)*
select_item ::= ("*" | table_name "." "*" | expr) (ws "AS" ws ident)?
from_clause ::= "FROM" ws table_ref (ws join_clause)*
table_ref ::= (table_name | "(" ws? select_core ws? ")") (ws ("AS" ws)? ident)?
join_clause ::= join_type? "JOIN" ws table_ref join_condition?
join_type ::= ("INNER" | "LEFT" (ws "OUTER")? | "RIGHT" (ws "OUTER")? | "FULL" (ws "OUTER")? | "CROSS") ws
join_condition ::= (ws "ON" ws cond) | (ws "USING" ws "(" ws? ident ("," ws ident)* ws? ")")
where_clause ::= ws "WHERE" ws cond
group_clause ::= ws "GROUP BY" ws expr ("," ws expr)*
having_clause ::= ws "HAVING" ws cond
order_clause ::= ws "ORDER BY" ws order_item ("," ws order_item)*
order_item ::= expr (ws ("ASC" | "DESC"))?
limit_clause ::= ws "LIMIT" ws [0-9]+ (ws "," ws [0-9]+ | (ws "OFFSET" ws [0-9]+)?)?
cond ::= cond_atom (ws ("AND" | "OR") ws cond_atom)*
cond_atom ::= ("NOT" ws)? (
    expr ws comp_op ws expr
    | expr ws ("NOT" ws)? "IN" ws "(" ws? (select_core | expr ("," ws expr)*) ws? ")"
    | expr ws ("NOT" ws)? "BETWEEN" ws expr ws "AND" ws expr
    | expr ws ("NOT" ws)? "LIKE" ws expr
    | expr ws "IS" ws ("NOT" ws)? "NULL"
    | "EXISTS" ws "(" ws? select_core ws? ")"
    | "(" ws? cond ws? ")"
)
comp_op ::= "=" | "!=" | "<>" | "<" | "<=" | ">" | ">="
expr ::= term (ws ("+" | "-" | "*" | "/" | "||") ws term)*
term ::= agg_call | scalar_call | case_expr | col_ref | literal | "(" ws? (expr | select_core) ws? ")"
agg_call ::= ("COUNT" | "SUM" | "AVG" | "MIN" | "MAX") "(" ("DISTINCT" ws)? ("*" | expr) ")"
scalar_call ::= ("ABS" | "ROUND" | "CAST" | "COALESCE" | "IFNULL" | "IIF" | "STRFTIME" | "SUBSTR" | "LENGTH" | "LOWER" | "UPPER" | "JULIANDAY") "(" expr (ws "AS" ws type_name | ("," ws expr)*) ")"
type_name ::= [A-Za-z]+ (ws "(" ws [0-9]+ ("," ws [0-9]+)? ws ")")?
case_expr ::= "CASE" (ws expr)? (ws "WHEN" ws cond ws "THEN" ws expr)+ (ws "ELSE" ws expr)? ws "END"
literal ::= "-"? [0-9]+ ("." [0-9]+)? | {string_rule} | "NULL"
table_name ::= {tbl_alts}
col_ref ::= ({col_alts}) | ident "." ({col_only_alts})
ident ::= [a-zA-Z_][a-zA-Z0-9_]* | quoted_ident
quoted_ident ::= "\"" [^"\n\r]* "\"" | "`" [^`\n\r]* "`" | "[" [^\]\n\r]* "]"
ws ::= [ \t\n]+
"""


class Seq2SeqXGrammarLogitsProcessor(LogitsProcessor):
    """
    Encoder-decoder-safe XGrammar LogitsProcessor for T5/CodeT5+.
    Avoids xgrammar's stock num_beams > 1 bug by operating on independent batch/sample sequences
    where decoder_input_ids begins with 1 decoder_start_token_id.
    """

    def __init__(
        self,
        compiled_grammar: Any,
        vocab_size: int,
        start_len: int = 1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        import xgrammar as xgr

        self._xgr = xgr
        self.compiled_grammar = compiled_grammar
        self.vocab_size = vocab_size
        self.start_len = start_len
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.matchers: list[Any] = []
        self._dead_rows: set[int] = set()
        self.bitmask: Optional[torch.Tensor] = None

    def _reset_row_to_eos(self, row: int) -> None:
        """Allow only EOS (and PAD) on a terminated row.

        The bitmask is allocated once per generation batch. Without an explicit reset,
        a terminated row keeps whatever tokens were legal at step ``t-1``, so the
        ``(bitmask == 0).all(dim=1)`` EOS-fallback guard never fires and generation
        can continue past a complete statement.
        """
        assert self.bitmask is not None
        self.bitmask[row].zero_()
        allowed = [tid for tid in (self.eos_token_id, self.pad_token_id) if tid is not None and 0 <= tid < self.vocab_size]
        if not allowed:
            # No stop token configured: leave the row unconstrained rather than
            # producing an all -inf row (which yields NaN sampling).
            self.bitmask[row].fill_(-1)
            return
        for tid in allowed:
            word, bit = divmod(int(tid), 32)
            self.bitmask[row, word] |= 1 << bit

    def _reset_row_unconstrained(self, row: int) -> None:
        """Dead rows must not inherit stale grammar bits from a previous step."""
        assert self.bitmask is not None
        self.bitmask[row].fill_(-1)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        if not self.matchers:
            self.matchers = [self._xgr.GrammarMatcher(self.compiled_grammar) for _ in range(batch_size)]
            self.bitmask = self._xgr.allocate_token_bitmask(batch_size, self.vocab_size)
        elif len(self.matchers) != batch_size:
            raise ValueError("Seq2SeqXGrammarLogitsProcessor requires num_beams=1 (use do_sample/num_return_sequences).")

        cur_len = input_ids.shape[1]
        if cur_len > self.start_len:
            for i in range(batch_size):
                if i in self._dead_rows or self.matchers[i].is_terminated():
                    continue
                try:
                    self.matchers[i].accept_token(int(input_ids[i, -1].item()))
                except Exception:
                    # A non-conforming token slipped through (e.g. forced by another processor):
                    # mark the row dead instead of crashing the whole generation batch.
                    self._dead_rows.add(i)

        for i in range(batch_size):
            if i in self._dead_rows:
                self._reset_row_unconstrained(i)
                continue
            if self.matchers[i].is_terminated():
                self._reset_row_to_eos(i)
                continue
            try:
                self.matchers[i].fill_next_token_bitmask(self.bitmask, i)
            except Exception:
                self._dead_rows.add(i)
                self._reset_row_unconstrained(i)

        # Robustness guard: a fully-masked row (dead row, terminated row, or a vocab with no
        # grammar-reachable token) must never produce all -inf logits, which yields NaN sampling.
        self._ensure_eos_fallback(batch_size)

        bitmask_dev = self.bitmask.to(scores.device)
        self._xgr.apply_token_bitmask_inplace(scores, bitmask_dev)
        return scores

    def _ensure_eos_fallback(self, batch_size: int) -> None:
        if self.eos_token_id is None or not (0 <= self.eos_token_id < self.vocab_size):
            return
        empty_rows = (self.bitmask == 0).all(dim=1).nonzero(as_tuple=True)[0]
        if len(empty_rows) == 0:
            return
        word, bit = divmod(int(self.eos_token_id), 32)
        for i in empty_rows.tolist():
            if i < batch_size:
                self.bitmask[i, word] |= 1 << bit


class GraphixCodeT5Parser(nn.Module):
    """CodeT5+-220M augmented with Graphix relation-aware encoder attention biases."""

    def __init__(self, model_id_or_path: str = "Salesforce/codet5p-220m") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id_or_path)
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [TBL_TOKEN, COL_TOKEN, "[PK]", "[FK]", "[VAL]", SCHEMA_TOKEN]}
        )
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        num_heads = self.model.config.num_heads
        self.rel_bias_emb = nn.Embedding(NUM_REL_TYPES, num_heads)
        nn.init.zeros_(self.rel_bias_emb.weight)
        self._active_rel_bias: Optional[torch.Tensor] = None
        # `self._active_rel_bias` is instance-wide shared state consulted by the encoder
        # attention hook. FastAPI serves requests from a threadpool, so concurrent
        # generate()/forward() calls would otherwise overwrite (or clear) each other's
        # bias tensor mid-forward and silently corrupt schema-aware attention.
        self._bias_lock = threading.Lock()
        self._install_graphix_hook()

        self._xgr_compiler = None
        self._xgr_available = False
        try:
            import xgrammar as xgr
            t_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer, vocab_size=self.model.config.vocab_size
            )
            self._xgr_compiler = xgr.GrammarCompiler(t_info, cache_enabled=True)
            self._xgr_available = True
        except Exception as exc:
            logger.warning("XGrammar unavailable (%s); decoding will run unconstrained.", exc)

    def _install_graphix_hook(self) -> None:
        attn_mod = self.model.encoder.block[0].layer[0].SelfAttention
        orig_compute_bias = attn_mod.compute_bias
        parent = self

        def wrapped_compute_bias(query_length: int, key_length: int, device=None, **kwargs):
            bias = orig_compute_bias(query_length, key_length, device=device, **kwargs)
            if (
                parent._active_rel_bias is not None
                and parent._active_rel_bias.shape[-2:] == (query_length, key_length)
            ):
                bias = bias + parent._active_rel_bias.to(device=bias.device, dtype=bias.dtype)
            return bias

        attn_mod.compute_bias = wrapped_compute_bias

    @staticmethod
    def _find_schema_anchor(toks: list[Any]) -> int:
        """Index of the first schema-section token, or len(toks) when absent.

        Prefers the explicit ``[SCHEMA]`` marker; falls back to the legacy ``Schema:``
        header for prompts produced before the marker existed.
        """
        for idx, tok in enumerate(toks):
            if str(tok) == SCHEMA_TOKEN or _clean_token(tok) == _clean_token(SCHEMA_TOKEN):
                return idx
        for idx, tok in enumerate(toks):
            clean = _clean_token(tok)
            if clean == "schema:":
                return idx
            if clean.startswith("schema") and not clean[6:].strip(":"):
                return idx
        return len(toks)

    @staticmethod
    def _parse_schema_owners(
        toks: list[Any],
        tables: list[TableInfo],
        q_end: int,
    ) -> tuple[list[Optional[str]], list[Optional[str]]]:
        """Map each schema token to its owning table/column (lowercase names).

        Names are matched by longest prefix against the known schema identifiers, so
        BPE multi-piece names (``order``/``_``/``items``) and names with spaces or
        punctuation all resolve, without depending on a ``:`` token surviving
        tokenization (WordLevel tokenizers map ``:`` to ``<unk>``).
        """
        seq_len = len(toks)
        owner_table: list[Optional[str]] = [None] * seq_len
        owner_col: list[Optional[str]] = [None] * seq_len

        # Normalized lookup maps tolerate quoted names, spaces and punctuation.
        t_by_norm: dict[str, TableInfo] = {}
        for t in tables:
            t_by_norm[_norm_ident(t.name)] = t
        col_norm_by_table: dict[str, dict[str, str]] = {}
        for t in tables:
            col_norm_by_table[_norm_ident(t.name)] = {
                _norm_ident(c.name): c.name.lower() for c in t.columns
            }

        cur_tbl: Optional[str] = None
        cur_tbl_norm: Optional[str] = None
        cur_col: Optional[str] = None

        def _is_marker(clean: str) -> bool:
            return clean == _clean_token(TBL_TOKEN) or clean == _clean_token(COL_TOKEN)

        idx = 0
        while idx < seq_len:
            tok = toks[idx]
            clean = _clean_token(tok)

            if clean == _clean_token(TBL_TOKEN):
                # Longest-prefix match of the joined token pieces against table names.
                j = idx + 1
                parts: list[str] = []
                best_end, best_tbl = -1, None
                while j < seq_len:
                    cj = _clean_token(toks[j])
                    if _is_marker(cj) or cj == ":" or cj.endswith(":"):
                        if cj != ":" and cj.endswith(":"):
                            parts.append(cj[:-1])
                            cand = _norm_ident("".join(parts))
                            if t_by_norm.get(cand) is not None:
                                best_end, best_tbl = j + 1, t_by_norm[cand]
                        break
                    parts.append(cj)
                    cand = _norm_ident("".join(parts))
                    if t_by_norm.get(cand) is not None:
                        best_end, best_tbl = j + 1, t_by_norm[cand]
                    j += 1
                if best_tbl is not None and best_end > 0:
                    cur_tbl = best_tbl.name.lower()
                    cur_tbl_norm = _norm_ident(cur_tbl)
                    cur_col = None
                    # Cover the full span including the trailing ':' separator token.
                    for k in range(idx, min(j + 1, seq_len)):
                        owner_table[k] = cur_tbl
                else:
                    cur_tbl = cur_tbl_norm = None
                    cur_col = None
                idx = j if j < seq_len else seq_len
                continue

            if clean == _clean_token(COL_TOKEN):
                # Longest-prefix match against the current table's column names.
                j = idx + 1
                parts = []
                best_end, best_col = -1, None
                cols = col_norm_by_table.get(cur_tbl_norm or "", {})
                while j < seq_len:
                    cj = _clean_token(toks[j])
                    if (
                        _is_marker(cj)
                        or (j > idx + 1 and (cj in _TYPE_TOKENS or not cj))
                        or cj.startswith("--")
                        or "=" in cj
                    ):
                        break
                    parts.append(cj)
                    cand = _norm_ident("".join(parts))
                    actual = cols.get(cand)
                    if actual is not None:
                        best_end, best_col = j + 1, actual
                    j += 1
                if best_col is not None and best_end > 0:
                    cur_col = best_col
                    for k in range(idx, best_end):
                        owner_table[k] = cur_tbl
                        owner_col[k] = cur_col
                idx = j if j < seq_len else seq_len
                continue

            if idx >= q_end and cur_tbl:
                owner_table[idx] = cur_tbl
                if cur_col:
                    owner_col[idx] = cur_col
            idx += 1

        return owner_table, owner_col

    @staticmethod
    def build_relation_matrix(
        input_ids: torch.Tensor,
        tokenizer: Any,
        bundles: list[SchemaBundle],
        tables_list: list[list[TableInfo]],
        values_list: list[list[ValueMatch]],
    ) -> torch.Tensor:
        """Build lightweight (B, L, L) token-level schema-graph edge matrix.

        The per-sample work is O(L) for span parsing plus a handful of tensor block
        fills, instead of the previous O(B * L^2) Python loops inside ``collate``.
        """
        bsz, seq_len = input_ids.shape
        rel = torch.zeros((bsz, seq_len, seq_len), dtype=torch.long)

        for b in range(bsz):
            toks = tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
            q_end = GraphixCodeT5Parser._find_schema_anchor(toks)
            owner_table, owner_col = GraphixCodeT5Parser._parse_schema_owners(
                toks, tables_list[b], q_end
            )

            # SAME_TABLE spans: fill each table's token block in one tensor operation.
            table_positions: dict[str, list[int]] = {}
            for i in range(q_end, seq_len):
                if owner_table[i]:
                    table_positions.setdefault(owner_table[i], []).append(i)
            for pos in table_positions.values():
                idx = torch.tensor(pos, dtype=torch.long)
                rel[b, idx[:, None], idx[None, :]] = REL_SAME_TABLE

            # FK_EDGE spans between tables connected in the schema graph.
            fk_pairs = {(e.left_table.lower(), e.right_table.lower()) for e in bundles[b].join_edges}
            fk_pairs |= {(r, l) for l, r in fk_pairs}
            for lt, rt in fk_pairs:
                lp = table_positions.get(lt)
                rp = table_positions.get(rt)
                if lp and rp:
                    li = torch.tensor(lp, dtype=torch.long)
                    ri = torch.tensor(rp, dtype=torch.long)
                    rel[b, li[:, None], ri[None, :]] = REL_FK_EDGE

            # QUESTION_EXACT token matches.
            q_words: dict[str, list[int]] = {}
            q_norm: list[str] = []
            for i in range(q_end):
                w = _norm_word(_clean_token(toks[i]))
                q_norm.append(w)
                if len(w) >= 3:
                    q_words.setdefault(w, []).append(i)
            for j in range(q_end, seq_len):
                w = _norm_word(_clean_token(toks[j]))
                if w in q_words:
                    for qi in q_words[w]:
                        rel[b, qi, j] = REL_QUESTION_EXACT
                        rel[b, j, qi] = REL_QUESTION_EXACT

            # VALUE_MATCH: connect a matched value only to the specific table/column
            # that produced it, and match multi-word values as token subsequences.
            for v in values_list[b]:
                phrase = [w for w in re.findall(r"\w+", v.value.lower(), flags=re.UNICODE) if w]
                if not phrase:
                    continue
                target = [
                    j
                    for j in range(q_end, seq_len)
                    if owner_table[j]
                    and _norm_ident(owner_table[j]) == _norm_ident(v.table_name)
                    and owner_col[j]
                    and _norm_ident(owner_col[j]) == _norm_ident(v.column_name)
                ]
                if not target:
                    target = [
                        j
                        for j in range(q_end, seq_len)
                        if owner_table[j] and _norm_ident(owner_table[j]) == _norm_ident(v.table_name)
                    ]
                if not target:
                    continue
                n = len(phrase)
                for i in range(0, max(0, len(q_norm) - n + 1)):
                    if q_norm[i : i + n] == phrase:
                        for j in target:
                            rel[b, i, j] = REL_VALUE_MATCH
        return rel

    def _compute_bias_tensor(self, rel_matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if rel_matrix is None:
            return None
        # (B, L, L) -> (B, L, L, H) -> (B, H, L, L)
        return self.rel_bias_emb(rel_matrix).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        rel_matrix: Optional[torch.Tensor] = None,
    ):
        bias = self._compute_bias_tensor(rel_matrix)
        with self._bias_lock:
            self._active_rel_bias = bias
            try:
                return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            finally:
                self._active_rel_bias = None

    def _make_logits_processor(self, compiled: Any, eos_id: Optional[int]) -> "Seq2SeqXGrammarLogitsProcessor":
        return Seq2SeqXGrammarLogitsProcessor(
            compiled,
            vocab_size=self.model.config.vocab_size,
            start_len=1,
            eos_token_id=eos_id,
            pad_token_id=self.model.config.pad_token_id,
        )

    @torch.no_grad()
    def generate_candidates(
        self,
        source_text: str,
        bundle: SchemaBundle,
        selected_tables: list[TableInfo],
        value_matches: list[ValueMatch],
        num_candidates: int = 4,
        max_source_len: int = 1024,
        max_target_len: int = 192,
        use_xgrammar: bool = True,
    ) -> list[str]:
        self.eval()
        num_candidates = max(1, int(num_candidates))
        device = next(self.parameters()).device
        enc = self.tokenizer(
            source_text, truncation=True, max_length=max_source_len, return_tensors="pt"
        ).to(device)
        rel_matrix = self.build_relation_matrix(
            enc["input_ids"].cpu(), self.tokenizer, [bundle], [selected_tables], [value_matches]
        ).to(device)

        # Precompute encoder hidden states once, then expand across K candidates.
        # The bias critical section (set -> encoder forward -> reset) runs under
        # `self._bias_lock` so concurrent requests cannot cross-contaminate attention biases.
        with self._bias_lock:
            self._active_rel_bias = self._compute_bias_tensor(rel_matrix)
            try:
                enc_out = self.model.get_encoder()(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
                )
            finally:
                self._active_rel_bias = None

        from transformers.modeling_outputs import BaseModelOutput

        compiled = None
        eos_id = self.model.config.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.eos_token_id
        if use_xgrammar and self._xgr_available and self._xgr_compiler is not None:
            try:
                ebnf = build_sqlite_schema_ebnf(selected_tables, value_matches)
                # GrammarCompiler holds an internal cache shared across threads; serialize
                # compilation together with the bias critical section to stay safe.
                with self._bias_lock:
                    compiled = self._xgr_compiler.compile_grammar(ebnf)
            except Exception as exc:
                logger.warning("Failed to compile schema EBNF (%s); falling back to unconstrained.", exc)
                compiled = None

        def _processors() -> list[Any]:
            return [self._make_logits_processor(compiled, eos_id)] if compiled is not None else []

        def _decode(outputs: torch.Tensor) -> list[str]:
            return [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]

        # Candidate 0 is always greedy: stochastic sampling must never displace the
        # highest-likelihood SQL. Candidates 1..K-1 add diversity.
        greedy_out = self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_out.last_hidden_state),
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_target_len,
            num_beams=1,
            do_sample=False,
            logits_processor=_processors(),
        )
        decoded = _decode(greedy_out)

        if num_candidates > 1:
            k = num_candidates - 1
            sampled_out = self.model.generate(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=enc_out.last_hidden_state.repeat_interleave(k, dim=0)
                ),
                attention_mask=enc["attention_mask"].repeat_interleave(k, dim=0),
                max_new_tokens=max_target_len,
                num_beams=1,
                do_sample=True,
                temperature=0.45,
                top_p=0.92,
                logits_processor=_processors(),
            )
            decoded.extend(_decode(sampled_out))

        # Preserve order while deduplicating
        unique: list[str] = []
        seen: set[str] = set()
        for sql in decoded:
            if sql and sql not in seen:
                seen.add(sql)
                unique.append(sql)
        return unique

    def save_pretrained(self, save_dir: str | Path) -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(p)
        self.model.save_pretrained(p)
        torch.save(self.rel_bias_emb.state_dict(), p / "graphix_rel_bias.pt")

    @classmethod
    def from_pretrained_dir(cls, load_dir: str | Path, map_location: str = "cpu") -> "GraphixCodeT5Parser":
        p = Path(load_dir)
        obj = cls(model_id_or_path=str(p))
        rb = p / "graphix_rel_bias.pt"
        if rb.exists():
            obj.rel_bias_emb.load_state_dict(
                torch.load(rb, map_location=map_location, weights_only=True)
            )
        return obj
