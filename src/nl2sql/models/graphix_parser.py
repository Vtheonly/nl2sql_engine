from __future__ import annotations

import json
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


def build_sqlite_schema_ebnf(tables: list[TableInfo], value_matches: list[ValueMatch]) -> str:
    """Construct compact EBNF constraining SQL tokens to valid syntax and linked schema symbols."""
    tbl_alts = " | ".join(f'"{t.name}"' for t in tables) or '"dual"'
    col_names = sorted({c.name for t in tables for c in t.columns})
    qual_cols = sorted({f"{t.name}.{c.name}" for t in tables for c in t.columns})
    col_alts = " | ".join(f'"{c}"' for c in (qual_cols + col_names)) or '"*"'
    lit_vals = sorted({v.value.replace('"', '\\"') for v in value_matches if "'" not in v.value})
    extra_strs = " | ".join(f'"\'{v}\'"' for v in lit_vals)

    # Body-only string rule (inlined into `literal` below); extra grounded literals are allowed verbatim
    string_rule = r"""("'" [^'\n\r]* "'")"""
    if extra_strs:
        string_rule += f" | {extra_strs}"

    return rf"""
root ::= ws? query ws?
query ::= with_clause? select_core (ws set_op ws select_core)* order_clause? limit_clause?
with_clause ::= "WITH" ws cte_item ("," ws cte_item)* ws
cte_item ::= ident ws "AS" ws "(" ws select_core ws ")"
set_op ::= "UNION" (ws "ALL")? | "INTERSECT" | "EXCEPT"
select_core ::= "SELECT" ws ("DISTINCT" ws)? select_list ws from_clause where_clause? group_clause? having_clause?
select_list ::= select_item ("," ws select_item)*
select_item ::= ("*" | expr) (ws "AS" ws ident)?
from_clause ::= "FROM" ws table_ref (ws join_clause)*
table_ref ::= (table_name | "(" ws select_core ws ")") (ws ("AS" ws)? ident)?
join_clause ::= (("INNER" | "LEFT" | "CROSS") ws)? "JOIN" ws table_ref (ws "ON" ws cond)?
where_clause ::= ws "WHERE" ws cond
group_clause ::= ws "GROUP BY" ws expr ("," ws expr)*
having_clause ::= ws "HAVING" ws cond
order_clause ::= ws "ORDER BY" ws order_item ("," ws order_item)*
order_item ::= expr (ws ("ASC" | "DESC"))?
limit_clause ::= ws "LIMIT" ws [0-9]+
cond ::= cond_atom (ws ("AND" | "OR") ws cond_atom)*
cond_atom ::= ("NOT" ws)? (
    expr ws comp_op ws expr
    | expr ws ("NOT" ws)? "IN" ws "(" ws (select_core | expr ("," ws expr)*) ws ")"
    | expr ws ("NOT" ws)? "BETWEEN" ws expr ws "AND" ws expr
    | expr ws ("NOT" ws)? "LIKE" ws expr
    | expr ws "IS" ws ("NOT" ws)? "NULL"
    | "EXISTS" ws "(" ws select_core ws ")"
    | "(" ws cond ws ")"
)
comp_op ::= "=" | "!=" | "<>" | "<" | "<=" | ">" | ">="
expr ::= term (ws ("+" | "-" | "*" | "/" | "||") ws term)*
term ::= agg_call | scalar_call | case_expr | col_ref | literal | "(" ws (expr | select_core) ws ")"
agg_call ::= ("COUNT" | "SUM" | "AVG" | "MIN" | "MAX") "(" ("DISTINCT" ws)? ("*" | expr) ")"
scalar_call ::= ("ABS" | "ROUND" | "CAST" | "COALESCE" | "IFNULL" | "IIF" | "STRFTIME" | "SUBSTR" | "LENGTH" | "LOWER" | "UPPER" | "JULIANDAY") "(" expr ("," ws (expr | "AS" ws [A-Z]+))* ")"
case_expr ::= "CASE" (ws "WHEN" ws cond ws "THEN" ws expr)+ (ws "ELSE" ws expr)? ws "END"
literal ::= "-"? [0-9]+ ("." [0-9]+)? | {string_rule} | "NULL"
table_name ::= {tbl_alts}
col_ref ::= {col_alts} | ident "." (ident | "*")
ident ::= [a-zA-Z_][a-zA-Z0-9_]*
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
    ) -> None:
        import xgrammar as xgr

        self._xgr = xgr
        self.compiled_grammar = compiled_grammar
        self.vocab_size = vocab_size
        self.start_len = start_len
        self.eos_token_id = eos_token_id
        self.matchers: list[Any] = []
        self._dead_rows: set[int] = set()
        self.bitmask: Optional[torch.Tensor] = None

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
            if i in self._dead_rows or self.matchers[i].is_terminated():
                continue
            try:
                self.matchers[i].fill_next_token_bitmask(self.bitmask, i)
            except Exception:
                self._dead_rows.add(i)

        # Robustness guard: a fully-masked row (terminated matcher, dead row, or a vocab with no
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
        word, bit = divmod(self.eos_token_id, 32)
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
            {"additional_special_tokens": ["[TBL]", "[COL]", "[PK]", "[FK]", "[VAL]"]}
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
    def build_relation_matrix(
        input_ids: torch.Tensor,
        tokenizer: Any,
        bundles: list[SchemaBundle],
        tables_list: list[list[TableInfo]],
        values_list: list[list[ValueMatch]],
    ) -> torch.Tensor:
        """Build lightweight (B, L, L) token-level schema-graph edge matrix."""
        bsz, seq_len = input_ids.shape
        rel = torch.zeros((bsz, seq_len, seq_len), dtype=torch.long)
        tbl_tok = "[TBL]"
        col_tok = "[COL]"

        for b in range(bsz):
            toks = tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
            t_objs = {t.name.lower(): t for t in tables_list[b]}
            cur_tbl: Optional[str] = None
            token_table: list[Optional[str]] = [None] * seq_len
            q_end = seq_len

            for idx, tok in enumerate(toks):
                clean = tok.replace("Ġ", "").replace("▁", "").lower().rstrip(":")
                if q_end == seq_len and idx > 0 and clean.startswith("schema") and "schema" not in clean[6:]:
                    # Subword tokenizers split "Schema:" across tokens; anchor on its first piece
                    q_end = idx
                if tok == tbl_tok and idx + 1 < seq_len:
                    nxt = toks[idx + 1].replace("Ġ", "").replace("▁", "").lower().rstrip(":")
                    cur_tbl = nxt if nxt in t_objs else None
                if idx >= q_end and cur_tbl:
                    token_table[idx] = cur_tbl

            # Mark SAME_TABLE spans
            for i in range(q_end, seq_len):
                if not token_table[i]:
                    continue
                for j in range(q_end, seq_len):
                    if token_table[i] == token_table[j]:
                        rel[b, i, j] = REL_SAME_TABLE

            # Mark FK_EDGE spans
            fk_pairs = {(e.left_table.lower(), e.right_table.lower()) for e in bundles[b].join_edges}
            fk_pairs |= {(r, l) for l, r in fk_pairs}
            for i in range(q_end, seq_len):
                ti = token_table[i]
                if not ti:
                    continue
                for j in range(q_end, seq_len):
                    tj = token_table[j]
                    if tj and (ti, tj) in fk_pairs:
                        rel[b, i, j] = REL_FK_EDGE

            # Mark QUESTION_EXACT token matches
            q_words: dict[str, list[int]] = {}
            for i in range(q_end):
                w = re.sub(r"[^a-z0-9]", "", toks[i].replace("Ġ", "").replace("▁", "").lower())
                if len(w) >= 3:
                    q_words.setdefault(w, []).append(i)
            for j in range(q_end, seq_len):
                w = re.sub(r"[^a-z0-9]", "", toks[j].replace("Ġ", "").replace("▁", "").lower())
                if w in q_words:
                    for qi in q_words[w]:
                        rel[b, qi, j] = REL_QUESTION_EXACT
                        rel[b, j, qi] = REL_QUESTION_EXACT

            # Mark VALUE_MATCH spans
            val_words = {
                re.sub(r"[^a-z0-9]", "", v.value.lower())
                for v in values_list[b]
                if len(v.value) >= 2
            }
            for i in range(q_end):
                w = re.sub(r"[^a-z0-9]", "", toks[i].replace("Ġ", "").replace("▁", "").lower())
                if w in val_words:
                    for j in range(q_end, seq_len):
                        if token_table[j]:
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

        expanded_enc = BaseModelOutput(
            last_hidden_state=enc_out.last_hidden_state.repeat_interleave(num_candidates, dim=0)
        )
        expanded_mask = enc["attention_mask"].repeat_interleave(num_candidates, dim=0)

        logits_processors = []
        if use_xgrammar and self._xgr_available and self._xgr_compiler is not None:
            try:
                ebnf = build_sqlite_schema_ebnf(selected_tables, value_matches)
                # GrammarCompiler holds an internal cache shared across threads; serialize
                # compilation together with the bias critical section to stay safe.
                with self._bias_lock:
                    compiled = self._xgr_compiler.compile_grammar(ebnf)
                eos_id = self.model.config.eos_token_id
                if eos_id is None:
                    eos_id = self.tokenizer.eos_token_id
                logits_processors.append(
                    Seq2SeqXGrammarLogitsProcessor(
                        compiled,
                        vocab_size=self.model.config.vocab_size,
                        start_len=1,
                        eos_token_id=eos_id,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to compile schema EBNF (%s); falling back to unconstrained.", exc)

        gen_kwargs: dict[str, Any] = {
            "encoder_outputs": expanded_enc,
            "attention_mask": expanded_mask,
            "max_new_tokens": max_target_len,
            "num_beams": 1,
            "logits_processor": logits_processors,
        }
        if num_candidates > 1:
            # Row 0 greedy-like low temp; diverse sampling over batch rows avoids xgrammar beam bug
            gen_kwargs.update({"do_sample": True, "temperature": 0.45, "top_p": 0.92})
        else:
            gen_kwargs.update({"do_sample": False})

        outputs = self.model.generate(**gen_kwargs)
        decoded = [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
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
            obj.rel_bias_emb.load_state_dict(torch.load(rb, map_location=map_location))
        return obj
