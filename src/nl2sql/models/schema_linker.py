from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nl2sql.db.introspect import SchemaBundle, TableInfo
from nl2sql.db.value_index import ValueMatch


def _load_backbone(model_id_or_path: str) -> nn.Module:
    """Load an AutoModel from a hub id / weights dir, or rebuild from a config-only artifact dir."""
    p = Path(model_id_or_path)
    weight_files = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    if p.is_dir() and (p / "config.json").exists() and not any((p / n).exists() for n in weight_files):
        return AutoModel.from_config(AutoConfig.from_pretrained(p))
    return AutoModel.from_pretrained(model_id_or_path)


class ModernBertSchemaLinker(nn.Module):
    """Long-context (up to 8192 tokens) schema linker scoring [TBL] and [COL] nodes in one pass."""

    TBL_TOKEN = "[TBL]"
    COL_TOKEN = "[COL]"

    def __init__(self, model_id_or_path: str = "answerdotai/ModernBERT-base", pos_weight: float = 3.0) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.encoder = _load_backbone(model_id_or_path)
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": [self.TBL_TOKEN, self.COL_TOKEN]})
        if added > 0:
            self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.tbl_token_id = self.tokenizer.convert_tokens_to_ids(self.TBL_TOKEN)
        self.col_token_id = self.tokenizer.convert_tokens_to_ids(self.COL_TOKEN)
        hidden = self.encoder.config.hidden_size
        self.table_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.column_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def format_inputs(
        self,
        question: str,
        tables: list[TableInfo],
        value_matches: list[ValueMatch],
        evidence: str = "",
    ) -> tuple[str, list[str], list[str]]:
        hint = f" | Hint: {evidence}" if evidence.strip() else ""
        val_str = "; ".join(f"{v.table_name}.{v.column_name}={v.value}" for v in value_matches[:5])
        val_part = f" | Values: {val_str}" if val_str else ""
        tbl_names: list[str] = []
        col_names: list[str] = []
        chunks = [f"Question: {question}{hint}{val_part}"]
        for t in tables:
            tbl_names.append(t.name)
            c_parts = []
            for c in t.columns:
                col_names.append(f"{t.name}.{c.name}")
                samp = f" ({', '.join(c.sample_values[:2])})" if c.sample_values else ""
                c_parts.append(f"{self.COL_TOKEN} {c.name}:{c.data_type}{samp}")
            chunks.append(f"{self.TBL_TOKEN} {t.name}: " + " ".join(c_parts))
        return "\n".join(chunks), tbl_names, col_names

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        table_labels: Optional[list[torch.Tensor]] = None,
        column_labels: Optional[list[torch.Tensor]] = None,
    ) -> dict[str, Any]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        bsz = input_ids.size(0)
        tbl_logits_list: list[torch.Tensor] = []
        col_logits_list: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=input_ids.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(input_ids.device))
        count = 0

        for i in range(bsz):
            t_pos = (input_ids[i] == self.tbl_token_id).nonzero(as_tuple=True)[0]
            c_pos = (input_ids[i] == self.col_token_id).nonzero(as_tuple=True)[0]
            t_logits = self.table_head(hidden[i, t_pos]).squeeze(-1) if len(t_pos) > 0 else torch.empty(0, device=input_ids.device)
            c_logits = self.column_head(hidden[i, c_pos]).squeeze(-1) if len(c_pos) > 0 else torch.empty(0, device=input_ids.device)
            tbl_logits_list.append(t_logits)
            col_logits_list.append(c_logits)

            if table_labels is not None and len(t_pos) > 0:
                n_t = min(len(t_logits), len(table_labels[i]))
                total_loss = total_loss + loss_fn(t_logits[:n_t], table_labels[i][:n_t].to(input_ids.device))
                count += 1
            if column_labels is not None and len(c_pos) > 0:
                n_c = min(len(c_logits), len(column_labels[i]))
                total_loss = total_loss + loss_fn(c_logits[:n_c], column_labels[i][:n_c].to(input_ids.device))
                count += 1

        return {
            "loss": total_loss / max(count, 1),
            "table_logits": tbl_logits_list,
            "column_logits": col_logits_list,
        }

    @torch.no_grad()
    def prune_schema(
        self,
        question: str,
        bundle: SchemaBundle,
        candidate_tables: list[TableInfo],
        value_matches: list[ValueMatch],
        evidence: str = "",
        threshold: float = 0.30,
        min_tables: int = 2,
        max_tables: int = 8,
        max_length: int = 4096,
    ) -> list[TableInfo]:
        self.eval()
        device = next(self.parameters()).device
        text_in, tbl_names, col_names = self.format_inputs(question, candidate_tables, value_matches, evidence)
        enc = self.tokenizer(text_in, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        out = self.forward(enc["input_ids"], enc["attention_mask"])

        t_probs = torch.sigmoid(out["table_logits"][0]).cpu().tolist() if len(out["table_logits"][0]) else []
        c_probs = torch.sigmoid(out["column_logits"][0]).cpu().tolist() if len(out["column_logits"][0]) else []
        tbl_score_map = {tbl_names[i]: float(t_probs[i]) for i in range(min(len(tbl_names), len(t_probs)))}

        # Boost tables whose columns or grounded values scored strongly
        for i, full_col in enumerate(col_names[: len(c_probs)]):
            tname = full_col.split(".", 1)[0]
            tbl_score_map[tname] = max(tbl_score_map.get(tname, 0.0), float(c_probs[i]))
        for vm in value_matches:
            tbl_score_map[vm.table_name] = max(tbl_score_map.get(vm.table_name, 0.0), 0.75)

        ranked = sorted(candidate_tables, key=lambda t: tbl_score_map.get(t.name, 0.0), reverse=True)
        selected = [t for t in ranked if tbl_score_map.get(t.name, 0.0) >= threshold][:max_tables]
        if len(selected) < min(min_tables, len(ranked)):
            selected = ranked[: min(min_tables, len(ranked))]

        # Ensure Hypergraph FK bridge connectivity between selected tables
        selected_names = {t.name for t in selected}
        if len(selected_names) > 1 and len(selected_names) < max_tables:
            adj: dict[str, set[str]] = {}
            for e in bundle.join_edges:
                adj.setdefault(e.left_table, set()).add(e.right_table)
                adj.setdefault(e.right_table, set()).add(e.left_table)
            anchor = selected[0].name
            for target in [t.name for t in selected[1:]]:
                if len(selected_names) >= max_tables:
                    break
                q = deque([(anchor, [anchor])])
                visited = {anchor}
                while q:
                    curr, path = q.popleft()
                    if curr == target:
                        for node in path:
                            if node not in selected_names and len(selected_names) < max_tables:
                                selected_names.add(node)
                        break
                    for nxt in adj.get(curr, set()):
                        if nxt not in visited:
                            visited.add(nxt)
                            q.append((nxt, path + [nxt]))

        return [t for t in candidate_tables if t.name in selected_names]

    def save_pretrained(self, save_dir: str | Path) -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(p)
        # Persist the encoder config so AutoModel.from_pretrained can rebuild the backbone
        self.encoder.config.save_pretrained(p)
        torch.save(self.state_dict(), p / "linker_weights.pt")
        meta = {"base_model": self.encoder.config._name_or_path, "pos_weight": float(self.pos_weight.item())}
        import json
        (p / "linker_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def from_pretrained_dir(cls, load_dir: str | Path, map_location: str = "cpu") -> "ModernBertSchemaLinker":
        import json
        p = Path(load_dir)
        meta = json.loads((p / "linker_meta.json").read_text(encoding="utf-8"))
        model = cls(model_id_or_path=str(p), pos_weight=meta["pos_weight"])
        model.load_state_dict(torch.load(p / "linker_weights.pt", map_location=map_location))
        return model
