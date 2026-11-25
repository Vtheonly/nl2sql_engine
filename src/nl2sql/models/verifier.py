from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from nl2sql.models.schema_linker import _load_backbone


class ModernBertVerifier(nn.Module):
    """Selective execution-guided correctness scorer (calibrated probability + temperature scalar)."""

    def __init__(self, model_id_or_path: str = "answerdotai/ModernBERT-base") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.encoder = _load_backbone(model_id_or_path)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.register_buffer("temperature", torch.tensor([1.0], dtype=torch.float32))

    @staticmethod
    def format_pair(question: str, schema_text: str, candidate_sql: str, exec_summary: str) -> str:
        return (
            f"Question: {question}\n"
            f"Candidate SQL: {candidate_sql}\n"
            f"Execution Summary: {exec_summary}\n"
            f"{schema_text}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        cls_emb = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.classifier(cls_emb).squeeze(-1)
        out: dict[str, Any] = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return out

    @torch.no_grad()
    def score_candidates(
        self,
        question: str,
        schema_text: str,
        candidates_with_exec: list[tuple[str, str]],
        max_length: int = 1024,
    ) -> list[float]:
        if not candidates_with_exec:
            return []
        self.eval()
        device = next(self.parameters()).device
        texts = [self.format_pair(question, schema_text, sql, summary) for sql, summary in candidates_with_exec]
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        logits = self.forward(enc["input_ids"], enc["attention_mask"])["logits"]
        probs = torch.sigmoid(logits / self.temperature.clamp(min=0.05).to(logits.device))
        return [float(p) for p in probs.cpu().tolist()]

    def save_pretrained(self, save_dir: str | Path) -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(p)
        # Persist the encoder config so AutoModel.from_pretrained can rebuild the backbone
        self.encoder.config.save_pretrained(p)
        torch.save(self.state_dict(), p / "verifier_weights.pt")
        (p / "verifier_meta.json").write_text(
            json.dumps({"temperature": float(self.temperature.item())}), encoding="utf-8"
        )

    @classmethod
    def from_pretrained_dir(cls, load_dir: str | Path, map_location: str = "cpu") -> "ModernBertVerifier":
        p = Path(load_dir)
        obj = cls(model_id_or_path=str(p))
        obj.load_state_dict(torch.load(p / "verifier_weights.pt", map_location=map_location))
        return obj
