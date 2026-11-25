from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from nl2sql.config import EngineConfig
from nl2sql.datasets.prepare import (
    EmbedderDataset,
    ExampleRecord,
    LinkerDataset,
    ParserDataset,
    VerifierDataset,
    format_schema_for_parser,
)
from nl2sql.db.introspect import SchemaBundle
from nl2sql.db.value_index import ValueIndex
from nl2sql.evaluation.evaluator import execute_sqlite_readonly, results_equivalent
from nl2sql.models.graphix_parser import GraphixCodeT5Parser
from nl2sql.models.schema_linker import ModernBertSchemaLinker
from nl2sql.models.verifier import ModernBertVerifier
from nl2sql.training.checkpoint import CheckpointManager
from nl2sql.utils.logging import get_logger, get_resource_stats
from nl2sql.utils.repro import set_seed

logger = get_logger(__name__)


class NL2SQLTrainer:
    """Unified multi-stage trainer supporting embedder, linker, parser, and verifier."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt_mgr = CheckpointManager(cfg.training.output_dir, keep_last_k=cfg.training.keep_last_k_checkpoints)
        self.metrics_history: list[dict[str, Any]] = []

    def _amp_dtype(self) -> Optional[torch.dtype]:
        if self.device.type != "cuda" or self.cfg.training.mixed_precision == "no":
            return None
        if self.cfg.training.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _run_generic_loop(
        self,
        stage: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        forward_step_fn,
        save_fn,
    ) -> dict[str, Any]:
        model = model.to(self.device)
        tcfg = self.cfg.training
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
        total_steps = max(1, (len(train_loader) * tcfg.epochs) // tcfg.grad_accum_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * tcfg.warmup_ratio), num_training_steps=total_steps
        )
        amp_dtype = self._amp_dtype()
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        patience_left = tcfg.early_stopping_patience
        history: list[dict[str, Any]] = []

        resume_dir = self.ckpt_mgr.resolve_resume_path(stage, tcfg.resume_from)
        if resume_dir is not None:
            payload = self.ckpt_mgr.load(
                resume_dir, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=str(self.device)
            )
            start_epoch = int(payload["epoch"])
            global_step = int(payload["global_step"])
            best_val_loss = float(payload["best_metric"])
            history = list(payload.get("metrics_history", []))
            logger.info("Resumed stage='%s' from %s (epoch=%d, step=%d)", stage, resume_dir, start_epoch, global_step)

        for epoch in range(start_epoch, tcfg.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            n_batches = 0

            for step_idx, batch in enumerate(train_loader):
                with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    loss = forward_step_fn(model, batch, self.device)
                    loss_scaled = loss / tcfg.grad_accum_steps

                scaler.scale(loss_scaled).backward()
                running_loss += float(loss.item())
                n_batches += 1

                if (step_idx + 1) % tcfg.grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % tcfg.save_every_steps == 0:
                        self.ckpt_mgr.save(
                            stage=stage, model=model, optimizer=optimizer, scheduler=scheduler,
                            scaler=scaler, epoch=epoch, global_step=global_step,
                            best_metric=best_val_loss, config_dict=self.cfg.model_dump(mode="json"),
                            metrics_history=history, is_best=False,
                        )

            train_loss = running_loss / max(n_batches, 1)
            val_loss = train_loss
            if val_loader is not None and (epoch + 1) % tcfg.eval_every_epochs == 0:
                model.eval()
                v_sum, v_cnt = 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                            v_sum += float(forward_step_fn(model, batch, self.device).item())
                        v_cnt += 1
                val_loss = v_sum / max(v_cnt, 1)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_left = tcfg.early_stopping_patience
                save_fn(model, Path(tcfg.output_dir) / "artifacts" / stage)
            else:
                patience_left -= 1

            record = {
                "stage": stage,
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "lr": float(scheduler.get_last_lr()[0]),
                "is_best": is_best,
                "resources": get_resource_stats(),
            }
            history.append(record)
            logger.info("[%s] Epoch %d/%d | train=%.4f | val=%.4f | lr=%.2e", stage, epoch + 1, tcfg.epochs, train_loss, val_loss, record["lr"])

            self.ckpt_mgr.save(
                stage=stage, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch + 1, global_step=global_step,
                best_metric=best_val_loss, config_dict=self.cfg.model_dump(mode="json"),
                metrics_history=history, is_best=is_best,
            )
            if patience_left <= 0:
                logger.info("[%s] Early stopping triggered at epoch %d", stage, epoch + 1)
                break

        self.metrics_history.extend(history)
        return {"stage": stage, "best_val_loss": best_val_loss, "history": history, "artifact_dir": str(Path(tcfg.output_dir) / "artifacts" / stage)}

    def train_embedder(self, train_records: list[ExampleRecord], val_records: list[ExampleRecord], caches: dict[str, tuple[SchemaBundle, ValueIndex]]) -> dict[str, Any]:
        model_id = self.cfg.models.embedder_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        encoder = AutoModel.from_pretrained(model_id)
        train_ds = EmbedderDataset(train_records, caches)
        val_ds = EmbedderDataset(val_records, caches) if val_records else None

        def collate(batch):
            q_enc = tokenizer([b["query"] for b in batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
            p_enc = tokenizer([b["positive"] for b in batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
            return {"q": q_enc, "p": p_enc}

        def step(m, batch, dev):
            q_out = nn.functional.normalize(m(**{k: v.to(dev) for k, v in batch["q"].items()}).last_hidden_state[:, 0], dim=-1)
            p_out = nn.functional.normalize(m(**{k: v.to(dev) for k, v in batch["p"].items()}).last_hidden_state[:, 0], dim=-1)
            logits = (q_out @ p_out.T) * 20.0
            targets = torch.arange(logits.size(0), device=dev)
            return nn.functional.cross_entropy(logits, targets)

        def save_fn(m, out_dir: Path):
            out_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(out_dir)
            m.save_pretrained(out_dir)

        t_loader = DataLoader(train_ds, batch_size=max(16, self.cfg.training.batch_size * 2), shuffle=True, collate_fn=collate)
        v_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate) if val_ds and len(val_ds) > 0 else None
        return self._run_generic_loop("embedder", encoder, t_loader, v_loader, step, save_fn)

    def train_linker(self, train_records: list[ExampleRecord], val_records: list[ExampleRecord], caches: dict[str, tuple[SchemaBundle, ValueIndex]]) -> dict[str, Any]:
        model = ModernBertSchemaLinker(self.cfg.models.linker_model_id, pos_weight=self.cfg.models.linker_recall_pos_weight)
        train_ds = LinkerDataset(train_records, caches, perturb_prob=self.cfg.data.schema_perturbation_prob, seed=self.cfg.seed)
        val_ds = LinkerDataset(val_records, caches, perturb_prob=0.0, seed=self.cfg.seed) if val_records else None

        def collate(batch):
            texts, t_lbls, c_lbls = [], [], []
            for item in batch:
                txt, _, _ = model.format_inputs(item["question"], item["bundle"].tables, item["value_matches"], item["evidence"])
                texts.append(txt)
                t_lbls.append(torch.tensor([t["label"] for t in item["table_items"]], dtype=torch.float32))
                c_lbls.append(torch.tensor([c["label"] for c in item["col_items"]], dtype=torch.float32))
            enc = model.tokenizer(texts, padding=True, truncation=True, max_length=self.cfg.models.linker_max_length, return_tensors="pt")
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "table_labels": t_lbls, "column_labels": c_lbls}

        def step(m, batch, dev):
            return m(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["table_labels"], batch["column_labels"])["loss"]

        t_loader = DataLoader(train_ds, batch_size=max(2, self.cfg.training.batch_size // 2), shuffle=True, collate_fn=collate)
        v_loader = DataLoader(val_ds, batch_size=max(2, self.cfg.training.batch_size // 2), shuffle=False, collate_fn=collate) if val_ds and len(val_ds) > 0 else None
        return self._run_generic_loop("linker", model, t_loader, v_loader, step, lambda m, d: m.save_pretrained(d))

    def train_parser(self, train_records: list[ExampleRecord], val_records: list[ExampleRecord], caches: dict[str, tuple[SchemaBundle, ValueIndex]]) -> dict[str, Any]:
        model = GraphixCodeT5Parser(self.cfg.models.parser_model_id)
        train_ds = ParserDataset(train_records, caches, perturb_prob=self.cfg.data.schema_perturbation_prob, seed=self.cfg.seed)
        val_ds = ParserDataset(val_records, caches, perturb_prob=0.0, seed=self.cfg.seed) if val_records else None

        def collate(batch):
            src_enc = model.tokenizer([b["source_text"] for b in batch], padding=True, truncation=True, max_length=self.cfg.models.parser_max_source_length, return_tensors="pt")
            tgt_enc = model.tokenizer(text_target=[b["target_sql"] for b in batch], padding=True, truncation=True, max_length=self.cfg.models.parser_max_target_length, return_tensors="pt")
            labels = tgt_enc["input_ids"].clone()
            labels[labels == model.tokenizer.pad_token_id] = -100
            rel = GraphixCodeT5Parser.build_relation_matrix(
                src_enc["input_ids"], model.tokenizer,
                [b["bundle"] for b in batch], [b["selected_tables"] for b in batch], [b["value_matches"] for b in batch],
            )
            return {"input_ids": src_enc["input_ids"], "attention_mask": src_enc["attention_mask"], "labels": labels, "rel_matrix": rel}

        def step(m, batch, dev):
            return m(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["labels"].to(dev), batch["rel_matrix"].to(dev)).loss

        t_loader = DataLoader(train_ds, batch_size=self.cfg.training.batch_size, shuffle=True, collate_fn=collate)
        v_loader = DataLoader(val_ds, batch_size=self.cfg.training.batch_size, shuffle=False, collate_fn=collate) if val_ds and len(val_ds) > 0 else None
        return self._run_generic_loop("parser", model, t_loader, v_loader, step, lambda m, d: m.save_pretrained(d))

    def mine_verifier_samples(
        self,
        records: list[ExampleRecord],
        caches: dict[str, tuple[SchemaBundle, ValueIndex]],
        parser_dir: Path,
        max_records: int = 1200,
    ) -> list[dict[str, Any]]:
        """Run trained parser over train DBs to collect execution-divergence hard negatives."""
        parser = GraphixCodeT5Parser.from_pretrained_dir(parser_dir, map_location=str(self.device)).to(self.device)
        rng = random.Random(self.cfg.seed)
        subset = records[:] if len(records) <= max_records else rng.sample(records, max_records)
        samples: list[dict[str, Any]] = []

        for rec in subset:
            bundle, v_idx = caches[rec.db_id]
            vals = v_idx.probe(rec.question, top_k=5)
            gold_set = {t.lower() for t in rec.gold_tables}
            tables = [t for t in bundle.tables if t.name.lower() in gold_set][:6] or bundle.tables[:4]
            schema_txt = format_schema_for_parser(bundle, tables, vals, rec.question, rec.evidence)

            g_ok, g_rows, _, g_msg = execute_sqlite_readonly(rec.db_path, rec.sql)
            if g_ok and g_rows is not None:
                samples.append({
                    "question": rec.question, "schema_text": schema_txt, "candidate_sql": rec.sql,
                    "exec_status": f"ok rows={len(g_rows)}", "label": 1.0,
                })

            cands = parser.generate_candidates(
                schema_txt, bundle, tables, vals, num_candidates=3,
                max_source_len=self.cfg.models.parser_max_source_length,
                max_target_len=self.cfg.models.parser_max_target_length,
                use_xgrammar=self.cfg.models.use_xgrammar,
            )
            for c_sql in cands:
                p_ok, p_rows, _, p_msg = execute_sqlite_readonly(rec.db_path, c_sql)
                is_right = bool(g_ok and p_ok and g_rows is not None and p_rows is not None and results_equivalent(rec.sql, g_rows, p_rows))
                if not is_right:
                    samples.append({
                        "question": rec.question, "schema_text": schema_txt, "candidate_sql": c_sql,
                        "exec_status": f"ok rows={len(p_rows)}" if p_ok and p_rows is not None else f"err={p_msg[:60]}",
                        "label": 0.0,
                    })
        return samples

    def train_verifier(self, train_records: list[ExampleRecord], val_records: list[ExampleRecord], caches: dict[str, tuple[SchemaBundle, ValueIndex]]) -> dict[str, Any]:
        parser_art = Path(self.cfg.training.output_dir) / "artifacts" / "parser"
        if not parser_art.exists():
            raise FileNotFoundError(f"Parser artifact missing at {parser_art}; train stage='parser' before 'verifier'.")
        t_samples = self.mine_verifier_samples(train_records, caches, parser_art, max_records=1500)
        v_samples = self.mine_verifier_samples(val_records, caches, parser_art, max_records=300) if val_records else []
        model = ModernBertVerifier(self.cfg.models.verifier_model_id)

        def collate(batch):
            texts = [model.format_pair(b["question"], b["schema_text"], b["candidate_sql"], b["exec_status"]) for b in batch]
            enc = model.tokenizer(texts, padding=True, truncation=True, max_length=self.cfg.models.verifier_max_length, return_tensors="pt")
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

        def step(m, batch, dev):
            return m(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["labels"].to(dev))["loss"]

        t_loader = DataLoader(VerifierDataset(t_samples), batch_size=self.cfg.training.batch_size, shuffle=True, collate_fn=collate)
        v_loader = DataLoader(VerifierDataset(v_samples), batch_size=self.cfg.training.batch_size, shuffle=False, collate_fn=collate) if v_samples else None
        return self._run_generic_loop("verifier", model, t_loader, v_loader, step, lambda m, d: m.save_pretrained(d))
