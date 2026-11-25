from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from nl2sql.compiler.sql_compiler import SQLSecurityError, compile_and_validate_sql
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
from nl2sql.retrieval.schema_retriever import DenseTableEncoder, HybridSchemaRetriever
from nl2sql.training.checkpoint import CheckpointManager
from nl2sql.utils.logging import get_logger, get_resource_stats
from nl2sql.utils.repro import set_seed

logger = get_logger(__name__)


class _EmbedderCollate:
    """MNRL bi-encoder pair collate (module-level so DataLoader workers can pickle it)."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Any]:
        q_enc = self.tokenizer([b["query"] for b in batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
        p_enc = self.tokenizer([b["positive"] for b in batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
        return {"q": q_enc, "p": p_enc}


class _LinkerCollate:
    """Holds only the tokenizer: DataLoader workers pickle the collate, so shipping
    model weights inside it would duplicate the network in every worker."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts, t_lbls, c_lbls = [], [], []
        for item in batch:
            txt, _, _ = ModernBertSchemaLinker.format_inputs(
                item["question"], item["bundle"].tables, item["value_matches"], item["evidence"]
            )
            texts.append(txt)
            t_lbls.append(torch.tensor([t["label"] for t in item["table_items"]], dtype=torch.float32))
            c_lbls.append(torch.tensor([c["label"] for c in item["col_items"]], dtype=torch.float32))
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "table_labels": t_lbls, "column_labels": c_lbls}


class _ParserCollate:
    def __init__(self, tokenizer: Any, max_source_length: int, max_target_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        src_enc = self.tokenizer(
            [b["source_text"] for b in batch], padding=True, truncation=True,
            max_length=self.max_source_length, return_tensors="pt",
        )
        tgt_enc = self.tokenizer(
            text_target=[b["target_sql"] for b in batch], padding=True, truncation=True,
            max_length=self.max_target_length, return_tensors="pt",
        )
        labels = tgt_enc["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100
        rel = GraphixCodeT5Parser.build_relation_matrix(
            src_enc["input_ids"], self.tokenizer,
            [b["bundle"] for b in batch], [b["selected_tables"] for b in batch], [b["value_matches"] for b in batch],
        )
        return {"input_ids": src_enc["input_ids"], "attention_mask": src_enc["attention_mask"], "labels": labels, "rel_matrix": rel}


class _VerifierCollate:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [ModernBertVerifier.format_pair(b["question"], b["schema_text"], b["candidate_sql"], b["exec_status"]) for b in batch]
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def _maybe_enable_gradient_checkpointing(model: nn.Module) -> None:
    """Cut activation memory for long-context encoders (ModernBERT at 4096 tokens)."""
    encoder = getattr(model, "encoder", None)
    # transformers renamed the API (gradient_checkpointing_enable -> activation_checkpointing_enable);
    # try both so the flag keeps working across versions.
    for attr in ("activation_checkpointing_enable", "gradient_checkpointing_enable"):
        enable = getattr(encoder, attr, None)
        if enable is None:
            continue
        try:
            try:
                enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                enable()
            return
        except Exception as exc:
            logger.warning("Gradient checkpointing unavailable for %s (%s).", type(model).__name__, exc)
            return


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

    def _loader(self, dataset: Any, collate: Any, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.cfg.training.num_workers,
        )

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
        # Exact step count: floor division undercounted whenever the last partial
        # accumulation window remained, decaying the LR schedule to zero early.
        total_steps = max(1, math.ceil(len(train_loader) / tcfg.grad_accum_steps) * tcfg.epochs)
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
            # Checkpoints always record the NEXT epoch to run (epoch-boundary resume),
            # so an interrupted run never re-processes already-trained batches while
            # global_step (and the LR schedule) races ahead.
            start_epoch = int(payload["epoch"])
            global_step = int(payload["global_step"])
            best_val_loss = float(payload["best_metric"])
            patience_left = int(payload.get("patience_left", tcfg.early_stopping_patience))
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
                        # Mid-epoch saves record epoch + 1: resuming continues at the
                        # next epoch boundary instead of replaying this epoch's batches.
                        # TRADE-OFF (documented): the remaining batches of the interrupted
                        # epoch are skipped, while global_step (and the LR schedule)
                        # still count them -- an interrupted large epoch therefore does
                        # not re-run its unprocessed tail. Exact mid-epoch continuation
                        # would require deterministic per-epoch shuffling plus a
                        # skip-batches marker; use epoch-boundary saves instead when
                        # every batch matters more than fast crash recovery.
                        self.ckpt_mgr.save(
                            stage=stage, model=model, optimizer=optimizer, scheduler=scheduler,
                            scaler=scaler, epoch=epoch + 1, global_step=global_step,
                            best_metric=best_val_loss, config_dict=self.cfg.model_dump(mode="json"),
                            metrics_history=history, is_best=False, patience_left=patience_left,
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
                metrics_history=history, is_best=is_best, patience_left=patience_left,
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

        collate = _EmbedderCollate(tokenizer)

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

        t_loader = self._loader(train_ds, collate, batch_size=max(16, self.cfg.training.batch_size * 2), shuffle=True)
        v_loader = self._loader(val_ds, collate, batch_size=32, shuffle=False) if val_ds and len(val_ds) > 0 else None
        return self._run_generic_loop("embedder", encoder, t_loader, v_loader, step, save_fn)

    def train_linker(self, train_records: list[ExampleRecord], val_records: list[ExampleRecord], caches: dict[str, tuple[SchemaBundle, ValueIndex]]) -> dict[str, Any]:
        model = ModernBertSchemaLinker(self.cfg.models.linker_model_id, pos_weight=self.cfg.models.linker_recall_pos_weight)
        if self.cfg.training.gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(model)
        train_ds = LinkerDataset(train_records, caches, perturb_prob=self.cfg.data.schema_perturbation_prob, seed=self.cfg.seed)
        val_ds = LinkerDataset(val_records, caches, perturb_prob=0.0, seed=self.cfg.seed) if val_records else None

        collate = _LinkerCollate(model.tokenizer, self.cfg.models.linker_max_length)

        def step(m, batch, dev):
            return m(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["table_labels"], batch["column_labels"])["loss"]

        t_loader = self._loader(train_ds, collate, batch_size=max(2, self.cfg.training.batch_size // 2), shuffle=True)
        v_loader = self._loader(val_ds, collate, batch_size=max(2, self.cfg.training.batch_size // 2), shuffle=False) if val_ds and len(val_ds) > 0 else None
        return self._run_generic_loop("linker", model, t_loader, v_loader, step, lambda m, d: m.save_pretrained(d))

    def train_parser(self, train_records: list[ExampleRecord], val_records: list[ExampleRecord], caches: dict[str, tuple[SchemaBundle, ValueIndex]]) -> dict[str, Any]:
        model = GraphixCodeT5Parser(self.cfg.models.parser_model_id)
        train_ds = ParserDataset(train_records, caches, perturb_prob=self.cfg.data.schema_perturbation_prob, seed=self.cfg.seed)
        val_ds = ParserDataset(val_records, caches, perturb_prob=0.0, seed=self.cfg.seed) if val_records else None

        collate = _ParserCollate(model.tokenizer, self.cfg.models.parser_max_source_length, self.cfg.models.parser_max_target_length)

        def step(m, batch, dev):
            return m(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["labels"].to(dev), batch["rel_matrix"].to(dev)).loss

        t_loader = self._loader(train_ds, collate, batch_size=self.cfg.training.batch_size, shuffle=True)
        v_loader = self._loader(val_ds, collate, batch_size=self.cfg.training.batch_size, shuffle=False) if val_ds and len(val_ds) > 0 else None
        return self._run_generic_loop("parser", model, t_loader, v_loader, step, lambda m, d: m.save_pretrained(d))

    def _serve_time_tables(
        self,
        rec: ExampleRecord,
        bundle: SchemaBundle,
        v_idx: ValueIndex,
        retriever: HybridSchemaRetriever,
        linker: Optional[ModernBertSchemaLinker],
    ) -> list[Any]:
        """Select tables the way the serving engine does: retrieval + linker pruning.

        Mining with oracle gold tables made the verifier learn under a table
        distribution it never sees at inference time.
        """
        ret = retriever.retrieve(rec.question, bundle, v_idx)
        cand_tables = ret.candidate_tables
        if linker is not None:
            return linker.prune_schema(
                rec.question, bundle, cand_tables, ret.value_matches, evidence=rec.evidence,
                threshold=self.cfg.models.linker_threshold,
                min_tables=self.cfg.models.linker_min_tables,
                max_tables=self.cfg.models.linker_max_tables,
                max_length=self.cfg.models.linker_max_length,
            )
        return cand_tables[: self.cfg.models.linker_max_tables]

    def mine_verifier_samples(
        self,
        records: list[ExampleRecord],
        caches: dict[str, tuple[SchemaBundle, ValueIndex]],
        parser_dir: Path,
        max_records: int = 1200,
    ) -> list[dict[str, Any]]:
        """Run trained parser over train DBs to collect execution-divergence hard negatives."""
        parser = GraphixCodeT5Parser.from_pretrained_dir(parser_dir, map_location=str(self.device)).to(self.device)

        art_dir = Path(self.cfg.training.output_dir) / "artifacts"
        # Same retrieval stack as the serving engine, so mined samples match the
        # inference-time table distribution (and SQLGlot-canonical SQL formatting).
        try:
            embedder = DenseTableEncoder(self.cfg.models.embedder_model_id, device=str(self.device))
            retriever = HybridSchemaRetriever(embedder)
        except Exception as exc:
            logger.warning("Could not load embedder for verifier mining (%s); using tables[:8].", exc)
            retriever = None
        linker: Optional[ModernBertSchemaLinker] = None
        if (art_dir / "linker" / "linker_weights.pt").exists():
            try:
                linker = ModernBertSchemaLinker.from_pretrained_dir(art_dir / "linker", map_location=str(self.device)).to(self.device)
            except Exception as exc:
                logger.warning("Could not load linker artifact for verifier mining (%s).", exc)

        rng = random.Random(self.cfg.seed)
        subset = records[:] if len(records) <= max_records else rng.sample(records, max_records)
        samples: list[dict[str, Any]] = []

        for rec in subset:
            bundle, v_idx = caches[rec.db_id]
            if retriever is not None:
                tables = self._serve_time_tables(rec, bundle, v_idx, retriever, linker)
            else:
                tables = bundle.tables[: self.cfg.models.linker_max_tables]
            tables = tables or bundle.tables[:4]
            val_hits = rec.value_matches or v_idx.probe(rec.question, top_k=5)
            schema_txt = format_schema_for_parser(bundle, tables, val_hits, rec.question, rec.evidence)

            # Both gold and predicted candidates pass through the same compiler the
            # engine uses, so the verifier cannot win by spotting raw-vs-compiled
            # formatting differences instead of semantic errors.
            try:
                gold_sql = compile_and_validate_sql(rec.sql, bundle, default_limit=0).sql
            except SQLSecurityError:
                gold_sql = rec.sql.strip()

            g_ok, g_rows, _, g_msg = execute_sqlite_readonly(rec.db_path, gold_sql)
            if g_ok and g_rows is not None:
                samples.append({
                    "question": rec.question, "schema_text": schema_txt, "candidate_sql": gold_sql,
                    "exec_status": f"ok rows={len(g_rows)}", "label": 1.0,
                })

            cands = parser.generate_candidates(
                schema_txt, bundle, tables, val_hits, num_candidates=3,
                max_source_len=self.cfg.models.parser_max_source_length,
                max_target_len=self.cfg.models.parser_max_target_length,
                use_xgrammar=self.cfg.models.use_xgrammar,
            )
            for c_sql in cands:
                try:
                    cand_sql = compile_and_validate_sql(c_sql, bundle, default_limit=0).sql
                except SQLSecurityError:
                    cand_sql = c_sql.strip()
                p_ok, p_rows, _, p_msg = execute_sqlite_readonly(rec.db_path, cand_sql)
                is_right = bool(g_ok and p_ok and g_rows is not None and p_rows is not None and results_equivalent(rec.sql, g_rows, p_rows))
                if not is_right:
                    samples.append({
                        "question": rec.question, "schema_text": schema_txt, "candidate_sql": cand_sql,
                        "exec_status": f"ok rows={len(p_rows)}" if p_ok and p_rows is not None else f"err={p_msg[:60]}",
                        "label": 0.0,
                    })
        return samples

    def _calibrate_temperature(self, model: ModernBertVerifier, val_samples: list[dict[str, Any]], max_length: int) -> None:
        """Fit the verifier temperature on validation logits (1-pass LBFGS).

        Without calibration the abstention threshold operates on raw sigmoid scores,
        making its 0.45 cut-off arbitrary. Only the validation forward pass runs
        under no_grad: the L-BFGS closure needs live autograd on ``temp`` -- a
        method-wide @torch.no_grad() would make loss.backward() raise "element 0
        of tensors does not require grad" on torch versions whose LBFGS does not
        internally re-enable grad for the closure, silently keeping T=1.0.
        """
        if len(val_samples) < 8 or len({s["label"] for s in val_samples}) < 2:
            return
        model.eval()
        collate = _VerifierCollate(model.tokenizer, self.cfg.models.verifier_max_length)
        loader = DataLoader(VerifierDataset(val_samples), batch_size=16, shuffle=False, collate_fn=collate)
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                out = model(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
                logits_all.append(out["logits"].float().cpu())
                labels_all.append(batch["labels"].cpu())
        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all)

        temp = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.LBFGS([temp], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(logits / temp.clamp(min=0.05), labels)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as exc:
            logger.warning("Temperature calibration failed (%s); keeping T=1.0.", exc)
            return
        fitted = float(temp.detach().clamp(min=0.05, max=10.0).item())
        model.temperature.fill_(fitted)
        logger.info("Verifier temperature calibrated to %.4f.", fitted)

    def train_verifier(self, train_records: list[ExampleRecord], val_records: list[ExampleRecord], caches: dict[str, tuple[SchemaBundle, ValueIndex]]) -> dict[str, Any]:
        parser_art = Path(self.cfg.training.output_dir) / "artifacts" / "parser"
        if not parser_art.exists():
            raise FileNotFoundError(f"Parser artifact missing at {parser_art}; train stage='parser' before 'verifier'.")
        t_samples = self.mine_verifier_samples(train_records, caches, parser_art, max_records=1500)
        v_samples = self.mine_verifier_samples(val_records, caches, parser_art, max_records=300) if val_records else []
        model = ModernBertVerifier(self.cfg.models.verifier_model_id)
        if self.cfg.training.gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(model)

        collate = _VerifierCollate(model.tokenizer, self.cfg.models.verifier_max_length)

        def step(m, batch, dev):
            return m(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["labels"].to(dev))["loss"]

        t_loader = self._loader(VerifierDataset(t_samples), collate, batch_size=self.cfg.training.batch_size, shuffle=True)
        v_loader = self._loader(VerifierDataset(v_samples), collate, batch_size=self.cfg.training.batch_size, shuffle=False) if v_samples else None
        result = self._run_generic_loop("verifier", model, t_loader, v_loader, step, lambda m, d: m.save_pretrained(d))

        # Calibrate + persist the BEST checkpoint, not the final epoch's weights.
        # _run_generic_loop saves the best-validation epoch to artifacts/verifier;
        # after early stopping the in-memory `model` sits on the overfit final epoch,
        # so calibrating and save_pretrained-ing it directly would clobber the best
        # checkpoint. Reload the best artifact first and calibrate that.
        verifier_dir = Path(self.cfg.training.output_dir) / "artifacts" / "verifier"
        if verifier_dir.exists():
            model = ModernBertVerifier.from_pretrained_dir(verifier_dir, map_location=str(self.device)).to(self.device)
        self._calibrate_temperature(model, v_samples, self.cfg.models.verifier_max_length)
        model.save_pretrained(verifier_dir)
        return result
