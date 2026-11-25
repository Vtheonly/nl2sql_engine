from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from nl2sql.artifacts.hub import HubArtifactManager
from nl2sql.config import EngineConfig, load_config, save_config
from nl2sql.datasets.acquire import DatasetAcquisitionResult, acquire_all_datasets
from nl2sql.datasets.prepare import ExampleRecord, prepare_records
from nl2sql.db.introspect import SchemaBundle
from nl2sql.db.value_index import ValueIndex
from nl2sql.engine import NL2SQLEngine
from nl2sql.evaluation.evaluator import EvalMetrics, NL2SQLEvaluator
from nl2sql.training.trainer import NL2SQLTrainer
from nl2sql.utils.logging import get_logger

logger = get_logger(__name__)


class NL2SQLPipeline:
    """Single public orchestration facade exposed to notebooks and CLI scripts."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.acq: Optional[DatasetAcquisitionResult] = None
        self.train_records: list[ExampleRecord] = []
        self.val_records: list[ExampleRecord] = []
        self.test_records: list[ExampleRecord] = []
        self.caches: dict[str, tuple[SchemaBundle, ValueIndex]] = {}

        run_dir = Path(cfg.training.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config_snapshot.yaml")

    @classmethod
    def from_config_file(cls, path: str | Path, overrides: Optional[dict] = None) -> "NL2SQLPipeline":
        return cls(load_config(path, overrides=overrides))

    def prepare_data(self) -> dict[str, int]:
        self.acq = acquire_all_datasets(self.cfg.data)
        cache_dir = Path(self.cfg.data.data_dir) / "schema_caches"
        self.train_records, self.val_records, self.test_records, self.caches = prepare_records(
            self.acq,
            cache_dir=cache_dir,
            val_ratio_by_db=self.cfg.data.val_ratio_by_db,
            seed=self.cfg.seed,
            max_distinct_per_col=self.cfg.data.max_distinct_values_per_col,
        )
        summary = {
            "train_examples": len(self.train_records),
            "val_examples": len(self.val_records),
            "test_examples": len(self.test_records),
            "cached_databases": len(self.caches),
        }
        logger.info("Prepared data summary: %s", summary)
        return summary

    def train(self, stage: Optional[str] = None) -> dict[str, Any]:
        if not self.train_records:
            self.prepare_data()
        target_stage = stage or self.cfg.training.stage
        trainer = NL2SQLTrainer(self.cfg)
        stages = ["embedder", "linker", "parser", "verifier"] if target_stage == "all" else [target_stage]
        results: dict[str, Any] = {}
        for st in stages:
            if st == "embedder":
                results[st] = trainer.train_embedder(self.train_records, self.val_records, self.caches)
            elif st == "linker":
                results[st] = trainer.train_linker(self.train_records, self.val_records, self.caches)
            elif st == "parser":
                results[st] = trainer.train_parser(self.train_records, self.val_records, self.caches)
            elif st == "verifier":
                results[st] = trainer.train_verifier(self.train_records, self.val_records, self.caches)
            else:
                raise ValueError(f"Unknown training stage: {st}")
        return results

    def evaluate(
        self,
        split: str = "test",
        use_oracle_evidence: bool = False,
        max_samples: Optional[int] = None,
    ) -> EvalMetrics:
        if not self.caches:
            self.prepare_data()
        records = self.val_records if split == "val" else self.test_records
        if max_samples is not None:
            records = records[:max_samples]

        art_dir = Path(self.cfg.training.output_dir) / "artifacts"
        engine = NL2SQLEngine(self.cfg, artifacts_dir=art_dir, device=self.cfg.training.device)
        for db_id, (bundle, v_idx) in self.caches.items():
            db_paths = [r.db_path for r in records if r.db_id == db_id]
            sqlite_p = db_paths[0] if db_paths else None
            engine.db_registry[db_id] = (bundle, v_idx, sqlite_p)

        evaluator = NL2SQLEvaluator()
        metrics = evaluator.evaluate(
            records,
            predict_fn=lambda rec: engine.answer(
                db_id=rec.db_id,
                question=rec.question,
                evidence=(rec.evidence if use_oracle_evidence else ""),
            ).model_dump(),
            dataset_name=f"{split}_{'with_evidence' if use_oracle_evidence else 'no_evidence'}",
        )
        eval_dir = Path(self.cfg.training.output_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / f"{metrics.dataset_name}.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
        return metrics

    def push_artifacts(self) -> Optional[str]:
        mgr = HubArtifactManager(self.cfg.hub)
        return mgr.push_run_artifacts(self.cfg.training.output_dir)
