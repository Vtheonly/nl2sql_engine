from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    data_dir: Path = Path("./workspace/data")
    use_bird: bool = True
    use_bird_filtered_train: bool = True
    use_spider: bool = False
    spider_db_zip: Optional[Path] = None
    spider_db_url: Optional[str] = None
    val_ratio_by_db: float = Field(default=0.15, ge=0.05, le=0.5)
    schema_perturbation_prob: float = Field(default=0.35, ge=0.0, le=1.0)
    max_distinct_values_per_col: int = Field(default=10000, ge=100)
    # Non-text columns (ints, dates, enums) are only indexed when they carry at
    # most this many distinct values; genuinely high-cardinality numerics (ids,
    # amounts) stay out of the literal-grounding index.
    low_cardinality_max_distinct: int = Field(default=100, ge=2, le=1000)
    timeout_sec: int = Field(default=120, ge=10)


class ModelConfig(BaseModel):
    embedder_model_id: str = "BAAI/bge-small-en-v1.5"
    linker_model_id: str = "answerdotai/ModernBERT-base"
    parser_model_id: str = "Salesforce/codet5p-220m"
    verifier_model_id: str = "answerdotai/ModernBERT-base"
    linker_max_length: int = 4096
    parser_max_source_length: int = 1024
    parser_max_target_length: int = 192
    verifier_max_length: int = 1024
    linker_recall_pos_weight: float = 3.0
    linker_threshold: float = 0.30
    linker_min_tables: int = 2
    linker_max_tables: int = 8
    num_return_candidates: int = 4
    use_xgrammar: bool = True
    abstention_threshold: float = 0.45
    abstention_margin: float = 0.05


class TrainingConfig(BaseModel):
    stage: Literal["embedder", "linker", "parser", "verifier", "all"] = "parser"
    output_dir: Path = Path("./workspace/runs/exp01")
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=8, ge=1)
    grad_accum_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.06, ge=0.0, le=0.5)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    device: Optional[str] = None
    eval_every_epochs: int = Field(default=1, ge=1)
    save_every_steps: int = Field(default=250, ge=10)
    keep_last_k_checkpoints: int = Field(default=3, ge=1)
    early_stopping_patience: int = Field(default=3, ge=1)
    resume_from: Optional[str] = "auto"
    num_workers: int = 2
    # Warmup-Stable-Decay (WSD) replaces the fixed-horizon cosine by default:
    # extending training across multiple Colab sessions never decays the LR to
    # exactly zero, so resumed runs keep learning instead of freezing.
    lr_schedule: Literal["wsd", "cosine"] = "wsd"
    # LR floor during the final decay phase, as a fraction of the peak LR.
    wsd_final_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    # Fraction of the *planned* optimizer steps reserved for the final decay phase.
    wsd_decay_ratio: float = Field(default=0.3, ge=0.05, le=1.0)
    # Chunk size for verifier negative mining (table selection + compilation run
    # per chunk instead of one monolithic record loop).
    mining_batch_size: int = Field(default=32, ge=1)
    # Cut activation memory for long-context encoders (ModernBERT at 4096 tokens on
    # free-tier GPUs) at the cost of a recompute pass.
    gradient_checkpointing: bool = False


class HubConfig(BaseModel):
    enabled: bool = False
    repo_id: Optional[str] = None
    token: Optional[str] = None
    private: bool = True
    push_checkpoints: bool = False
    push_final_models: bool = True
    push_eval_reports: bool = True


class EngineConfig(BaseModel):
    project_name: str = "nl2sql-engine"
    work_dir: Path = Path("./workspace")
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hub: HubConfig = Field(default_factory=HubConfig)


def load_config(path: str | Path, overrides: Optional[dict] = None) -> EngineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(raw.get(k), dict):
                raw[k].update(v)
            else:
                raw[k] = v
    return EngineConfig.model_validate(raw)


def save_config(cfg: EngineConfig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return out
