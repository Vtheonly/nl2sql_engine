from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

from nl2sql.utils.repro import capture_rng_state, restore_rng_state

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Atomic, resumable training checkpoint manager with top-k / latest retention."""

    def __init__(self, run_dir: str | Path, keep_last_k: int = 3) -> None:
        self.run_dir = Path(run_dir)
        self.ckpt_root = self.run_dir / "checkpoints"
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k

    def save(
        self,
        stage: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[torch.amp.GradScaler],
        epoch: int,
        global_step: int,
        best_metric: float,
        config_dict: dict[str, Any],
        metrics_history: list[dict[str, Any]],
        is_best: bool = False,
        patience_left: Optional[int] = None,
        batch_idx_in_epoch: int = 0,
    ) -> Path:
        """Persist resumable trainer state.

        ``epoch`` is the epoch to run NEXT (end-of-epoch saves pass ``epoch + 1``);
        mid-epoch saves pass the CURRENT epoch together with ``batch_idx_in_epoch``
        = the number of batches of that epoch already processed, so resume can
        fast-forward to the exact interruption point via ``itertools.islice``.
        """
        step_dir = self.ckpt_root / stage / f"step_{global_step:07d}"
        tmp_dir = self.ckpt_root / stage / f".tmp_step_{global_step:07d}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "stage": stage,
            "epoch": epoch,
            "batch_idx_in_epoch": int(batch_idx_in_epoch),
            "global_step": global_step,
            "best_metric": best_metric,
            "patience_left": patience_left,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "rng_state": capture_rng_state(),
            "config": config_dict,
            "metrics_history": metrics_history,
        }
        torch.save(payload, tmp_dir / "trainer_state.pt")
        (tmp_dir / "meta.json").write_text(
            json.dumps(
                {
                    "stage": stage, "epoch": epoch, "batch_idx_in_epoch": int(batch_idx_in_epoch),
                    "global_step": global_step,
                    "best_metric": best_metric, "patience_left": patience_left,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if step_dir.exists():
            shutil.rmtree(step_dir)
        os.replace(tmp_dir, step_dir)

        latest_ptr = self.ckpt_root / stage / "latest.txt"
        latest_ptr.write_text(str(step_dir), encoding="utf-8")

        if is_best:
            best_dir = self.ckpt_root / stage / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(step_dir, best_dir)

        self._prune_old(stage)
        return step_dir

    def _prune_old(self, stage: str) -> None:
        stage_dir = self.ckpt_root / stage
        if not stage_dir.exists():
            return
        step_dirs = sorted([d for d in stage_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])
        while len(step_dirs) > self.keep_last_k:
            oldest = step_dirs.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)

    def resolve_resume_path(self, stage: str, resume_from: Optional[str]) -> Optional[Path]:
        if not resume_from:
            return None
        if resume_from == "auto":
            latest_ptr = self.ckpt_root / stage / "latest.txt"
            if latest_ptr.exists():
                p = Path(latest_ptr.read_text(encoding="utf-8").strip())
                if (p / "trainer_state.pt").exists():
                    return p
            return None
        p = Path(resume_from)
        if not (p / "trainer_state.pt").exists():
            return None
        # An explicit path may belong to a DIFFERENT stage (copied directory, wrong
        # hand-written path): loading it would silently graft the parser's optimizer
        # state onto the verifier. meta.json records the stage without a torch.load.
        meta_path = p / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read checkpoint meta at %s (%s); refusing to resume.", p, exc)
                return None
            if meta.get("stage") != stage:
                logger.warning(
                    "Checkpoint %s belongs to stage '%s', not '%s'; refusing cross-stage resume.",
                    p, meta.get("stage"), stage,
                )
                return None
        return p

    def load(
        self,
        ckpt_dir: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        payload = torch.load(ckpt_dir / "trainer_state.pt", map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict"):
            scaler.load_state_dict(payload["scaler_state_dict"])
        if payload.get("rng_state"):
            restore_rng_state(payload["rng_state"])
        return payload
