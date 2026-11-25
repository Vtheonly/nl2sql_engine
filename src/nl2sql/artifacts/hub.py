from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from nl2sql.config import HubConfig
from nl2sql.utils.logging import get_logger

logger = get_logger(__name__)


class HubArtifactManager:
    """Uploads trained final models, eval reports, and (optionally) checkpoints—never raw dataset archives."""

    def __init__(self, cfg: HubConfig) -> None:
        self.cfg = cfg
        token = cfg.token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=token) if cfg.enabled else None

    def push_run_artifacts(self, run_dir: str | Path) -> Optional[str]:
        if not self.cfg.enabled or not self.cfg.repo_id or self.api is None:
            logger.info("Hub upload skipped (hub.enabled=False or repo_id unset).")
            return None

        run_path = Path(run_dir)
        self.api.create_repo(repo_id=self.cfg.repo_id, private=self.cfg.private, exist_ok=True)
        allow_patterns: list[str] = ["config_snapshot.yaml"]
        if self.cfg.push_final_models:
            allow_patterns.append("artifacts/**")
        if self.cfg.push_eval_reports:
            allow_patterns.append("eval/**")
        if self.cfg.push_checkpoints:
            allow_patterns.append("checkpoints/**")

        ignore_patterns = ["*.sqlite", "*.zip", "*.part", "data/**"]
        logger.info("Uploading run artifacts from %s to hf://%s ...", run_path, self.cfg.repo_id)
        commit_info = self.api.upload_folder(
            repo_id=self.cfg.repo_id,
            folder_path=str(run_path),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            commit_message="Upload NL2SQL trained artifacts and evaluation report",
        )
        return str(commit_info.commit_url)
