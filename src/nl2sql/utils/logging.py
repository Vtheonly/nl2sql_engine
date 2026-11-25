from __future__ import annotations

import logging
import sys
from typing import Any

import psutil
import torch


def get_logger(name: str = "nl2sql", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_resource_stats() -> dict[str, Any]:
    proc = psutil.Process()
    stats: dict[str, Any] = {
        "ram_rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
        "cpu_percent": psutil.cpu_percent(interval=None),
    }
    if torch.cuda.is_available():
        stats["gpu_name"] = torch.cuda.get_device_name(0)
        stats["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        stats["vram_peak_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)
    return stats
