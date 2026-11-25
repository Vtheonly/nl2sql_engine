from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from nl2sql.config import load_config
from nl2sql.engine import NL2SQLEngine, create_app
from nl2sql.utils.logging import get_logger

logger = get_logger(__name__)


def build_engine(
    config_path: str | Path,
    artifacts_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> NL2SQLEngine:
    cfg = load_config(config_path)
    art = Path(artifacts_dir) if artifacts_dir else Path(cfg.training.output_dir) / "artifacts"
    return NL2SQLEngine(cfg, artifacts_dir=art, device=device)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="nl2sql.serve",
        description="Serve the NL2SQL engine FastAPI app (/query, /health).",
    )
    parser.add_argument("--config", default="configs/default_train.yaml", help="Engine config YAML")
    parser.add_argument("--artifacts", default=None, help="Artifacts dir (defaults to <output_dir>/artifacts)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args(argv)

    engine = build_engine(args.config, args.artifacts, args.device)
    app = create_app(engine)
    logger.info("Serving NL2SQL engine on %s:%d (models_loaded=%s)", args.host, args.port, engine.models_loaded)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
