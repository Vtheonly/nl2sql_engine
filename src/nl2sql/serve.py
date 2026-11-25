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


def register_databases_at_startup(engine: NL2SQLEngine, db_specs: list[str], cache_dir: Path) -> list[str]:
    """Register ``NAME=URL`` specs passed via ``--db`` before the server starts.

    Without this, ``nl2sql-serve`` exposes /query but an empty db_registry, so
    every request 404s ("Database not registered") until custom Python glue
    registers a database.
    """
    registered: list[str] = []
    for spec in db_specs:
        db_id, sep, url = spec.partition("=")
        db_id, url = db_id.strip(), url.strip()
        if not sep or not db_id or not url:
            raise SystemExit(f"Invalid --db spec {spec!r}: expected NAME=CONNECTION_URL (e.g. shop=sqlite:///shop.sqlite)")
        engine.register_database(db_id, url, cache_dir=cache_dir)
        registered.append(db_id)
        logger.info("Registered database '%s' from CLI (%s).", db_id, url)
    return registered


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="nl2sql.serve",
        description="Serve the NL2SQL engine FastAPI app (/health, /register, /query).",
    )
    parser.add_argument("--config", default="configs/default_train.yaml", help="Engine config YAML")
    parser.add_argument("--artifacts", default=None, help="Artifacts dir (defaults to <output_dir>/artifacts)")
    parser.add_argument(
        "--db",
        action="append",
        default=None,
        metavar="NAME=URL",
        dest="db_specs",
        help="Register a database at startup (repeatable), e.g. --db shop=sqlite:///shop.sqlite",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Schema/value-index cache dir for --db registrations (defaults to <data_dir>/schema_caches)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args(argv)

    engine = build_engine(args.config, args.artifacts, args.device)
    if args.db_specs:
        cache_dir = Path(args.cache_dir) if args.cache_dir else Path(engine.cfg.data.data_dir) / "schema_caches"
        register_databases_at_startup(engine, args.db_specs, cache_dir)
    app = create_app(engine)
    logger.info("Serving NL2SQL engine on %s:%d (models_loaded=%s)", args.host, args.port, engine.models_loaded)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
