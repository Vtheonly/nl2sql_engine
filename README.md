# nl2sql_engine

Cross-database, compact neuro-symbolic NL2SQL engine: BIRD/Spider-trained schema
retrieval, schema linking, grammar-constrained SQL generation, verification, and
safe read-only execution — all behind one orchestration facade.

**Repository:** https://github.com/Vtheonly/nl2sql_engine

## Architecture (Layers 0–7)

| Layer | Module | Role |
| --- | --- | --- |
| 0 | `nl2sql.db.introspect` | SQLAlchemy introspection, FK inference, schema caching with DDL-hash invalidation |
| 1 | `nl2sql.db.value_index` | Persistent FTS5 distinct-value index for literal grounding (Unicode-aware) |
| 2 | `nl2sql.retrieval` | Hybrid BM25 + dense (column-level, max-pooled) table retrieval with 1-hop FK expansion |
| 2 | `nl2sql.models.schema_linker` | ModernBERT table/column multi-head pruning |
| 3 | `nl2sql.engine` | Skeleton-keyed template cache (RBAC/dialect/evidence-safe) |
| 4 | `nl2sql.models.graphix_parser` | CodeT5+ 220M with Graphix relation-bias attention + XGrammar constrained decoding |
| 5 | `nl2sql.compiler.sql_compiler` | SQLGlot validation, join synthesis (alias-aware), RBAC, fan-out guard |
| 6 | `nl2sql.evaluation.evaluator` | Read-only SQLite execution (`PRAGMA query_only`, statement gate, timeouts) |
| 7 | `nl2sql.models.verifier` | ModernBERT selective correctness scorer with fitted temperature |

## Quickstart

```bash
pip install -e ".[constrained,dev]"   # constrained = xgrammar; dev = pytest + httpx
pytest -q                              # full suite runs offline on tiny models
```

### Train / evaluate (notebook or code)

The notebook clones this repository directly and drives the public pipeline:

```python
from nl2sql.pipeline import NL2SQLPipeline

pipeline = NL2SQLPipeline.from_config_file("configs/default_train.yaml", overrides={
    "training": {"stage": "all", "epochs": 5, "output_dir": "./workspace/runs/exp01"},
})
pipeline.prepare_data()          # downloads BIRD (train/dev/mini-dev) + builds caches
pipeline.train()                 # embedder -> linker -> parser -> verifier
pipeline.evaluate(split="test")  # also: "val", "minidev"
pipeline.push_artifacts()        # optional HuggingFace upload
```

See [`notebooks/train_nl2sql.ipynb`](notebooks/train_nl2sql.ipynb) for the
end-to-end Colab/Kaggle orchestrator.

### Serve

```bash
# Register databases directly from the CLI (repeatable) so /query works out of the box:
python -m nl2sql.serve --config configs/default_train.yaml --port 8000 \
    --db shop=sqlite:///shop.sqlite --db hr=sqlite:///hr.sqlite
# or: nl2sql-serve --db shop=sqlite:///shop.sqlite
```

Endpoints: `GET /health` (models loaded, registered DBs), `POST /register`
(`db_id`, `connection_url`, optional `cache_dir`/`column_meanings`/`example_queries`)
and `POST /query` (`db_id`, `question`, `evidence`, `allowed_tables`, `execute`,
`default_limit`).

```python
from nl2sql.engine import NL2SQLEngine, create_app

engine = NL2SQLEngine(cfg, artifacts_dir="./workspace/runs/exp01/artifacts")
engine.register_database("my_db", "sqlite:///my.db", cache_dir="./cache",
                          column_meanings={"my_db|t|c": "..."},
                          example_queries=[{"question": "...", "sql": "SELECT ..."}])
app = create_app(engine)
```

### Checkpointing & resume

`resume_from: auto` restores model/optimizer/scheduler/RNG state plus early-stopping
patience and best-metric bookkeeping. Mid-epoch checkpoints (`save_every_steps`)
record the *next* epoch boundary, so an interrupted epoch's remaining batches are
skipped on resume (no duplicate-batch replay) while `global_step` and the LR
schedule stay consistent — documented trade-off favoring crash recovery over exact
mid-epoch continuation.

## Testing

226+ tests run fully offline against synthetic SQLite databases and tiny
(~200K param) stand-in models — no network or GPU required:

```bash
pytest -q                 # everything
pytest tests/test_core.py -q
```

## Layout

```
configs/default_train.yaml   # engine + training configuration
notebooks/train_nl2sql.ipynb # thin orchestrator (no implementation logic)
src/nl2sql/                  # all implementation code
tests/                       # unit + heavy regression suites
```
