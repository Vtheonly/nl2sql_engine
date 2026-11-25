"""Notebook contract tests: the training notebook must stay a thin orchestrator.

Checks (a) the notebook is structurally valid, (b) no code cell contains core
implementation logic (models, losses, loops, downloads, uploads), and (c) the
cell sequence drives exactly the public pipeline interface: from_config_file ->
prepare_data -> train -> evaluate -> push_artifacts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "train_nl2sql.ipynb"

# Implementation-logic patterns that must never appear in the training notebook
FORBIDDEN_PATTERNS = {
    "nn.Module / model class definition": re.compile(r"class\s+\w+\s*\(.*nn\.Module"),
    "loss function definition": re.compile(r"def\s+\w*loss\w*\s*\("),
    "training loop definition": re.compile(r"def\s+\w*train\w*loop\w*\s*\(|for\s+\w+\s+in\s+range\(\s*\w*epochs"),
    "optimizer construction": re.compile(r"torch\.optim\.|AdamW\(|SGD\("),
    "backpropagation": re.compile(r"\.backward\(|\.step\(\s*\)|zero_grad"),
    "checkpoint serialization": re.compile(r"torch\.save|torch\.load"),
    "direct tensor/model code": re.compile(r"\bimport\s+torch\b|\bimport\s+transformers\b|from\s+transformers"),
    "HTTP download code": re.compile(r"requests\.(get|post)|urllib\.request|wget|curl\s"),
    "Hugging Face upload code": re.compile(r"HfApi|upload_folder|create_repo|push_to_hub"),
    "SQL string assembly": re.compile(r"sqlglot|SELECT\s+.+\s+FROM\s+.*['\"]", re.IGNORECASE),
    "class definitions of any kind": re.compile(r"^\s*class\s+\w+", re.MULTILINE),
}

def _load_notebook() -> dict:
    assert NOTEBOOK.exists(), f"missing notebook: {NOTEBOOK}"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_config_entry_point_loads() -> None:
    """The notebook's config file must parse into a validated EngineConfig."""
    from nl2sql.config import load_config

    cfg = load_config(NOTEBOOK.parent.parent / "configs" / "default_train.yaml")
    assert cfg.project_name == "nl2sql-engine"
    assert cfg.training.stage in ("embedder", "linker", "parser", "verifier", "all")
    assert cfg.models.parser_model_id == "Salesforce/codet5p-220m"
    assert cfg.hub.enabled is False


def test_notebook_is_structurally_valid() -> None:
    nb = _load_notebook()
    assert nb.get("nbformat") in (4,)
    assert nb.get("nbformat_minor") in range(0, 6)
    assert isinstance(nb.get("metadata"), dict)
    cells = nb["cells"]
    assert 5 <= len(cells) <= 9
    for cell in cells:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        assert all(isinstance(line, str) for line in cell["source"])
        if cell["cell_type"] == "code":
            assert isinstance(cell.get("outputs", []), list)
            assert cell.get("execution_count") is None or isinstance(cell["execution_count"], int)
    # Exactly one markdown title cell followed by the six documented sections
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert len(code_cells) == 6
    joined = "\n".join("".join(c["source"]) for c in code_cells)
    for section in range(1, 7):
        assert f"Section {section}:" in joined, f"notebook section {section} missing"


def test_notebook_contains_no_implementation_logic() -> None:
    nb = _load_notebook()
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        for label, pattern in FORBIDDEN_PATTERNS.items():
            assert not pattern.search(src), f"code cell {i} contains forbidden logic: {label}"
    # Only orchestration-level imports are allowed
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        for match in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", src, re.MULTILINE):
            root = match.group(1)
            allowed_roots = {"nl2sql", "pathlib", "os", "sys", "subprocess", "json", "pandas"}
            assert root.split(".")[0] in allowed_roots, (
                f"code cell {i} imports {root!r} - notebook must only orchestrate the codebase"
            )


def test_notebook_orchestrates_public_pipeline_interface(tmp_path: Path, monkeypatch, engine_config) -> None:
    """Execute the notebook's code cells (minus the pip-install setup cell) against
    a stubbed pipeline, proving the cells drive the intended public API sequence."""
    import nl2sql.pipeline as pipeline_mod
    from nl2sql.pipeline import NL2SQLPipeline

    nb = _load_notebook()
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    calls: list[str] = []
    train_results = {
        "parser": {
            "best_val_loss": 0.42,
            "artifact_dir": str(tmp_path / "artifacts" / "parser"),
            "history": [
                {"epoch": 1, "global_step": 2, "train_loss": 1.1, "val_loss": 1.0, "lr": 9e-5, "is_best": True},
                {"epoch": 2, "global_step": 4, "train_loss": 0.9, "val_loss": 0.8, "lr": 5e-5, "is_best": True},
            ],
        }
    }
    metrics_stub = type("Metrics", (), {
        "model_dump": lambda self: {"execution_accuracy": 0.5},
        "model_dump_json": lambda self, indent=2: "{}",
    })()

    class StubPipeline:
        def __init__(self, cfg) -> None:
            calls.append("init")

        @classmethod
        def from_config_file(cls, path, overrides=None):
            calls.append("from_config_file")
            assert overrides is not None
            return cls(overrides)

        def prepare_data(self):
            calls.append("prepare_data")
            return {"train_examples": 4, "val_examples": 2, "test_examples": 2, "cached_databases": 3}

        def train(self, stage=None):
            calls.append("train")
            return train_results

        def evaluate(self, split="test", use_oracle_evidence=False, max_samples=None):
            calls.append(f"evaluate:{split}:{use_oracle_evidence}")
            return metrics_stub

        def push_artifacts(self):
            calls.append("push_artifacts")
            return None

    # Point the notebook's config file at a real config and stub the pipeline class
    config_path = tmp_path / "default_train.yaml"
    from nl2sql.config import save_config
    save_config(engine_config, config_path)
    monkeypatch.setattr(pipeline_mod, "NL2SQLPipeline", StubPipeline)

    notebook_globals = {"__name__": "__nb__", "display": lambda *a, **k: None}

    # Skip the pip-install setup cell (cell 0); sandbox already has the package installed
    for src in code_cells[1:]:
        src = src.replace('"configs/default_train.yaml"', repr(str(config_path)))
        exec(compile(src, "<notebook-cell>", "exec"), notebook_globals)  # noqa: S102 - controlled execution

    assert calls == [
        "from_config_file",
        "init",
        "prepare_data",
        "train",
        "evaluate:test:False",
        "evaluate:test:True",
        "push_artifacts",
    ]
    # The notebook kept references to the results for its monitoring display
    assert notebook_globals["data_summary"]["train_examples"] == 4
    assert notebook_globals["train_results"]["parser"]["best_val_loss"] == 0.42
