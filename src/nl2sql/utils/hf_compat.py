"""Robust Hugging Face tokenizer loading across transformers releases.

transformers 5.2x regression: when a model repo ships only slow tokenizer
files (e.g. Salesforce/codet5p-220m: vocab/merges, no tokenizer.json),
``AutoTokenizer.from_pretrained`` auto-converts to fast and feeds the
converted added tokens (sentinels such as ``<extra_id_*>``) into the Rust
``tokenizers`` binding as plain dicts. The binding rejects them with
``TypeError: Input must be a List[Union[str, AddedToken]]``.

``load_tokenizer`` catches exactly that failure and retries once with an
explicit ``added_tokens_decoder`` kwarg whose entries are proper
``AddedToken`` objects (converted from the repo's tokenizer_config.json
when it declares any). The retry loads the fast tokenizer with every
engine-relevant behavior intact: the special tokens map, pad/eos ids, and
subsequent ``add_special_tokens`` calls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_BINDING_ERROR = "Input must be a List[Union[str, AddedToken]]"
_KNOWN_ADDED_TOKEN_FIELDS = {"single_word", "lstrip", "rstrip", "normalized", "special"}


def _read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_added_token(value: Any, added_token_cls) -> Any:
    """Converts a JSON dict entry into an ``AddedToken``; passes other values through."""
    if not isinstance(value, dict):
        return value
    fields = {k: v for k, v in value.items() if k in _KNOWN_ADDED_TOKEN_FIELDS}
    fields.setdefault("special", True)
    content = value.get("content")
    if content is None:
        return value
    try:
        return added_token_cls(content, **fields)
    except TypeError:
        # Unexpected flag set in a future config format; degrade gracefully.
        return value


def load_tokenizer(model_id_or_path: str, **kwargs):
    """Loads a tokenizer, retrying the transformers 5.2x binding TypeError.

    Args:
      model_id_or_path: Hub repo id or local directory.
      kwargs: Forwarded to ``AutoTokenizer.from_pretrained`` (an explicit
        ``added_tokens_decoder`` from the caller is merged into the retry).

    Returns:
      The loaded tokenizer (fast when available).

    Raises:
      Any error other than the specific binding TypeError, including errors
      from the retry attempt itself.
    """
    from transformers import AddedToken, AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id_or_path, **kwargs)
    except TypeError as exc:
        if _BINDING_ERROR not in str(exc):
            raise

    decoder: dict[int, Any] = {}
    for key, value in (kwargs.pop("added_tokens_decoder", None) or {}).items():
        decoder[int(key)] = _to_added_token(value, AddedToken)

    try:
        if os.path.isdir(model_id_or_path):
            config = _read_json(Path(model_id_or_path) / "tokenizer_config.json")
        else:
            from huggingface_hub import hf_hub_download

            config = _read_json(
                hf_hub_download(model_id_or_path, "tokenizer_config.json")
            )
        for key, value in (config.get("added_tokens_decoder") or {}).items():
            decoder[int(key)] = _to_added_token(value, AddedToken)
    except Exception:  # noqa: BLE001 - decoder merge is best-effort
        pass

    return AutoTokenizer.from_pretrained(
        model_id_or_path, **kwargs, added_tokens_decoder=decoder
    )
