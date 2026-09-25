from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def order_fields(value: dict[str, Any], preferred_order: Iterable[str]) -> dict[str, Any]:
    """Return ``value`` with known fields first and other fields preserved after.

    This is strictly presentation order.  ``canonical_json`` remains the only
    representation used for identity hashes.
    """
    preferred = set(preferred_order)
    return {
        **{key: value[key] for key in preferred_order if key in value},
        **{key: child for key, child in value.items() if key not in preferred},
    }


def write_json(path: Path, value: Any) -> None:
    """Write a human-readable JSON document.

    Hashing uses ``canonical_json`` explicitly, so presentation whitespace in
    standalone JSON artifacts never changes benchmark semantics.
    """
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, sort_keys: bool = False) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict[str, Any], *, sort_keys: bool = False) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
