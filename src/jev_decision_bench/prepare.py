from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

from .util import canonical_json, order_fields, sha256_bytes, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "jev-decision-bench/0.1"})
    with urlopen(request, timeout=30) as response:  # nosec B310: URL is pinned in tracked spec
        return response.read()


def _pipeline_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "working-tree"


def _load_spec(experiment: str) -> tuple[dict, dict]:
    if experiment != "banking77-choice-v0":
        raise ValueError(f"Unsupported experiment: {experiment}")
    directory = ROOT / "experiments" / "banking77-choice" / "v0"
    spec = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    rubric = json.loads((directory / spec["task"]["rubric_path"]).read_text(encoding="utf-8"))
    return spec, rubric


def prepare(experiment: str, artifacts_root: Path, downloader=_download) -> Path:
    """Create an immutable, Choice-only experiment package and return its path."""
    spec, rubric = _load_spec(experiment)
    dataset = spec["dataset"]
    source_bytes: dict[str, bytes] = {}
    for name, file_spec in dataset["files"].items():
        content = downloader(file_spec["url"])
        actual_hash = sha256_bytes(content)
        if actual_hash != file_spec["sha256"]:
            raise ValueError(f"Hash mismatch for {name}: expected {file_spec['sha256']}, got {actual_hash}")
        source_bytes[name] = content

    categories = json.loads(source_bytes["categories.json"].decode("utf-8"))
    criteria = rubric["criteria"]
    if categories != list(criteria):
        raise ValueError("Rubric labels must match BANKING77 categories exactly and in source order")

    rows = list(csv.DictReader(io.StringIO(source_bytes["test.csv"].decode("utf-8"))))
    if len(rows) != dataset["expected_row_count"]:
        raise ValueError(f"Unexpected test row count: {len(rows)}")
    if any(set(row) != {"text", "category"} or row["category"] not in criteria for row in rows):
        raise ValueError("BANKING77 source rows do not match the expected schema")

    records = [
        {
            "decision_id": f"banking77-choice-v0:test:{index:04d}",
            "state": {"query": row["text"]},
            "gold": row["category"],
            "source_id": f"banking77:test:{index:04d}",
            "task_type": "choice",
            "question": spec["task"]["question"],
            "criteria": criteria,
            "metadata": {"dataset_row_index": index, "record_set": "choice"},
        }
        for index, row in enumerate(rows)
    ]
    records_bytes = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n" for record in records
    ).encode("utf-8")
    base_manifest = {
        "experiment_id": spec["experiment_id"],
        "experiment_version": spec["experiment_version"],
        "package_format_version": "pretty-json-v2",
        "pipeline_version": _pipeline_version(),
        "dataset": {
            "name": dataset["name"],
            "source_repository": dataset["source_repository"],
            "revision": dataset["revision"],
            "split": "banking_data/test.csv",
            "row_count": len(rows),
            "license": dataset["license"],
            "file_hashes": {name: sha256_bytes(content) for name, content in source_bytes.items()},
        },
        "rubric_version": rubric["rubric_version"],
        "record_sets": ["choice"],
        "validation_contract_version": spec["validation_contract_version"],
        "evaluator_version": spec["evaluator_version"],
        "threshold_policy_id": spec["threshold_policy_id"],
        "records_sha256": sha256_bytes(records_bytes),
    }
    package_hash = sha256_bytes(canonical_json(base_manifest).encode("utf-8") + records_bytes)
    manifest = order_fields(
        {**base_manifest, "experiment_package_hash": package_hash},
        (
            "experiment_id",
            "experiment_version",
            "experiment_package_hash",
            "package_format_version",
            "pipeline_version",
            "dataset",
            "rubric_version",
            "record_sets",
            "validation_contract_version",
            "evaluator_version",
            "threshold_policy_id",
            "records_sha256",
        ),
    )
    package_dir = artifacts_root / "experiments" / spec["experiment_id"] / spec["experiment_version"] / package_hash
    if package_dir.exists():
        raise FileExistsError(f"Experiment package already exists: {package_dir}")
    package_dir.mkdir(parents=True)
    source_dir = package_dir / "source"
    source_dir.mkdir()
    for name, content in source_bytes.items():
        (source_dir / name).write_bytes(content)
    write_json(package_dir / "rubric.json", rubric)
    write_jsonl(package_dir / "records.choice.jsonl", records, sort_keys=False)
    if sha256_file(package_dir / "records.choice.jsonl") != manifest["records_sha256"]:
        raise RuntimeError("Written record hash differs from compiled record hash")
    write_json(package_dir / "experiment-manifest.json", manifest)
    return package_dir
