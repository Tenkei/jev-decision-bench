from __future__ import annotations

import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from .prepare import ROOT
from .task_contracts import contract_for_manifest
from .util import canonical_json, read_json, read_jsonl, sha256_bytes, write_json


_EXPORT_FORMAT = "jev-decision-bench-result-export-v1"
_REDACTED = "[REDACTED]"
_SENSITIVE_EXACT_KEYS = {"api_key", "apikey", "authorization", "auth", "bearer", "access_token", "token", "secret", "password", "credential", "cookie", "headers", "http_headers"}
_SENSITIVE_SUFFIXES = ("_api_key", "_token", "_secret", "_password", "_credential")


def _comparison_path(path: Path) -> Path:
    comparison = path / "comparison.json" if path.is_dir() else path
    if not comparison.is_file():
        raise ValueError(f"Comparison JSON does not exist: {comparison}")
    return comparison


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized != "api_key_env" and (normalized in _SENSITIVE_EXACT_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES))


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _REDACTED if _is_sensitive_key(str(key)) else _sanitize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize(child) for child in value]
    return value


def _safe_component(value: str, description: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Unsafe {description}: {value!r}")
    return value


def _evaluation_config(artifacts_root: Path, run_id: str, expected_hash: str) -> dict[str, Any]:
    root = artifacts_root / "evaluations" / run_id
    if not root.is_dir():
        raise ValueError(f"No evaluation artifacts found for run: {run_id}")
    for path in sorted(root.glob("*/evaluation-config.json")):
        config = read_json(path)
        if sha256_bytes(canonical_json(config).encode("utf-8")) == expected_hash:
            return config
    raise ValueError(f"No evaluation artifact for run {run_id} matches comparison policy {expected_hash}")


def _archive_bytes(archive: tarfile.TarFile, path: str, data: bytes, *, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(path)
    info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, mode, 0, 0
    archive.addfile(info, io.BytesIO(data))


def _archive_json(archive: tarfile.TarFile, path: str, value: Any) -> None:
    _archive_bytes(archive, path, (canonical_json(value) + "\n").encode("utf-8"))


def _archive_file(archive: tarfile.TarFile, path: str, source: Path) -> None:
    data = source.read_bytes()
    _archive_bytes(archive, path, data)


def _sanitized_jsonl(path: Path) -> bytes:
    return b"".join((canonical_json(_sanitize(row)) + "\n").encode("utf-8") for row in read_jsonl(path))


def _export_readme(comparison: dict[str, Any], run_manifests: list[dict[str, Any]]) -> str:
    lines = [
        "# JEV Decision Bench result export",
        "",
        f"Comparison: `{comparison['comparison_id']}`",
        f"Experiment package: `{comparison['experiment_package_hash']}`",
        "",
        "## Contents",
        "",
        "- [Published summary](summary/README.md) — human-readable comparison report.",
        "- [Comparison data](summary/comparison.json) — aggregate metrics and baseline deltas.",
        "- [Experiment manifest](package/experiment-manifest.json) — immutable package provenance.",
        "- [Evaluation policy](evaluation-config.json) — scoring-policy snapshot.",
        "- [Experiment documentation](experiment/README.md) — dataset and decision contract.",
        "- `runs/` — sanitized run manifests and normalized predictions for every compared run.",
        "- `package/records.*.jsonl` — prepared decision records needed to rescore the saved predictions.",
        "",
        "Run `./reproduce.sh` after unpacking to rescore and compare without calling a model provider.",
        "Raw provider events, logs, credentials, API-key values, and downloaded source copies are excluded.",
        "",
        "## Included runs",
        "",
        "| Model | Run manifest |",
        "| --- | --- |",
    ]
    for manifest in run_manifests:
        model = str(manifest.get("model_config", {}).get("model_id", "unknown"))
        run_id = str(manifest["run_id"])
        lines.append(f"| `{model}` | [runs/{run_id}/run-manifest.json](runs/{run_id}/run-manifest.json) |")
    return "\n".join(lines) + "\n"


def _portable_report(report: str, package_manifest: dict[str, Any], experiment_link: str) -> str:
    experiment_id, version = package_manifest.get("experiment_id"), package_manifest.get("experiment_version")
    if not isinstance(experiment_id, str) or not isinstance(version, str):
        return report
    source_link = f"../../../experiments/{experiment_id}/{version}/README.md"
    return report.replace(source_link, experiment_link)


def _reproduce_script(run_ids: list[str], comparison: dict[str, Any]) -> str:
    baseline, candidates = run_ids[0], run_ids[1:]
    candidate_args = " ".join(f'"$ROOT/runs/{run_id}"' for run_id in candidates)
    score_lines = "\n".join(
        f'"$BENCH" score --package "$ROOT/package" --run "$ROOT/runs/{run_id}" --evaluation "$ROOT/evaluation-config.json" --artifacts-root "$ROOT"'
        for run_id in run_ids
    )
    return (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        "BENCH=${JEV_DECISION_BENCH:-jev-decision-bench}\n"
        'ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n\n'
        f"{score_lines}\n"
        '"$BENCH" compare --package "$ROOT/package" --baseline '
        f'"$ROOT/runs/{baseline}" --runs {candidate_args} --evaluation "$ROOT/evaluation-config.json" --artifacts-root "$ROOT"\n'
    )


def _write_summary(summary_dir: Path, *, report: str, comparison: dict[str, Any], package_manifest: dict[str, Any], evaluation: dict[str, Any], run_manifests: list[dict[str, Any]], overwrite: bool) -> None:
    if summary_dir.exists():
        if not overwrite:
            raise ValueError(f"Published result summary already exists: {summary_dir}")
        shutil.rmtree(summary_dir)
    (summary_dir / "run-manifests").mkdir(parents=True)
    (summary_dir / "README.md").write_text(report, encoding="utf-8")
    write_json(summary_dir / "comparison.json", _sanitize(comparison))
    write_json(summary_dir / "experiment-manifest.json", _sanitize(package_manifest))
    write_json(summary_dir / "evaluation-config.json", evaluation)
    for manifest in run_manifests:
        write_json(summary_dir / "run-manifests" / f"{manifest['run_id']}.json", manifest)


def export_results(package_dir: Path, comparison_path: Path, artifacts_root: Path, results_root: Path, output_dir: Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Publish a Git-friendly summary and an auditable, sanitized result archive."""
    package_manifest = read_json(package_dir / "experiment-manifest.json")
    comparison_file = _comparison_path(comparison_path)
    comparison = read_json(comparison_file)
    package_hash = package_manifest.get("experiment_package_hash")
    if not isinstance(package_hash, str) or comparison.get("experiment_package_hash") != package_hash:
        raise ValueError("Comparison and experiment package hashes do not match")
    comparison_id = comparison.get("comparison_id")
    evaluation_hash = comparison.get("evaluation_config_sha256")
    if not isinstance(comparison_id, str):
        raise ValueError("Comparison lacks a comparison ID")
    comparison_id = _safe_component(comparison_id, "comparison ID")
    if not isinstance(evaluation_hash, str):
        raise ValueError("Export requires a scored comparison with an evaluation policy")

    rows = [comparison.get("baseline"), *(comparison.get("runs") or [])]
    if not all(isinstance(row, dict) and isinstance(row.get("run_id"), str) for row in rows):
        raise ValueError("Comparison lacks one or more run IDs")
    run_ids = [_safe_component(str(row["run_id"]), "run ID") for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Comparison repeats a run ID")

    run_manifests: list[dict[str, Any]] = []
    prediction_paths: dict[str, Path] = {}
    for run_id in run_ids:
        manifest = read_json(artifacts_root / "runs" / run_id / "run-manifest.json")
        if manifest.get("run_id") != run_id:
            raise ValueError(f"Run manifest ID does not match its directory: {run_id}")
        if manifest.get("experiment_package_hash") != package_hash:
            raise ValueError(f"Run manifest uses a different experiment package: {run_id}")
        run_manifests.append(_sanitize(manifest))
        prediction_path = artifacts_root / "runs" / run_id / "predictions.jsonl"
        if not prediction_path.is_file():
            raise ValueError(f"Run predictions do not exist: {run_id}")
        prediction_paths[run_id] = prediction_path

    evaluation = _sanitize(_evaluation_config(artifacts_root, run_ids[0], evaluation_hash))
    report_path = comparison_file.parent / "README.md"
    report = report_path.read_text(encoding="utf-8") if report_path.is_file() else "# Comparison report\n\nSee [comparison.json](comparison.json).\n"
    experiment_readme = ROOT / "experiments" / str(package_manifest.get("experiment_id", "")) / str(package_manifest.get("experiment_version", "")) / "README.md"
    experiment_doc = experiment_readme.read_text(encoding="utf-8") if experiment_readme.is_file() else "# Experiment documentation\n\nNo tracked experiment README was available when this archive was exported.\n"

    contract = contract_for_manifest(package_manifest)
    record_path = package_dir / f"records.{contract.record_set}.jsonl"
    if not record_path.is_file():
        raise ValueError(f"Experiment package is missing prepared records: {record_path}")
    package_files = [record_path, *(path for path in (package_dir / "preflight.json", package_dir / "rubric.json") if path.is_file())]

    summary_dir = results_root / comparison_id
    summary_report = _portable_report(report, package_manifest, f"../../experiments/{package_manifest['experiment_id']}/{package_manifest['experiment_version']}/README.md")
    _write_summary(summary_dir, report=summary_report, comparison=comparison, package_manifest=package_manifest, evaluation=evaluation, run_manifests=run_manifests, overwrite=overwrite)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{comparison_id}.tar.gz"
    if output.exists() and not overwrite:
        raise ValueError(f"Export already exists: {output}")
    if output.exists():
        output.unlink()
    root = f"jev-decision-bench-export-{comparison_id}"
    export_manifest = {
        "export_format": _EXPORT_FORMAT,
        "comparison_id": comparison_id,
        "experiment_package_hash": package_hash,
        "evaluation_config_sha256": evaluation_hash,
        "run_ids": run_ids,
        "sanitization": "Sensitive configuration fields are replaced with [REDACTED]; api_key_env names are retained.",
        "reproducibility": "Includes normalized predictions and prepared decision records; raw provider events and source downloads are excluded.",
    }
    with output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
        _archive_bytes(archive, f"{root}/README.md", _export_readme(comparison, run_manifests).encode("utf-8"))
        _archive_bytes(archive, f"{root}/reproduce.sh", _reproduce_script(run_ids, comparison).encode("utf-8"), mode=0o755)
        _archive_bytes(archive, f"{root}/summary/README.md", _portable_report(report, package_manifest, "../experiment/README.md").encode("utf-8"))
        _archive_json(archive, f"{root}/summary/comparison.json", _sanitize(comparison))
        _archive_json(archive, f"{root}/summary/experiment-manifest.json", _sanitize(package_manifest))
        _archive_json(archive, f"{root}/summary/evaluation-config.json", evaluation)
        for manifest in run_manifests:
            _archive_json(archive, f"{root}/summary/run-manifests/{manifest['run_id']}.json", manifest)
        _archive_json(archive, f"{root}/package/experiment-manifest.json", _sanitize(package_manifest))
        for path in package_files:
            _archive_file(archive, f"{root}/package/{path.name}", path)
        _archive_json(archive, f"{root}/evaluation-config.json", evaluation)
        _archive_bytes(archive, f"{root}/experiment/README.md", experiment_doc.encode("utf-8"))
        for manifest in run_manifests:
            run_id = str(manifest["run_id"])
            _archive_json(archive, f"{root}/runs/{run_id}/run-manifest.json", manifest)
            _archive_bytes(archive, f"{root}/runs/{run_id}/predictions.jsonl", _sanitized_jsonl(prediction_paths[run_id]))
        _archive_json(archive, f"{root}/export-manifest.json", export_manifest)
    return summary_dir, output
