from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .scoring import evaluation_output_dir, resolve_evaluation_config
from .util import canonical_json, order_fields, read_json, sha256_bytes, write_json


_ROW_ORDER = (
    "run_id",
    "provider",
    "model_id",
    "model_revision",
    "adapter_version",
    "coverage",
    "accuracy_on_valid",
    "macro_f1_on_valid",
    "top_label_brier_on_valid",
    "top_label_ece_10_bins_on_valid",
    "p50_latency_ms",
    "p95_latency_ms",
    "cost_per_attempted_usd",
)


def _run_row(run_dir: Path, experiment_package_hash: str, evaluation: dict[str, Any] | None, artifacts_root: Path) -> dict[str, Any]:
    run_manifest = read_json(run_dir / "run-manifest.json")
    if run_manifest.get("status") != "completed":
        raise ValueError(f"Run is not completed: {run_dir}")
    if run_manifest.get("experiment_package_hash") != experiment_package_hash:
        raise ValueError(f"Run uses a different experiment package: {run_dir}")
    if evaluation is None:
        # Read-only compatibility for evaluation artifacts made before policy
        # configurations were introduced.
        metrics = read_json(run_dir / "evaluation.json")
    else:
        scores_path = evaluation_output_dir(run_manifest, evaluation, artifacts_root) / "scores.json"
        if not scores_path.is_file():
            raise ValueError(f"Run has not been scored with this evaluation policy: {run_dir}")
        metrics = read_json(scores_path)
    if metrics.get("experiment_package_hash") != experiment_package_hash:
        raise ValueError(f"Evaluation uses a different experiment package: {run_dir}")
    return order_fields(
        {
            "run_id": run_manifest["run_id"],
            "provider": run_manifest["model_config"]["provider"],
            "model_id": run_manifest["model_config"]["model_id"],
            "model_revision": run_manifest["model_config"].get("model_revision"),
            "adapter_version": run_manifest["adapter_version"],
            "coverage": metrics["coverage"],
            "accuracy_on_valid": metrics["accuracy_on_valid"],
            "macro_f1_on_valid": metrics["macro_f1_on_valid"],
            "top_label_brier_on_valid": metrics["top_label_brier_on_valid"],
            # Current score artifacts use the metric's canonical name. Retain
            # the former spelling for legacy evaluation.json artifacts.
            "top_label_ece_10_bins_on_valid": (
                metrics["top_label_ece_10_bins"]
                if "top_label_ece_10_bins" in metrics
                else metrics["top_label_ece_10_bins_on_valid"]
            ),
            "p50_latency_ms": metrics["latency_ms"]["p50"],
            "p95_latency_ms": metrics["latency_ms"]["p95"],
            "cost_per_attempted_usd": metrics["cost_usd"]["per_attempted"],
        },
        _ROW_ORDER,
    )


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if not isinstance(candidate, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return candidate - baseline


def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if not isinstance(candidate, (int, float)) or not isinstance(baseline, (int, float)) or baseline == 0:
        return None
    return candidate / baseline


def _relative_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Return deltas or ratios without changing either run's absolute metrics.

    Deltas are candidate minus baseline. A negative Brier/ECE delta is better;
    a latency or cost ratio below one is better.
    """
    return order_fields(
        {
            "baseline_run_id": baseline["run_id"],
            "coverage_delta": _delta(candidate["coverage"], baseline["coverage"]),
            "accuracy_on_valid_delta": _delta(candidate["accuracy_on_valid"], baseline["accuracy_on_valid"]),
            "macro_f1_on_valid_delta": _delta(candidate["macro_f1_on_valid"], baseline["macro_f1_on_valid"]),
            "top_label_brier_on_valid_delta": _delta(
                candidate["top_label_brier_on_valid"], baseline["top_label_brier_on_valid"]
            ),
            "top_label_ece_10_bins_on_valid_delta": _delta(
                candidate["top_label_ece_10_bins_on_valid"], baseline["top_label_ece_10_bins_on_valid"]
            ),
            "p50_latency_ratio": _ratio(candidate["p50_latency_ms"], baseline["p50_latency_ms"]),
            "p95_latency_ratio": _ratio(candidate["p95_latency_ms"], baseline["p95_latency_ms"]),
            "cost_per_attempted_ratio": _ratio(
                candidate["cost_per_attempted_usd"], baseline["cost_per_attempted_usd"]
            ),
        },
        (
            "baseline_run_id",
            "coverage_delta",
            "accuracy_on_valid_delta",
            "macro_f1_on_valid_delta",
            "top_label_brier_on_valid_delta",
            "top_label_ece_10_bins_on_valid_delta",
            "p50_latency_ratio",
            "p95_latency_ratio",
            "cost_per_attempted_ratio",
        ),
    )


def compare(
    package_dir: Path,
    baseline_run_dir: Path,
    run_dirs: list[Path],
    artifacts_root: Path,
    evaluation_config_path: Path | None = None,
) -> Path:
    if not run_dirs:
        raise ValueError("Comparison requires at least one non-baseline run")
    if baseline_run_dir.resolve() in {run_dir.resolve() for run_dir in run_dirs}:
        raise ValueError("Baseline run must not also appear in --runs")

    experiment = read_json(package_dir / "experiment-manifest.json")
    package_hash = experiment["experiment_package_hash"]
    evaluation = resolve_evaluation_config(experiment, evaluation_config_path) if experiment.get("experiment_id") else None
    evaluation_hash = sha256_bytes(canonical_json(evaluation).encode("utf-8")) if evaluation else None
    baseline = _run_row(baseline_run_dir, package_hash, evaluation, artifacts_root)
    rows = []
    for run_dir in run_dirs:
        row = _run_row(run_dir, package_hash, evaluation, artifacts_root)
        row["relative_to_baseline"] = _relative_to_baseline(row, baseline)
        rows.append(order_fields(row, (*_ROW_ORDER, "relative_to_baseline")))

    comparison_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    output_dir = artifacts_root / "comparisons" / comparison_id
    output_dir.mkdir(parents=True, exist_ok=False)
    result = order_fields(
        {
            "comparison_id": comparison_id,
            "experiment_package_hash": package_hash,
            "evaluation_config_sha256": evaluation_hash,
            "baseline": baseline,
            "runs": rows,
        },
        ("comparison_id", "experiment_package_hash", "evaluation_config_sha256", "baseline", "runs"),
    )
    write_json(output_dir / "comparison.json", result)
    return output_dir
