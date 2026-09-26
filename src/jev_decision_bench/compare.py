from __future__ import annotations

import math
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .scoring import evaluation_output_dir, resolve_evaluation_config
from .util import canonical_json, order_fields, read_json, read_jsonl, sha256_bytes, write_json


_QUALITY_FIELDS = (
    "accuracy_on_valid", "exact_tier_accuracy_on_valid", "ordinal_mae_on_valid", "macro_f1_on_valid", "precision_on_valid", "recall_on_valid", "auroc_on_valid",
    "true_probability_brier_on_valid", "true_probability_ece_10_bins", "top_label_brier_on_valid", "top_label_ece_10_bins",
)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(percentile / 100 * len(values)) - 1)]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _usage_value(usage: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = usage.get(name)
        if _is_number(value):
            return float(value)
    return None


def _usage_detail(usage: dict[str, Any], detail_name: str, *parents: str) -> float | None:
    for parent in parents:
        details = usage.get(parent)
        if isinstance(details, dict) and _is_number(details.get(detail_name)):
            return float(details[detail_name])
    return None


def _sum_usage(values: list[dict[str, Any]], extractor) -> tuple[float | None, int]:
    reported = [value for usage in values if (value := extractor(usage)) is not None]
    return (sum(reported) if reported else None, len(reported))


def _token_usage(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(predictions)
    usages = [prediction["provider_usage"] for prediction in predictions if isinstance(prediction.get("provider_usage"), dict)]
    input_tokens, input_count = _sum_usage(usages, lambda usage: _usage_value(usage, "input_tokens", "prompt_tokens"))
    output_tokens, output_count = _sum_usage(usages, lambda usage: _usage_value(usage, "output_tokens", "completion_tokens"))
    cached_input_tokens, cached_count = _sum_usage(usages, lambda usage: _usage_detail(usage, "cached_tokens", "input_tokens_details", "prompt_tokens_details"))
    reasoning_output_tokens, reasoning_count = _sum_usage(usages, lambda usage: _usage_detail(usage, "reasoning_tokens", "output_tokens_details"))
    total = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
    return order_fields(
        {
            "usage_record_count": len(usages), "usage_coverage": _rate(len(usages), attempted),
            "input_token_record_count": input_count, "input_tokens": input_tokens,
            "input_tokens_per_attempted": input_tokens / attempted if input_tokens is not None and attempted else None,
            "output_token_record_count": output_count, "output_tokens": output_tokens,
            "output_tokens_per_attempted": output_tokens / attempted if output_tokens is not None and attempted else None,
            "cached_input_token_record_count": cached_count, "cached_input_tokens": cached_input_tokens,
            "reasoning_output_token_record_count": reasoning_count, "reasoning_output_tokens": reasoning_output_tokens,
            "reported_total_tokens": total,
            "reported_total_tokens_per_attempted": total / attempted if total is not None and attempted else None,
        },
        (
            "usage_record_count", "usage_coverage", "input_token_record_count", "input_tokens", "input_tokens_per_attempted",
            "output_token_record_count", "output_tokens", "output_tokens_per_attempted",
            "cached_input_token_record_count", "cached_input_tokens", "reasoning_output_token_record_count",
            "reasoning_output_tokens", "reported_total_tokens", "reported_total_tokens_per_attempted",
        ),
    )


def _provider_metrics(run_manifest: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(prediction.get("status")) for prediction in predictions)
    attempted = len(predictions)
    configured_total = run_manifest.get("counts", {}).get("total")
    total = configured_total if isinstance(configured_total, int) else attempted
    valid, invalid, provider_error = counts["valid"], counts["invalid"], counts["provider_error"]
    return order_fields(
        {
            "total_records": total, "attempted_records": attempted, "completion_rate": _rate(attempted, total),
            "status_counts": {"valid": valid, "invalid": invalid, "provider_error": provider_error},
            "provider_success_rate": _rate(valid + invalid, attempted), "valid_output_rate": _rate(valid, attempted),
        },
        ("total_records", "attempted_records", "completion_rate", "status_counts", "provider_success_rate", "valid_output_rate"),
    )


def _model_performance(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(predictions)
    latencies = [float(prediction["timing_ms"]) for prediction in predictions if _is_number(prediction.get("timing_ms"))]
    costs = [float(prediction["cost_usd"]) for prediction in predictions if _is_number(prediction.get("cost_usd"))]
    known_total = sum(costs) if costs else None
    return order_fields(
        {
            "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
            "token_usage": _token_usage(predictions),
            "cost_usd": {"known_total": known_total, "known_cost_record_count": len(costs), "known_cost_coverage": _rate(len(costs), attempted), "per_attempted": known_total / attempted if known_total is not None and attempted else None},
        },
        ("latency_ms", "token_usage", "cost_usd"),
    )


def _run_row(run_dir: Path, experiment_package_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run_manifest = read_json(run_dir / "run-manifest.json")
    if run_manifest.get("status") != "completed":
        raise ValueError(f"Run is not completed: {run_dir}")
    if run_manifest.get("experiment_package_hash") != experiment_package_hash:
        raise ValueError(f"Run uses a different experiment package: {run_dir}")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    return order_fields(
        {
            "run_id": run_manifest["run_id"], "provider": run_manifest["model_config"]["provider"],
            "model_id": run_manifest["model_config"]["model_id"], "model_revision": run_manifest["model_config"].get("model_revision"),
            "adapter_version": run_manifest["adapter_version"], "provider_metrics": _provider_metrics(run_manifest, predictions),
            "model_performance": _model_performance(predictions), "model_quality": None,
        },
        ("run_id", "provider", "model_id", "model_revision", "adapter_version", "provider_metrics", "model_performance", "model_quality"),
    ), run_manifest


def _load_quality(run_manifest: dict[str, Any], experiment_package_hash: str, evaluation: dict[str, Any], artifacts_root: Path) -> dict[str, Any] | None:
    scores_path = evaluation_output_dir(run_manifest, evaluation, artifacts_root) / "scores.json"
    if not scores_path.is_file():
        return None
    scores = read_json(scores_path)
    if scores.get("experiment_package_hash") != experiment_package_hash:
        raise ValueError(f"Evaluation uses a different experiment package: {scores_path}")
    metrics = {field: scores[field] for field in _QUALITY_FIELDS if field in scores}
    return {"metrics": order_fields(metrics, _QUALITY_FIELDS)}


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if not isinstance(candidate, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return candidate - baseline


def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if not isinstance(candidate, (int, float)) or not isinstance(baseline, (int, float)) or baseline == 0:
        return None
    return candidate / baseline


def _relative_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_provider, baseline_provider = candidate["provider_metrics"], baseline["provider_metrics"]
    candidate_performance, baseline_performance = candidate["model_performance"], baseline["model_performance"]
    candidate_tokens, baseline_tokens = candidate_performance["token_usage"], baseline_performance["token_usage"]
    quality = None
    if candidate["model_quality"] is not None and baseline["model_quality"] is not None:
        candidate_quality, baseline_quality = candidate["model_quality"]["metrics"], baseline["model_quality"]["metrics"]
        quality = {f"{field}_delta": _delta(candidate_quality[field], baseline_quality[field]) for field in _QUALITY_FIELDS if field in candidate_quality and field in baseline_quality}
    return order_fields(
        {
            "baseline_run_id": baseline["run_id"],
            "provider_metrics": {
                "completion_rate_delta": _delta(candidate_provider["completion_rate"], baseline_provider["completion_rate"]),
                "provider_success_rate_delta": _delta(candidate_provider["provider_success_rate"], baseline_provider["provider_success_rate"]),
                "valid_output_rate_delta": _delta(candidate_provider["valid_output_rate"], baseline_provider["valid_output_rate"]),
            },
            "model_performance": {
                "p50_latency_ratio": _ratio(candidate_performance["latency_ms"]["p50"], baseline_performance["latency_ms"]["p50"]),
                "p95_latency_ratio": _ratio(candidate_performance["latency_ms"]["p95"], baseline_performance["latency_ms"]["p95"]),
                "input_tokens_per_attempted_ratio": _ratio(candidate_tokens["input_tokens_per_attempted"], baseline_tokens["input_tokens_per_attempted"]),
                "output_tokens_per_attempted_ratio": _ratio(candidate_tokens["output_tokens_per_attempted"], baseline_tokens["output_tokens_per_attempted"]),
                "reported_total_tokens_per_attempted_ratio": _ratio(candidate_tokens["reported_total_tokens_per_attempted"], baseline_tokens["reported_total_tokens_per_attempted"]),
                "cost_per_attempted_ratio": _ratio(candidate_performance["cost_usd"]["per_attempted"], baseline_performance["cost_usd"]["per_attempted"]),
            },
            "model_quality": quality,
        },
        ("baseline_run_id", "provider_metrics", "model_performance", "model_quality"),
    )


def _default_evaluation(manifest: dict[str, Any]) -> dict[str, Any] | None:
    if not manifest.get("experiment_id"):
        return None
    try:
        return resolve_evaluation_config(manifest)
    except ValueError:
        return None


def compare(package_dir: Path, baseline_run_dir: Path, run_dirs: list[Path], artifacts_root: Path, evaluation_config_path: Path | None = None) -> Path:
    if not run_dirs:
        raise ValueError("Comparison requires at least one non-baseline run")
    if baseline_run_dir.resolve() in {run_dir.resolve() for run_dir in run_dirs}:
        raise ValueError("Baseline run must not also appear in --runs")
    experiment = read_json(package_dir / "experiment-manifest.json")
    package_hash = experiment["experiment_package_hash"]
    baseline, baseline_manifest = _run_row(baseline_run_dir, package_hash)
    candidates_with_manifests = [_run_row(run_dir, package_hash) for run_dir in run_dirs]
    evaluation = resolve_evaluation_config(experiment, evaluation_config_path) if evaluation_config_path else _default_evaluation(experiment)
    quality_available = evaluation is not None
    if evaluation is not None:
        all_rows = [(baseline, baseline_manifest), *candidates_with_manifests]
        quality = [_load_quality(run_manifest, package_hash, evaluation, artifacts_root) for _, run_manifest in all_rows]
        quality_available = all(item is not None for item in quality)
        if evaluation_config_path is not None and not quality_available:
            raise ValueError("Every compared run must be scored with the requested evaluation policy")
        if quality_available:
            for (row, _), item in zip(all_rows, quality):
                row["model_quality"] = item
    if not quality_available:
        evaluation = None
    rows = []
    for row, _ in candidates_with_manifests:
        row["relative_to_baseline"] = _relative_to_baseline(row, baseline)
        rows.append(row)
    evaluation_hash = sha256_bytes(canonical_json(evaluation).encode("utf-8")) if evaluation else None
    comparison_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    output_dir = artifacts_root / "comparisons" / comparison_id
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "comparison.json", order_fields({"comparison_id": comparison_id, "experiment_package_hash": package_hash, "evaluation_config_sha256": evaluation_hash, "baseline": baseline, "runs": rows}, ("comparison_id", "experiment_package_hash", "evaluation_config_sha256", "baseline", "runs")))
    return output_dir
