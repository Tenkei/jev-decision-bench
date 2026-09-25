from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from .util import order_fields, read_json, read_jsonl, write_json


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(values)) - 1)
    return values[index]


def _ece(confidences: list[float], correct: list[int], bins: int = 10) -> float | None:
    if not confidences:
        return None
    total = len(confidences)
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [position for position, value in enumerate(confidences) if low <= value < high or (index == bins - 1 and value == 1)]
        if members:
            average_confidence = sum(confidences[position] for position in members) / len(members)
            average_accuracy = sum(correct[position] for position in members) / len(members)
            ece += len(members) / total * abs(average_confidence - average_accuracy)
    return ece


def score(run_dir: Path, package_dir: Path) -> dict[str, Any]:
    manifest = read_json(package_dir / "experiment-manifest.json")
    run_manifest = read_json(run_dir / "run-manifest.json")
    if run_manifest["experiment_package_hash"] != manifest["experiment_package_hash"]:
        raise ValueError("Run and experiment package hashes do not match")
    records = {record["decision_id"]: record for record in read_jsonl(package_dir / "records.choice.jsonl")}
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    by_id = {prediction["decision_id"]: prediction for prediction in predictions}
    if set(by_id) != set(records):
        raise ValueError("Run does not have one terminal prediction per decision record")

    valid = [prediction for prediction in predictions if prediction["status"] == "valid"]
    labels = list(next(iter(records.values()))["criteria"])
    correct = [int(prediction["answer"] == records[prediction["decision_id"]]["gold"]) for prediction in valid]
    confidences = [float(prediction["selected_probability"]) for prediction in valid]
    confusion: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for prediction in valid:
        confusion[records[prediction["decision_id"]]["gold"]][prediction["answer"]] += 1
    f1_values: list[float] = []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(count for predicted, count in confusion[label].items() if predicted != label)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append((2 * true_positive / denominator) if denominator else 0.0)
    known_costs = [float(prediction["cost_usd"]) for prediction in predictions if isinstance(prediction["cost_usd"], (int, float))]
    latencies = [float(prediction["timing_ms"]) for prediction in predictions]
    total = len(records)
    metrics = {
        "run_id": run_manifest["run_id"],
        "experiment_package_hash": manifest["experiment_package_hash"],
        "total_records": total,
        "status_counts": dict(Counter(prediction["status"] for prediction in predictions)),
        "valid_predictions": len(valid),
        "coverage": len(valid) / total if total else None,
        "valid_output_rate": len(valid) / total if total else None,
        "accuracy_on_valid": sum(correct) / len(correct) if correct else None,
        "macro_f1_on_valid": sum(f1_values) / len(f1_values) if valid else None,
        "top_label_brier_on_valid": sum((confidence - outcome) ** 2 for confidence, outcome in zip(confidences, correct)) / len(valid) if valid else None,
        "top_label_ece_10_bins_on_valid": _ece(confidences, correct),
        "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        "cost_usd": {
            "known_total": sum(known_costs) if known_costs else None,
            "known_cost_record_count": len(known_costs),
            "per_attempted": sum(known_costs) / total if known_costs else None,
            "per_valid": sum(known_costs) / len(valid) if known_costs and valid else None,
        },
        "confusion_matrix": {label: dict(confusion[label]) for label in labels},
    }
    for legacy_name in ("metrics.json", "report.md"):
        (run_dir / legacy_name).unlink(missing_ok=True)
    write_json(
        run_dir / "evaluation.json",
        order_fields(
            metrics,
            (
                "run_id",
                "experiment_package_hash",
                "total_records",
                "status_counts",
                "valid_predictions",
                "coverage",
                "valid_output_rate",
                "accuracy_on_valid",
                "macro_f1_on_valid",
                "top_label_brier_on_valid",
                "top_label_ece_10_bins_on_valid",
                "latency_ms",
                "cost_usd",
                "confusion_matrix",
            ),
        ),
    )
    return metrics
