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


def _auroc(probabilities: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(enumerate(probabilities), key=lambda item: item[1])
    ranks = [0.0] * len(probabilities)
    position = 0
    while position < len(ordered):
        end = position
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[position][1]:
            end += 1
        average_rank = (position + 1 + end + 1) / 2
        for index in range(position, end + 1):
            ranks[ordered[index][0]] = average_rank
        position = end + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _common_metrics(
    run_manifest: dict[str, Any],
    manifest: dict[str, Any],
    predictions: list[dict[str, Any]],
    valid: list[dict[str, Any]],
) -> dict[str, Any]:
    known_costs = [float(prediction["cost_usd"]) for prediction in predictions if isinstance(prediction["cost_usd"], (int, float))]
    latencies = [float(prediction["timing_ms"]) for prediction in predictions]
    total = len(predictions)
    return {
        "run_id": run_manifest["run_id"],
        "experiment_package_hash": manifest["experiment_package_hash"],
        "total_records": total,
        "status_counts": dict(Counter(prediction["status"] for prediction in predictions)),
        "valid_predictions": len(valid),
        "coverage": len(valid) / total if total else None,
        "valid_output_rate": len(valid) / total if total else None,
        "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        "cost_usd": {
            "known_total": sum(known_costs) if known_costs else None,
            "known_cost_record_count": len(known_costs),
            "per_attempted": sum(known_costs) / total if known_costs else None,
            "per_valid": sum(known_costs) / len(valid) if known_costs and valid else None,
        },
    }


def _save_evaluation(run_dir: Path, metrics: dict[str, Any]) -> None:
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
                "precision_on_valid",
                "recall_on_valid",
                "auroc_on_valid",
                "true_probability_brier_on_valid",
                "true_probability_ece_10_bins_on_valid",
                "top_label_brier_on_valid",
                "top_label_ece_10_bins_on_valid",
                "latency_ms",
                "cost_usd",
                "confusion_matrix",
                "functionality_slices",
            ),
        ),
    )


def _score_noul(
    manifest: dict[str, Any],
    run_manifest: dict[str, Any],
    records: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [prediction for prediction in predictions if prediction["status"] == "valid"]
    gold = [int(records[prediction["decision_id"]]["gold"]) for prediction in valid]
    answers = [int(prediction["answer"]) for prediction in valid]
    probabilities = [float(prediction["positive_probability"]) for prediction in valid]
    true_positive = sum(answer and label for answer, label in zip(answers, gold))
    false_positive = sum(answer and not label for answer, label in zip(answers, gold))
    false_negative = sum(not answer and label for answer, label in zip(answers, gold))
    true_negative = sum(not answer and not label for answer, label in zip(answers, gold))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None

    def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true_positive + false_positive + false_negative
        return 2 * true_positive / denominator if denominator else 0.0

    function_names = sorted({record["metadata"]["functionality"] for record in records.values()})
    slices: dict[str, dict[str, Any]] = {}
    for name in function_names:
        ids = [decision_id for decision_id, record in records.items() if record["metadata"]["functionality"] == name]
        slice_predictions = [prediction for prediction in valid if prediction["decision_id"] in ids]
        slice_correct = [
            int(prediction["answer"] == records[prediction["decision_id"]]["gold"])
            for prediction in slice_predictions
        ]
        slices[name] = {
            "total_records": len(ids),
            "valid_predictions": len(slice_predictions),
            "accuracy_on_valid": sum(slice_correct) / len(slice_correct) if slice_correct else None,
        }

    correct = [int(answer == label) for answer, label in zip(answers, gold)]
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, gold)) / len(valid) if valid else None
    ece = _ece(probabilities, gold)
    metrics = _common_metrics(run_manifest, manifest, predictions, valid)
    metrics.update(
        {
            "accuracy_on_valid": sum(correct) / len(correct) if correct else None,
            "macro_f1_on_valid": (f1(true_positive, false_positive, false_negative) + f1(true_negative, false_negative, false_positive)) / 2 if valid else None,
            "precision_on_valid": precision,
            "recall_on_valid": recall,
            "auroc_on_valid": _auroc(probabilities, gold),
            "true_probability_brier_on_valid": brier,
            "true_probability_ece_10_bins_on_valid": ece,
            "top_label_brier_on_valid": brier,
            "top_label_ece_10_bins_on_valid": ece,
            "confusion_matrix": {
                "true": {"true": true_positive, "false": false_negative},
                "false": {"true": false_positive, "false": true_negative},
            },
            "functionality_slices": slices,
        }
    )
    return metrics


def score(run_dir: Path, package_dir: Path) -> dict[str, Any]:
    manifest = read_json(package_dir / "experiment-manifest.json")
    run_manifest = read_json(run_dir / "run-manifest.json")
    if run_manifest["experiment_package_hash"] != manifest["experiment_package_hash"]:
        raise ValueError("Run and experiment package hashes do not match")
    record_sets = manifest.get("record_sets", ["choice"])
    if not isinstance(record_sets, list) or len(record_sets) != 1:
        raise ValueError("Experiment package must contain exactly one record set")
    records = {record["decision_id"]: record for record in read_jsonl(package_dir / f"records.{record_sets[0]}.jsonl")}
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    by_id = {prediction["decision_id"]: prediction for prediction in predictions}
    if set(by_id) != set(records):
        raise ValueError("Run does not have one terminal prediction per decision record")

    task_type = next(iter(records.values())).get("task_type", "choice")
    if task_type == "noul":
        metrics = _score_noul(manifest, run_manifest, records, predictions)
        _save_evaluation(run_dir, metrics)
        return metrics
    if task_type != "choice":
        raise ValueError(f"Unsupported task type: {task_type}")

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
    metrics = _common_metrics(run_manifest, manifest, predictions, valid)
    metrics.update(
        {
        "accuracy_on_valid": sum(correct) / len(correct) if correct else None,
        "macro_f1_on_valid": sum(f1_values) / len(f1_values) if valid else None,
        "top_label_brier_on_valid": sum((confidence - outcome) ** 2 for confidence, outcome in zip(confidences, correct)) / len(valid) if valid else None,
        "top_label_ece_10_bins_on_valid": _ece(confidences, correct),
        "confusion_matrix": {label: dict(confusion[label]) for label in labels},
        }
    )
    _save_evaluation(run_dir, metrics)
    return metrics
