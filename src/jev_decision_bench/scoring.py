from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .prepare import ROOT
from .task_contracts import contract_for_manifest
from .util import canonical_json, order_fields, read_json, read_jsonl, sha256_bytes, write_json


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(percentile / 100 * len(values)) - 1)]


def _ece(confidences: list[float], correct: list[int], bins: int) -> float | None:
    if not confidences:
        return None
    total, ece = len(confidences), 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [position for position, value in enumerate(confidences) if low <= value < high or (index == bins - 1 and value == 1)]
        if members:
            average_confidence = sum(confidences[position] for position in members) / len(members)
            average_accuracy = sum(correct[position] for position in members) / len(members)
            ece += len(members) / total * abs(average_confidence - average_accuracy)
    return ece


def _auroc(probabilities: list[float], labels: list[int]) -> float | None:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    ordered, ranks, position = sorted(enumerate(probabilities), key=lambda item: item[1]), [0.0] * len(probabilities), 0
    while position < len(ordered):
        end = position
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[position][1]:
            end += 1
        for index in range(position, end + 1):
            ranks[ordered[index][0]] = (position + 1 + end + 1) / 2
        position = end + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _common_metrics(run_manifest: dict[str, Any], manifest: dict[str, Any], predictions: list[dict[str, Any]], valid: list[dict[str, Any]]) -> dict[str, Any]:
    known_costs = [float(prediction["cost_usd"]) for prediction in predictions if isinstance(prediction["cost_usd"], (int, float))]
    latencies = [float(prediction["timing_ms"]) for prediction in predictions]
    total = len(predictions)
    return {
        "run_id": run_manifest["run_id"], "experiment_package_hash": manifest["experiment_package_hash"],
        "total_records": total, "status_counts": dict(Counter(prediction["status"] for prediction in predictions)),
        "valid_predictions": len(valid), "partial_predictions": sum(prediction["status"] == "partial" for prediction in predictions),
        "coverage": len(valid) / total if total else None,
        "valid_output_rate": len(valid) / total if total else None,
        "partial_output_rate": sum(prediction["status"] == "partial" for prediction in predictions) / total if total else None,
        "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        "cost_usd": {"known_total": sum(known_costs) if known_costs else None, "known_cost_record_count": len(known_costs), "per_attempted": sum(known_costs) / total if known_costs else None, "per_valid": sum(known_costs) / len(valid) if known_costs and valid else None},
    }


def _score_choice(manifest: dict[str, Any], run_manifest: dict[str, Any], records: dict[str, dict[str, Any]], predictions: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
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
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    metrics = _common_metrics(run_manifest, manifest, predictions, valid)
    bins = int(evaluation.get("calibration_bins", 10))
    metrics.update({
        "accuracy_on_valid": sum(correct) / len(correct) if correct else None,
        "macro_f1_on_valid": sum(f1_values) / len(f1_values) if valid else None,
        "top_label_brier_on_valid": sum((confidence - outcome) ** 2 for confidence, outcome in zip(confidences, correct)) / len(valid) if valid else None,
        "top_label_ece_10_bins": _ece(confidences, correct, bins),
        "confusion_matrix": {label: dict(confusion[label]) for label in labels},
    })
    return metrics


def _score_noul(manifest: dict[str, Any], run_manifest: dict[str, Any], records: dict[str, dict[str, Any]], predictions: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    valid = [prediction for prediction in predictions if prediction["status"] == "valid"]
    gold = [int(records[prediction["decision_id"]]["gold"]) for prediction in valid]
    probabilities = [float(prediction["positive_probability"]) for prediction in valid]
    threshold = float(evaluation.get("positive_threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError("Noul evaluation positive_threshold must be in [0, 1]")
    answers = [int(probability >= threshold) for probability in probabilities]
    true_positive = sum(answer and label for answer, label in zip(answers, gold))
    false_positive = sum(answer and not label for answer, label in zip(answers, gold))
    false_negative = sum(not answer and label for answer, label in zip(answers, gold))
    true_negative = sum(not answer and not label for answer, label in zip(answers, gold))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None

    def f1(tp: int, fp: int, fn: int) -> float:
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0

    slices: dict[str, dict[str, Any]] = {}
    for name in sorted({record["metadata"]["functionality"] for record in records.values()}):
        ids = [decision_id for decision_id, record in records.items() if record["metadata"]["functionality"] == name]
        slice_predictions = [prediction for prediction in valid if prediction["decision_id"] in ids]
        slice_correct = [int(prediction["answer"] == records[prediction["decision_id"]]["gold"]) for prediction in slice_predictions]
        slices[name] = {"total_records": len(ids), "valid_predictions": len(slice_predictions), "accuracy_on_valid": sum(slice_correct) / len(slice_correct) if slice_correct else None}
    correct = [int(answer == label) for answer, label in zip(answers, gold)]
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, gold)) / len(valid) if valid else None
    metrics = _common_metrics(run_manifest, manifest, predictions, valid)
    bins = int(evaluation.get("calibration_bins", 10))
    metrics.update({
        "accuracy_on_valid": sum(correct) / len(correct) if correct else None,
        "macro_f1_on_valid": (f1(true_positive, false_positive, false_negative) + f1(true_negative, false_negative, false_positive)) / 2 if valid else None,
        "precision_on_valid": precision, "recall_on_valid": recall, "auroc_on_valid": _auroc(probabilities, gold),
        "true_probability_brier_on_valid": brier, "true_probability_ece_10_bins": _ece(probabilities, gold, bins),
        "top_label_brier_on_valid": brier, "top_label_ece_10_bins": _ece(probabilities, gold, bins),
        "confusion_matrix": {"true": {"true": true_positive, "false": false_negative}, "false": {"true": false_positive, "false": true_negative}},
        "functionality_slices": slices,
    })
    return metrics


def _score_score(manifest: dict[str, Any], run_manifest: dict[str, Any], records: dict[str, dict[str, Any]], predictions: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    valid = [prediction for prediction in predictions if prediction["status"] == "valid"]
    gold = [int(records[prediction["decision_id"]]["gold"]) for prediction in valid]
    answers = [int(prediction["answer"]) for prediction in valid]
    scores = [float(prediction["score"]) for prediction in valid]
    confidences = [float(prediction["selected_probability"]) for prediction in valid]
    correct = [int(answer == label) for answer, label in zip(answers, gold)]
    metrics = _common_metrics(run_manifest, manifest, predictions, valid)
    bins = int(evaluation.get("calibration_bins", 10))
    metrics.update({
        "accuracy_on_valid": sum(correct) / len(correct) if correct else None,
        "exact_tier_accuracy_on_valid": sum(correct) / len(correct) if correct else None,
        "ordinal_mae_on_valid": sum(abs(score - label) for score, label in zip(scores, gold)) / len(valid) if valid else None,
        "top_label_brier_on_valid": sum((confidence - outcome) ** 2 for confidence, outcome in zip(confidences, correct)) / len(valid) if valid else None,
        "top_label_ece_10_bins": _ece(confidences, correct, bins),
    })
    return metrics


_TASK_SCORERS: dict[str, Callable[..., dict[str, Any]]] = {"choice": _score_choice, "noul": _score_noul, "score": _score_score}


def _default_evaluation_path(manifest: dict[str, Any]) -> Path:
    experiment_id, version = manifest.get("experiment_id"), manifest.get("experiment_version")
    if not isinstance(experiment_id, str) or not isinstance(version, str):
        raise ValueError("Specify --evaluation because this package has no experiment identifier")
    return ROOT / "experiments" / experiment_id / version / "default-evaluation.json"


def resolve_evaluation_config(manifest: dict[str, Any], evaluation_config_path: Path | None = None) -> dict[str, Any]:
    path = evaluation_config_path or _default_evaluation_path(manifest)
    if not path.is_file():
        raise ValueError(f"Evaluation configuration does not exist: {path}")
    evaluation = read_json(path)
    expected_task = contract_for_manifest(manifest).task_type
    if evaluation.get("task_type") != expected_task:
        raise ValueError("Evaluation configuration task type does not match the experiment package")
    if not isinstance(evaluation.get("evaluation_id"), str) or not isinstance(evaluation.get("evaluation_version"), str):
        raise ValueError("Evaluation configuration requires evaluation_id and evaluation_version")
    return evaluation


def _path_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return token or "evaluation"


def evaluation_output_dir(run_manifest: dict[str, Any], evaluation: dict[str, Any], artifacts_root: Path) -> Path:
    config_hash = sha256_bytes(canonical_json(evaluation).encode("utf-8"))
    name = f"{_path_token(evaluation['evaluation_id'])}--{config_hash[:12]}"
    return artifacts_root / "evaluations" / run_manifest["run_id"] / name


def _ordered_scores(metrics: dict[str, Any]) -> dict[str, Any]:
    return order_fields(metrics, (
        "evaluation_id", "evaluation_version", "evaluation_config_sha256", "task_type", "run_id", "experiment_package_hash", "total_records", "status_counts", "valid_predictions", "partial_predictions", "coverage", "valid_output_rate", "partial_output_rate", "accuracy_on_valid", "exact_tier_accuracy_on_valid", "ordinal_mae_on_valid", "macro_f1_on_valid", "precision_on_valid", "recall_on_valid", "auroc_on_valid", "true_probability_brier_on_valid", "true_probability_ece_10_bins", "top_label_brier_on_valid", "top_label_ece_10_bins", "latency_ms", "cost_usd", "confusion_matrix", "functionality_slices",
    ))


def score(run_dir: Path, package_dir: Path, evaluation_config_path: Path | None = None, artifacts_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    manifest, run_manifest = read_json(package_dir / "experiment-manifest.json"), read_json(run_dir / "run-manifest.json")
    if run_manifest["experiment_package_hash"] != manifest["experiment_package_hash"]:
        raise ValueError("Run and experiment package hashes do not match")
    contract = contract_for_manifest(manifest)
    task_type = run_manifest.get("task_type", contract.task_type)
    if task_type != contract.task_type:
        raise ValueError("Run task type does not match the experiment package")
    records = {record["decision_id"]: record for record in read_jsonl(package_dir / f"records.{contract.record_set}.jsonl")}
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    if {prediction["decision_id"] for prediction in predictions} != set(records):
        raise ValueError("Run does not have one terminal prediction per decision record")
    evaluation = resolve_evaluation_config(manifest, evaluation_config_path)
    try:
        scorer = _TASK_SCORERS[task_type]
    except KeyError as error:
        raise ValueError(f"No scorer registered for task type: {task_type}") from error
    metrics = scorer(manifest, run_manifest, records, predictions, evaluation)
    config_hash = sha256_bytes(canonical_json(evaluation).encode("utf-8"))
    metrics.update({"evaluation_id": evaluation["evaluation_id"], "evaluation_version": evaluation["evaluation_version"], "evaluation_config_sha256": config_hash, "task_type": task_type})
    root = artifacts_root or (run_dir.parent.parent if run_dir.parent.name == "runs" else run_dir.parent)
    output_dir = evaluation_output_dir(run_manifest, evaluation, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation-config.json", evaluation)
    write_json(output_dir / "scores.json", _ordered_scores(metrics))
    return output_dir, metrics
