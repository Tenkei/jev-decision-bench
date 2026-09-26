from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jev_decision_bench.scoring import score
from jev_decision_bench.util import write_json, write_jsonl


def _write_evaluation(path: Path, task_type: str) -> None:
    write_json(
        path,
        {
            "evaluation_id": f"fixture-{task_type}",
            "evaluation_version": "v0",
            "task_type": task_type,
            "calibration_bins": 10,
        },
    )


class ScoringTests(unittest.TestCase):
    def test_scores_valid_predictions_and_keeps_provider_errors_out_of_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            run = root / "run"
            package.mkdir()
            run.mkdir()
            write_json(package / "experiment-manifest.json", {"experiment_package_hash": "package-hash"})
            records = [
                {"decision_id": "a", "criteria": {"one": "1", "two": "2"}, "gold": "one"},
                {"decision_id": "b", "criteria": {"one": "1", "two": "2"}, "gold": "two"},
                {"decision_id": "c", "criteria": {"one": "1", "two": "2"}, "gold": "one"},
            ]
            write_jsonl(package / "records.choice.jsonl", records)
            write_json(run / "run-manifest.json", {"experiment_package_hash": "package-hash", "run_id": "run-1"})
            write_jsonl(
                run / "predictions.jsonl",
                [
                    {"decision_id": "a", "status": "valid", "answer": "one", "selected_probability": 0.9, "timing_ms": 10, "cost_usd": 0.1},
                    {"decision_id": "b", "status": "valid", "answer": "one", "selected_probability": 0.7, "timing_ms": 20, "cost_usd": 0.2},
                    {"decision_id": "c", "status": "provider_error", "answer": None, "selected_probability": None, "timing_ms": 30, "cost_usd": None},
                ],
            )
            evaluation = root / "choice-evaluation.json"
            _write_evaluation(evaluation, "choice")
            output_dir, metrics = score(run, package, evaluation)
            self.assertEqual(metrics["valid_predictions"], 2)
            self.assertAlmostEqual(metrics["coverage"], 2 / 3)
            self.assertAlmostEqual(metrics["accuracy_on_valid"], 0.5)
            self.assertEqual(metrics["status_counts"], {"valid": 2, "provider_error": 1})
            self.assertTrue((output_dir / "scores.json").exists())
            self.assertTrue((output_dir / "evaluation-config.json").exists())
            self.assertFalse((run / "evaluation.json").exists())
            self.assertFalse((run / "metrics.json").exists())
            self.assertFalse((run / "report.md").exists())

    def test_scores_noul_predictions_and_functionality_slices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            run = root / "run"
            package.mkdir()
            run.mkdir()
            write_json(package / "experiment-manifest.json", {"experiment_package_hash": "package-hash", "record_sets": ["noul"]})
            write_jsonl(
                package / "records.noul.jsonl",
                [
                    {"decision_id": "a", "task_type": "noul", "gold": True, "metadata": {"functionality": "threat"}},
                    {"decision_id": "b", "task_type": "noul", "gold": False, "metadata": {"functionality": "counter_speech"}},
                ],
            )
            write_json(run / "run-manifest.json", {"experiment_package_hash": "package-hash", "run_id": "run-1"})
            write_jsonl(
                run / "predictions.jsonl",
                [
                    {"decision_id": "a", "status": "valid", "answer": True, "selected_probability": 0.9, "positive_probability": 0.9, "timing_ms": 10, "cost_usd": None},
                    {"decision_id": "b", "status": "valid", "answer": False, "selected_probability": 0.8, "positive_probability": 0.2, "timing_ms": 20, "cost_usd": None},
                ],
            )
            evaluation = root / "noul-evaluation.json"
            _write_evaluation(evaluation, "noul")
            output_dir, metrics = score(run, package, evaluation)
            self.assertAlmostEqual(metrics["accuracy_on_valid"], 1.0)
            self.assertAlmostEqual(metrics["precision_on_valid"], 1.0)
            self.assertAlmostEqual(metrics["recall_on_valid"], 1.0)
            self.assertAlmostEqual(metrics["auroc_on_valid"], 1.0)
            self.assertAlmostEqual(metrics["true_probability_brier_on_valid"], 0.025)
            self.assertEqual(metrics["functionality_slices"]["threat"]["total_records"], 1)
            self.assertTrue((output_dir / "scores.json").exists())

            strict_evaluation = root / "noul-strict-evaluation.json"
            _write_evaluation(strict_evaluation, "noul")
            strict = json.loads(strict_evaluation.read_text(encoding="utf-8"))
            strict.update({"evaluation_id": "fixture-noul-strict", "positive_threshold": 0.95})
            write_json(strict_evaluation, strict)
            strict_output, strict_metrics = score(run, package, strict_evaluation)
            self.assertNotEqual(output_dir, strict_output)
            self.assertAlmostEqual(strict_metrics["accuracy_on_valid"], 0.5)

    def test_scores_ordered_level_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, run = root / "package", root / "run"
            package.mkdir()
            run.mkdir()
            write_json(package / "experiment-manifest.json", {"experiment_package_hash": "package-hash", "record_sets": ["score"]})
            write_jsonl(
                package / "records.score.jsonl",
                [
                    {"decision_id": "a", "task_type": "score", "gold": 2, "criteria": ["0", "1", "2", "3"]},
                    {"decision_id": "b", "task_type": "score", "gold": 0, "criteria": ["0", "1", "2", "3"]},
                    {"decision_id": "c", "task_type": "score", "gold": 3, "criteria": ["0", "1", "2", "3"]},
                ],
            )
            write_json(run / "run-manifest.json", {"experiment_package_hash": "package-hash", "run_id": "run-1"})
            write_jsonl(
                run / "predictions.jsonl",
                [
                    {"decision_id": "a", "status": "valid", "answer": 2, "score": 2.4, "selected_probability": 0.6, "timing_ms": 10, "cost_usd": None},
                    {"decision_id": "b", "status": "valid", "answer": 1, "score": 1.0, "selected_probability": 1.0, "timing_ms": 20, "cost_usd": None},
                    {"decision_id": "c", "status": "partial", "answer": 3, "score": 3.0, "selected_probability": 1.0, "timing_ms": 30, "cost_usd": None},
                ],
            )
            evaluation = root / "score-evaluation.json"
            _write_evaluation(evaluation, "score")
            _, metrics = score(run, package, evaluation)
        self.assertAlmostEqual(metrics["exact_tier_accuracy_on_valid"], 0.5)
        self.assertAlmostEqual(metrics["ordinal_mae_on_valid"], 0.7)
        self.assertEqual(metrics["partial_predictions"], 1)
        self.assertAlmostEqual(metrics["partial_output_rate"], 1 / 3)
