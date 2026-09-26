from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jev_decision_bench.compare import compare
from jev_decision_bench.scoring import evaluation_output_dir
from jev_decision_bench.util import read_json, write_json, write_jsonl


def _write_run(directory: Path, run_id: str, predictions: list[dict]) -> dict:
    directory.mkdir()
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "experiment_package_hash": "package-hash",
        "adapter_version": "fixture-v1",
        "model_config": {"provider": "fixture", "model_id": run_id, "model_revision": None},
        "counts": {"total": len(predictions)},
    }
    write_json(directory / "run-manifest.json", manifest)
    write_jsonl(directory / "predictions.jsonl", predictions)
    return manifest


def _prediction(status: str, timing_ms: float, *, input_tokens: int | None = 100, output_tokens: int | None = 10) -> dict:
    usage = None if input_tokens is None else {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return {
        "decision_id": f"decision-{timing_ms}",
        "status": status,
        "timing_ms": timing_ms,
        "cost_usd": None,
        "provider_usage": usage,
    }


class CompareTests(unittest.TestCase):
    def test_compares_provider_and_performance_metrics_without_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            write_json(
                package / "experiment-manifest.json",
                {"experiment_id": "fixture", "experiment_version": "v0", "experiment_package_hash": "package-hash"},
            )
            baseline, candidate = root / "baseline", root / "candidate"
            _write_run(baseline, "jev", [_prediction("valid", 10), _prediction("valid", 20)])
            _write_run(candidate, "llm", [_prediction("valid", 20), _prediction("provider_error", 30, input_tokens=None, output_tokens=None)])
            output_dir = compare(package, baseline, [candidate], root / "artifacts")
            comparison = read_json(output_dir / "comparison.json")

        row = comparison["runs"][0]
        self.assertEqual(row["provider_metrics"], {
            "total_records": 2,
            "attempted_records": 2,
            "completion_rate": 1.0,
            "status_counts": {"valid": 1, "invalid": 0, "provider_error": 1},
            "provider_success_rate": 0.5,
            "valid_output_rate": 0.5,
        })
        self.assertEqual(row["model_performance"]["latency_ms"], {"p50": 20.0, "p95": 30.0})
        self.assertEqual(row["model_performance"]["token_usage"]["usage_coverage"], 0.5)
        self.assertEqual(row["model_performance"]["token_usage"]["reported_total_tokens"], 110.0)
        self.assertIsNone(row["model_quality"])
        self.assertEqual(row["relative_to_baseline"]["model_performance"]["p50_latency_ratio"], 2.0)
        self.assertTrue(output_dir.name.startswith("fixture-v0--package-hash--"))

    def test_includes_quality_when_all_runs_have_a_requested_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            write_json(package / "experiment-manifest.json", {"experiment_package_hash": "package-hash", "task": {"task_type": "choice"}})
            evaluation = {"evaluation_id": "fixture", "evaluation_version": "v1", "task_type": "choice"}
            evaluation_path = root / "evaluation.json"
            write_json(evaluation_path, evaluation)
            baseline, candidate = root / "baseline", root / "candidate"
            baseline_manifest = _write_run(baseline, "jev", [_prediction("valid", 10)])
            candidate_manifest = _write_run(candidate, "llm", [_prediction("valid", 20)])
            for manifest, accuracy in ((baseline_manifest, 0.5), (candidate_manifest, 0.7)):
                score_dir = evaluation_output_dir(manifest, evaluation, root / "artifacts")
                score_dir.mkdir(parents=True)
                write_json(score_dir / "scores.json", {"experiment_package_hash": "package-hash", "accuracy_on_valid": accuracy})
            output_dir = compare(package, baseline, [candidate], root / "artifacts", evaluation_path)
            comparison = read_json(output_dir / "comparison.json")

        row = comparison["runs"][0]
        self.assertEqual(row["model_quality"], {"metrics": {"accuracy_on_valid": 0.7}})
        self.assertEqual(row["relative_to_baseline"]["model_quality"], {"accuracy_on_valid_delta": 0.19999999999999996})

    def test_rejects_baseline_repeated_as_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            write_json(package / "experiment-manifest.json", {"experiment_package_hash": "package-hash"})
            baseline = root / "baseline"
            _write_run(baseline, "jev", [_prediction("valid", 10)])
            with self.assertRaisesRegex(ValueError, "Baseline run"):
                compare(package, baseline, [baseline], root / "artifacts")
