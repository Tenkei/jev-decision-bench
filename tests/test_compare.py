from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jev_decision_bench.compare import compare
from jev_decision_bench.util import read_json, write_json


def _write_run(directory: Path, run_id: str, metrics: dict[str, float | None]) -> None:
    directory.mkdir()
    write_json(
        directory / "run-manifest.json",
        {
            "run_id": run_id,
            "status": "completed",
            "experiment_package_hash": "package-hash",
            "adapter_version": "fixture-v1",
            "model_config": {"provider": "fixture", "model_id": run_id, "model_revision": None},
        },
    )
    write_json(
        directory / "evaluation.json",
        {
            "run_id": run_id,
            "experiment_package_hash": "package-hash",
            "coverage": metrics["coverage"],
            "accuracy_on_valid": metrics["accuracy"],
            "macro_f1_on_valid": metrics["macro_f1"],
            "top_label_brier_on_valid": metrics["brier"],
            "top_label_ece_10_bins": metrics["ece"],
            "latency_ms": {"p50": metrics["p50"], "p95": metrics["p95"]},
            "cost_usd": {"per_attempted": metrics["cost"]},
        },
    )


class CompareTests(unittest.TestCase):
    def test_compares_each_run_to_an_explicit_baseline_without_changing_absolute_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            write_json(package / "experiment-manifest.json", {"experiment_package_hash": "package-hash"})
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_run(
                baseline,
                "jev",
                {"coverage": 0.8, "accuracy": 0.5, "macro_f1": 0.4, "brier": 0.2, "ece": 0.1, "p50": 10, "p95": 20, "cost": 0.1},
            )
            _write_run(
                candidate,
                "llm",
                {"coverage": 0.9, "accuracy": 0.7, "macro_f1": 0.6, "brier": 0.15, "ece": 0.05, "p50": 25, "p95": 30, "cost": 0.25},
            )
            output_dir = compare(package, baseline, [candidate], root / "artifacts")
            comparison = read_json(output_dir / "comparison.json")

        self.assertEqual(comparison["baseline"]["run_id"], "jev")
        row = comparison["runs"][0]
        self.assertEqual(row["accuracy_on_valid"], 0.7)
        self.assertEqual(
            row["relative_to_baseline"],
            {
                "baseline_run_id": "jev",
                "coverage_delta": 0.09999999999999998,
                "accuracy_on_valid_delta": 0.19999999999999996,
                "macro_f1_on_valid_delta": 0.19999999999999996,
                "top_label_brier_on_valid_delta": -0.05000000000000002,
                "top_label_ece_10_bins_on_valid_delta": -0.05,
                "p50_latency_ratio": 2.5,
                "p95_latency_ratio": 1.5,
                "cost_per_attempted_ratio": 2.5,
            },
        )

    def test_rejects_baseline_repeated_as_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            write_json(package / "experiment-manifest.json", {"experiment_package_hash": "package-hash"})
            baseline = root / "baseline"
            _write_run(
                baseline,
                "jev",
                {"coverage": 1, "accuracy": 1, "macro_f1": 1, "brier": 0, "ece": 0, "p50": 1, "p95": 1, "cost": None},
            )
            with self.assertRaisesRegex(ValueError, "Baseline run"):
                compare(package, baseline, [baseline], root / "artifacts")
