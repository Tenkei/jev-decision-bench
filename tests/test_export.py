from __future__ import annotations

import gzip
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from jev_decision_bench.compare import compare
from jev_decision_bench.export import export_results
from jev_decision_bench.scoring import score
from jev_decision_bench.util import canonical_json, read_json, sha256_bytes, write_json, write_jsonl


class ExportTests(unittest.TestCase):
    def test_exports_sanitized_evidence_that_can_be_rescored_and_compared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, artifacts, comparison_dir = root / "package", root / "artifacts", root / "comparison"
            package.mkdir()
            comparison_dir.mkdir()
            package_manifest = {
                "experiment_id": "fixture", "experiment_version": "v0", "experiment_package_hash": "package-hash",
                "task": {"task_type": "choice"},
            }
            evaluation = {"evaluation_id": "fixture", "evaluation_version": "v0", "task_type": "choice"}
            evaluation_hash = sha256_bytes(canonical_json(evaluation).encode("utf-8"))
            write_json(package / "experiment-manifest.json", package_manifest)
            write_jsonl(
                package / "records.choice.jsonl",
                [
                    {"decision_id": "a", "task_type": "choice", "criteria": {"one": "first", "two": "second"}, "gold": "one"},
                    {"decision_id": "b", "task_type": "choice", "criteria": {"one": "first", "two": "second"}, "gold": "two"},
                ],
            )
            (comparison_dir / "README.md").write_text(
                "# Fixture report\n\n[Experiment](../../../experiments/fixture/v0/README.md)\n", encoding="utf-8"
            )
            write_json(
                comparison_dir / "comparison.json",
                {
                    "comparison_id": "fixture-v0--package-hash--comparison",
                    "experiment_package_hash": "package-hash",
                    "evaluation_config_sha256": evaluation_hash,
                    "baseline": {"run_id": "jev-run"},
                    "runs": [{"run_id": "model-run"}],
                },
            )
            for run_id, answers in (("jev-run", ["one", "two"]), ("model-run", ["one", "one"])):
                run_dir = artifacts / "runs" / run_id
                run_dir.mkdir(parents=True)
                write_json(
                    run_dir / "run-manifest.json",
                    {
                        "run_id": run_id,
                        "status": "completed",
                        "experiment_package_hash": "package-hash",
                        "adapter_version": "fixture-v1",
                        "model_config": {
                            "provider": "fixture", "model_id": run_id, "model_revision": None,
                            "api_key_env": "SAFE_KEY_NAME", "api_key": "do-not-export",
                            "headers": {"authorization": "Bearer do-not-export"}, "max_tokens": 1024,
                            "model_config": {"reasoning": {"effort": "low"}},
                        },
                        "counts": {"total": 2},
                    },
                )
                write_jsonl(
                    run_dir / "predictions.jsonl",
                    [
                        {"decision_id": decision_id, "status": "valid", "answer": answer, "selected_probability": 0.9, "timing_ms": 10, "cost_usd": None, "provider_usage": {"input_tokens": 10, "output_tokens": 2}}
                        for decision_id, answer in zip(("a", "b"), answers)
                    ],
                )
            evaluation_dir = artifacts / "evaluations" / "jev-run" / "fixture-policy"
            evaluation_dir.mkdir(parents=True)
            write_json(evaluation_dir / "evaluation-config.json", evaluation)

            summary, archive = export_results(package, comparison_dir, artifacts, root / "results", root / "exports")

            self.assertEqual(summary.name, "fixture-v0--package-hash--comparison")
            self.assertEqual(archive.name, "fixture-v0--package-hash--comparison.tar.gz")
            self.assertTrue((summary / "run-manifests" / "model-run.json").is_file())
            with gzip.open(archive, "rb") as compressed, tarfile.open(fileobj=io.BytesIO(compressed.read())) as contents:
                names = set(contents.getnames())
                exported_root = "jev-decision-bench-export-fixture-v0--package-hash--comparison"
                self.assertIn(f"{exported_root}/package/records.choice.jsonl", names)
                self.assertIn(f"{exported_root}/runs/model-run/predictions.jsonl", names)
                self.assertNotIn(f"{exported_root}/runs/model-run/events.jsonl", names)
                manifest_file = contents.extractfile(f"{exported_root}/runs/model-run/run-manifest.json")
                self.assertIsNotNone(manifest_file)
                manifest = json.load(io.TextIOWrapper(manifest_file, encoding="utf-8"))
                contents.extractall(root / "unpacked", filter="data")
            config = manifest["model_config"]
            self.assertEqual(config["api_key_env"], "SAFE_KEY_NAME")
            self.assertEqual(config["api_key"], "[REDACTED]")
            self.assertEqual(config["headers"], "[REDACTED]")
            self.assertEqual(config["max_tokens"], 1024)
            self.assertEqual(config["model_config"]["reasoning"]["effort"], "low")

            unpacked = root / "unpacked" / exported_root
            for run_id in ("jev-run", "model-run"):
                score(unpacked / "runs" / run_id, unpacked / "package", unpacked / "evaluation-config.json", unpacked)
            comparison_output = compare(
                unpacked / "package", unpacked / "runs" / "jev-run", [unpacked / "runs" / "model-run"], unpacked,
                unpacked / "evaluation-config.json",
            )
            rebuilt = read_json(comparison_output / "comparison.json")
            self.assertAlmostEqual(rebuilt["baseline"]["model_quality"]["metrics"]["accuracy_on_valid"], 1.0)
            self.assertAlmostEqual(rebuilt["runs"][0]["model_quality"]["metrics"]["accuracy_on_valid"], 0.5)
