from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_decision_bench.adapters import AdapterError, AdapterResult
from jev_decision_bench.runner import run
from jev_decision_bench.scoring import score
from jev_decision_bench.util import canonical_json, read_json, read_jsonl, sha256_bytes, write_json, write_jsonl


class _FakeAdapter:
    adapter_version = "openai-compatible-chat-completions-choice-v2"

    def render(self, record):
        return {"fixture": record["decision_id"], "query": record["state"]["query"]}

    def predict(self, record):
        return AdapterResult(
            answer="one",
            selected_probability=0.8,
            model_revision="fake-revision",
            provider_usage={"input_tokens": 1, "output_tokens": 1},
            cost_usd=0.01,
            raw_response={"answer": "one"},
        )


class _InvalidOutputAdapter(_FakeAdapter):
    def __init__(self):
        self.calls = 0

    def predict(self, record):
        self.calls += 1
        if self.calls == 1:
            return super().predict(record)
        raise AdapterError("invalid JSON", invalid_output=True, body="not JSON")


class RunnerTests(unittest.TestCase):
    def test_runner_records_every_terminal_prediction_and_scores_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            write_json(
                package / "experiment-manifest.json",
                {"experiment_package_hash": "package-hash", "experiment_id": "fixture", "experiment_version": "v0"},
            )
            records = [
                {"decision_id": "a", "state": {"query": "first"}, "question": "q", "criteria": {"one": "one", "two": "two"}, "gold": "one"},
                {"decision_id": "b", "state": {"query": "second"}, "question": "q", "criteria": {"one": "one", "two": "two"}, "gold": "two"},
            ]
            write_jsonl(package / "records.choice.jsonl", records)
            config = root / "model.json"
            write_json(
                config,
                {"adapter": "openai_compatible_chat_completions", "provider": "fixture", "endpoint": "https://example.invalid", "api_key_env": "UNUSED", "model_id": "fake", "max_retries": 0},
            )
            progress: list[tuple[int, int]] = []
            with patch("jev_decision_bench.runner.build_adapter", return_value=_FakeAdapter()):
                run_dir = run(package, config, root / "artifacts", on_progress=lambda completed, total: progress.append((completed, total)))
            manifest = read_json(run_dir / "run-manifest.json")
            self.assertRegex(run_dir.name, r"^fixture-v0--fixture--fake--\d{8}T\d{6}Z--[0-9a-f]{8}$")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["counts"]["valid"], 2)
            self.assertEqual(progress, [(0, 2), (1, 2), (2, 2)])
            pinned_config = read_json(run_dir / "model-config.json")
            self.assertEqual(pinned_config, read_json(config))
            self.assertEqual(manifest["model_config"], pinned_config)
            self.assertEqual(
                manifest["model_config_sha256"],
                sha256_bytes(canonical_json(pinned_config).encode("utf-8")),
            )
            preflight_event = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
            self.assertLess(preflight_event.index('"response"'), preflight_event.index('"request"'))
            self.assertTrue(preflight_event.endswith('}}'))
            self.assertIn('"My physical card has not arrived."', preflight_event)
            predictions = read_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual(len(predictions), 2)
            self.assertTrue(all(item["status"] == "valid" for item in predictions))
            first_prediction = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
            self.assertLess(first_prediction.index('"decision_id"'), first_prediction.index('"status"'))
            self.assertLess(first_prediction.index('"status"'), first_prediction.index('"answer"'))
            metrics = score(run_dir, package)
            self.assertAlmostEqual(metrics["coverage"], 1.0)
            self.assertAlmostEqual(metrics["accuracy_on_valid"], 0.5)
            with self.assertRaisesRegex(ValueError, "equivalent run already exists"):
                run(package, config, root / "artifacts")
            with patch("jev_decision_bench.runner.build_adapter", return_value=_FakeAdapter()):
                repeated_run = run(package, config, root / "artifacts", repeat=True)
            self.assertNotEqual(run_dir, repeated_run)

    def test_runner_rejects_inline_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            write_json(
                package / "experiment-manifest.json",
                {"experiment_package_hash": "package-hash", "experiment_id": "fixture", "experiment_version": "v0"},
            )
            write_jsonl(
                package / "records.choice.jsonl",
                [{"decision_id": "a", "state": {"query": "first"}, "question": "q", "criteria": {"one": "one"}, "gold": "one"}],
            )
            config = root / "model.json"
            write_json(
                config,
                {
                    "adapter": "openai_compatible_chat_completions",
                    "provider": "fixture",
                    "endpoint": "https://example.invalid",
                    "api_key_env": "UNUSED",
                    "api_key": "must-not-be-accepted",
                    "model_id": "fake",
                },
            )
            with self.assertRaisesRegex(ValueError, "must name an environment variable"):
                run(package, config, root / "artifacts")

    def test_runner_rejects_inline_api_key_in_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            write_json(
                package / "experiment-manifest.json",
                {"experiment_package_hash": "package-hash", "experiment_id": "fixture", "experiment_version": "v0"},
            )
            write_jsonl(
                package / "records.choice.jsonl",
                [{"decision_id": "a", "state": {"query": "first"}, "question": "q", "criteria": {"one": "one"}, "gold": "one"}],
            )
            config = root / "model.json"
            write_json(
                config,
                {
                    "adapter": "openai_compatible_chat_completions",
                    "provider": "fixture",
                    "endpoint": "https://example.invalid",
                    "api_key_env": "UNUSED",
                    "model_id": "fake",
                    "model_config": {"api_key": "must-not-be-accepted"},
                },
            )
            with self.assertRaisesRegex(ValueError, "model_config.api_key"):
                run(package, config, root / "artifacts")

    def test_runner_records_malformed_model_output_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            write_json(
                package / "experiment-manifest.json",
                {"experiment_package_hash": "package-hash", "experiment_id": "fixture", "experiment_version": "v0"},
            )
            write_jsonl(
                package / "records.choice.jsonl",
                [{"decision_id": "a", "state": {"query": "first"}, "question": "q", "criteria": {"one": "one"}, "gold": "one"}],
            )
            config = root / "model.json"
            write_json(
                config,
                {"adapter": "openai_compatible_chat_completions", "provider": "fixture", "endpoint": "https://example.invalid", "api_key_env": "UNUSED", "model_id": "fake", "max_retries": 0},
            )
            with patch("jev_decision_bench.runner.build_adapter", return_value=_InvalidOutputAdapter()):
                run_dir = run(package, config, root / "artifacts")
            manifest = read_json(run_dir / "run-manifest.json")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["counts"]["invalid"], 1)
            self.assertEqual(manifest["counts"]["provider_error"], 0)
