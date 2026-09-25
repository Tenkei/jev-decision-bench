from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_decision_bench.cli import main
from jev_decision_bench.util import write_json


class CliTests(unittest.TestCase):
    def test_expected_command_error_is_printed_without_a_traceback(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = main(["run", "--package", "missing-package", "--model-config", "missing-config"])
        self.assertEqual(exit_code, 1)
        self.assertIn("Error:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_preflight_failure_is_printed_and_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            write_json(
                run_dir / "run-manifest.json",
                {"status": "preflight_failed", "preflight_error": "Missing API key environment variable: EXAMPLE_API_KEY"},
            )
            stderr = io.StringIO()
            with patch("jev_decision_bench.cli.run", return_value=run_dir) as mocked_run, contextlib.redirect_stderr(stderr):
                exit_code = main(["run", "--package", "package", "--model-config", "config", "--repeat"])
        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr.getvalue(), "Error: Missing API key environment variable: EXAMPLE_API_KEY\n")
        self.assertTrue(mocked_run.call_args.kwargs["repeat"])
