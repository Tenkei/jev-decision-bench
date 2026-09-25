from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Banking77SpecificationTests(unittest.TestCase):
    def test_rubric_has_77_nonempty_definitions(self) -> None:
        rubric = json.loads((ROOT / "experiments" / "banking77-choice" / "v0" / "rubric.json").read_text(encoding="utf-8"))
        criteria = rubric["criteria"]
        self.assertEqual(len(criteria), 77)
        self.assertEqual(len(set(criteria)), 77)
        self.assertTrue(all(value.strip() for value in criteria.values()))

    def test_spec_uses_immutable_revision_and_hashes(self) -> None:
        spec = json.loads((ROOT / "experiments" / "banking77-choice" / "v0" / "spec.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["experiment_id"], "banking77-choice")
        self.assertEqual(len(spec["dataset"]["revision"]), 40)
        for file_spec in spec["dataset"]["files"].values():
            self.assertEqual(len(file_spec["sha256"]), 64)
            self.assertIn(spec["dataset"]["revision"], file_spec["url"])
