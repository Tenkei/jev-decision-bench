from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jev_decision_bench.util import canonical_json, order_fields, write_json


class JsonPresentationTests(unittest.TestCase):
    def test_preferred_field_order_is_human_facing_only(self) -> None:
        value = {"result": 1, "identity": "example", "details": {"z": 2, "a": 1}}
        ordered = order_fields(value, ("identity", "details", "result"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            write_json(path, ordered)
            rendered = path.read_text(encoding="utf-8")
        self.assertLess(rendered.index('"identity"'), rendered.index('"details"'))
        self.assertLess(rendered.index('"details"'), rendered.index('"result"'))
        self.assertEqual(canonical_json(value), canonical_json(ordered))
