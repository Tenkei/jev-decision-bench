from __future__ import annotations

import gzip
import unittest

from jev_decision_bench.prepare import _compile_trec_passage_relevance


class PrepareTests(unittest.TestCase):
    def test_trec_compiler_uses_authoritative_queries_and_preserves_line_separators(self) -> None:
        queries = "10\tTest query\r\n"
        candidates, qrels = [], []
        for grade in range(4):
            for index in range(400):
                passage_id = f"p{grade}-{index}"
                candidates.append(f"10\t{passage_id}\ttest QUERY\tPassage {grade}-{index}\u2028continued")
                qrels.append(f"10 Q0 {passage_id} {grade}")
        spec = {
            "dataset": {"sample_per_grade": 400, "expected_row_count": 1600},
            "task": {"question": "How well does this passage answer the search query?"},
        }
        rubric = {"criteria": ["zero", "one", "two", "three"]}
        source_bytes = {
            "queries.tsv.gz": gzip.compress(queries.encode("utf-8")),
            "top1000.tsv.gz": gzip.compress(("\n".join(candidates) + "\n").encode("utf-8")),
            "qrels.txt": ("\n".join(qrels) + "\n").encode("utf-8"),
        }

        records = _compile_trec_passage_relevance(spec, rubric, source_bytes)

        self.assertEqual(len(records), 1600)
        self.assertEqual({record["state"]["query"] for record in records}, {"Test query"})
        self.assertTrue(all("\u2028continued" in record["state"]["passage"] for record in records))
        self.assertEqual({record["gold"] for record in records}, {0, 1, 2, 3})
