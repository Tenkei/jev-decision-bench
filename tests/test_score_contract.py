from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from jev_decision_bench.adapters import BedrockConverseAdapter, OpenAICompatibleChatCompletionsAdapter, TypeSafeSystemOneAdapter
from jev_decision_bench.task_contracts import TaskOutputError, get_task_contract


RECORD = {
    "task_type": "score",
    "state": {"query": "capital of france", "passage": "Paris is the capital of France."},
    "question": "How well does this passage answer the search query?",
    "criteria": ["Not relevant.", "Related but does not answer.", "Highly relevant.", "Perfect answer."],
}


class ScoreContractTests(unittest.TestCase):
    def test_typesafe_score_preserves_the_native_level_distribution(self) -> None:
        response = {"answers": {"score": {"type": "score", "score": 2.4, "probabilities": {"0": 0.0, "1": 0.0, "2": 0.6, "3": 0.4}}}}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch("jev_decision_bench.adapters._post_json", return_value=response):
            adapter = TypeSafeSystemOneAdapter({"adapter": "typesafe_system_one", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "jev"})
            payload, result = adapter.render(RECORD), adapter.predict(RECORD)
        self.assertEqual(payload["questions"]["score"], {"type": "score", "instructions": RECORD["question"], "criteria": RECORD["criteria"]})
        self.assertEqual(result.answer, 2)
        self.assertAlmostEqual(result.score, 2.4)
        self.assertEqual(result.level_probabilities, {"0": 0.0, "1": 0.0, "2": 0.6, "3": 0.4})

    def test_llm_score_requires_a_complete_normalized_distribution(self) -> None:
        response = {"choices": [{"message": {"content": '{"score":2.4,"probabilities":{"0":0.0,"1":0.0,"2":0.6,"3":0.4}}'}}]}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch("jev_decision_bench.adapters._post_json", return_value=response):
            adapter = OpenAICompatibleChatCompletionsAdapter({"adapter": "openai_compatible_chat_completions", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "model"})
            result = adapter.predict(RECORD)
        self.assertEqual(result.answer, 2)
        self.assertAlmostEqual(result.score, 2.4)

    def test_llm_score_rejects_a_missing_reported_score(self) -> None:
        with self.assertRaisesRegex(TaskOutputError, "numeric score"):
            get_task_contract("score").parse_llm_output(
                {"probabilities": {"0": 0.0, "1": 0.0, "2": 0.6, "3": 0.4}}, RECORD
            )

    def test_typesafe_score_keeps_its_reported_score_when_probabilities_are_rounded(self) -> None:
        response = {"answers": {"score": {"type": "score", "score": 0.32, "probabilities": {"0": 0.69, "1": 0.31, "2": 0.0, "3": 0.0}}}}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ):
            adapter = TypeSafeSystemOneAdapter(
                {"adapter": "typesafe_system_one", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "jev"}
            )
            result = adapter.predict(RECORD)
        self.assertAlmostEqual(result.score, 0.32)

    def test_score_distribution_mismatch_is_a_partial_answer(self) -> None:
        response = {"answers": {"score": {"type": "score", "score": 3.0, "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0}}}}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ):
            adapter = TypeSafeSystemOneAdapter(
                {"adapter": "typesafe_system_one", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "jev"}
            )
            result = adapter.predict(RECORD)
        self.assertEqual(result.partial_error, "Score does not match its level probabilities")
        self.assertEqual(result.score, 3.0)

    def test_bedrock_score_schema_removes_nested_numeric_bounds(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": "secret"}):
            adapter = BedrockConverseAdapter(
                {
                    "adapter": "bedrock_converse",
                    "provider": "test",
                    "endpoint": "https://example.invalid",
                    "api_key_env": "TEST_KEY",
                    "model_id": "model",
                }
            )
            payload = adapter.render(RECORD)
        schema = json.loads(payload["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
        probabilities = schema["properties"]["probabilities"]["properties"]
        self.assertEqual(probabilities["2"], {"type": "number"})
        self.assertEqual(schema["properties"]["score"], {"type": "number"})
