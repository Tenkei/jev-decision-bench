from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from jev_decision_bench.adapters import (
    AdapterError,
    OpenAICompatibleAdapter,
    OpenAIResponsesAdapter,
    TypeSafeDirectAdapter,
)


class AdapterTests(unittest.TestCase):
    def test_typesafe_adapter_includes_model_id_in_native_request(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1"}}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}):
            adapter = TypeSafeDirectAdapter(
                {
                    "adapter": "typesafe_direct",
                    "provider": "test",
                    "endpoint": "https://example.invalid",
                    "api_key_env": "TEST_KEY",
                    "model_id": "jev",
                }
            )
            payload = adapter.render(record)
        self.assertEqual(payload["model"], "jev")

    def test_openai_adapter_parses_schema_constrained_output(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1", "two": "2"}}
        response = {"choices": [{"message": {"content": '{"choice":"one","confidence":0.8}'}}]}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch("jev_decision_bench.adapters._post_json", return_value=response):
            adapter = OpenAICompatibleAdapter(
                {"adapter": "openai_compatible", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "model"}
            )
            result = adapter.predict(record)
        self.assertEqual(result.answer, "one")
        self.assertEqual(result.selected_probability, 0.8)

    def test_openai_adapter_discards_a_complete_reasoning_prefix(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1", "two": "2"}}
        response = {
            "choices": [
                {"message": {"content": '<reasoning>private analysis</reasoning>{"choice":"one","confidence":0.8}'}}
            ]
        }
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ):
            adapter = OpenAICompatibleAdapter(
                {"adapter": "openai_compatible", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "model"}
            )
            result = adapter.predict(record)
        self.assertEqual(result.answer, "one")
        self.assertEqual(result.selected_probability, 0.8)

    def test_openai_responses_adapter_parses_structured_output(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1", "two": "2"}}
        response = {
            "model": "response-model",
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"choice":"two","confidence":0.7}'}],
                }
            ],
        }
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ):
            adapter = OpenAIResponsesAdapter(
                {
                    "adapter": "openai_responses",
                    "provider": "test",
                    "endpoint": "https://example.invalid",
                    "api_key_env": "TEST_KEY",
                    "model_id": "model",
                    "max_tokens": 91,
                    "model_config": {"reasoning": {"effort": "low"}, "temperature": 0.2},
                }
            )
            result = adapter.predict(record)
            payload = adapter.render(record)
        self.assertEqual(result.answer, "two")
        self.assertEqual(result.selected_probability, 0.7)
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertEqual(payload["text"]["format"]["schema"]["properties"]["choice"]["enum"], ["one", "two"])
        self.assertEqual(payload["max_output_tokens"], 91)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["temperature"], 0.2)

    def test_model_config_cannot_override_the_benchmark_contract(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1"}}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}):
            adapter = OpenAICompatibleAdapter(
                {
                    "adapter": "openai_compatible",
                    "provider": "test",
                    "endpoint": "https://example.invalid",
                    "api_key_env": "TEST_KEY",
                    "model_id": "model",
                    "model_config": {"messages": []},
                }
            )
            with self.assertRaisesRegex(ValueError, "cannot override"):
                adapter.render(record)

    def test_adapter_error_marks_retryability(self) -> None:
        error = AdapterError("rate limited", retryable=True)
        self.assertTrue(error.retryable)

    def test_openai_adapter_preserves_invalid_response_for_diagnosis(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1"}}
        response = {"choices": [{"message": {"content": "not JSON"}}]}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ):
            adapter = OpenAICompatibleAdapter(
                {"adapter": "openai_compatible", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "model"}
            )
            with self.assertRaises(AdapterError) as raised:
                adapter.predict(record)
        self.assertEqual(raised.exception.body, '{"choices":[{"message":{"content":"not JSON"}}]}')
        self.assertTrue(raised.exception.invalid_output)
