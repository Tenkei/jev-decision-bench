from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from jev_decision_bench.adapters import (
    AdapterError,
    BedrockConverseAdapter,
    OpenAICompatibleChatCompletionsAdapter,
    OpenAICompatibleResponsesAdapter,
    TypeSafeSystemOneAdapter,
)


class AdapterTests(unittest.TestCase):
    def test_typesafe_system_one_adapter_includes_model_id_in_native_request(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1"}}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}):
            adapter = TypeSafeSystemOneAdapter(
                {
                    "adapter": "typesafe_system_one",
                    "provider": "test",
                    "endpoint": "https://example.invalid",
                    "api_key_env": "TEST_KEY",
                    "model_id": "jev",
                }
            )
            payload = adapter.render(record)
        self.assertEqual(payload["model"], "jev")

    def test_typesafe_system_one_adapter_renders_and_parses_noul(self) -> None:
        record = {
            "task_type": "noul",
            "state": {"text": "I hate women."},
            "question": "Is this hateful content?",
            "criteria": {"true": "Hateful content.", "false": "Non-hateful content."},
        }
        response = {"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": {"decision": {"type": "noul", "noul": 0.9}}}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ):
            adapter = TypeSafeSystemOneAdapter(
                {"adapter": "typesafe_system_one", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "jev"}
            )
            payload = adapter.render(record)
            result = adapter.predict(record)
        self.assertEqual(payload["questions"]["decision"], {"type": "noul", "instructions": "Is this hateful content?", "criteria": record["criteria"]})
        self.assertTrue(result.answer)
        self.assertEqual(result.positive_probability, 0.9)
        self.assertEqual(result.selected_probability, 0.9)

    def test_openai_chat_completions_adapter_parses_schema_constrained_output(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1", "two": "2"}}
        response = {"choices": [{"message": {"content": '{"choice":"one","confidence":0.8}'}}]}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch("jev_decision_bench.adapters._post_json", return_value=response):
            adapter = OpenAICompatibleChatCompletionsAdapter(
                {"adapter": "openai_compatible_chat_completions", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "model"}
            )
            result = adapter.predict(record)
        self.assertEqual(result.answer, "one")
        self.assertEqual(result.selected_probability, 0.8)

    def test_openai_chat_completions_adapter_discards_a_complete_reasoning_prefix(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1", "two": "2"}}
        response = {
            "choices": [
                {"message": {"content": '<reasoning>private analysis</reasoning>{"choice":"one","confidence":0.8}'}}
            ]
        }
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ):
            adapter = OpenAICompatibleChatCompletionsAdapter(
                {"adapter": "openai_compatible_chat_completions", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "model"}
            )
            result = adapter.predict(record)
        self.assertEqual(result.answer, "one")
        self.assertEqual(result.selected_probability, 0.8)

    def test_openai_chat_completions_adapter_parses_noul_output(self) -> None:
        record = {
            "task_type": "noul",
            "state": {"text": "I do not hate anyone."},
            "question": "Is this hateful content?",
            "criteria": {"true": "Hateful content.", "false": "Non-hateful content."},
        }
        response = {"choices": [{"message": {"content": '{"answer":false,"probability_true":0.1}'}}]}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ):
            adapter = OpenAICompatibleChatCompletionsAdapter(
                {"adapter": "openai_compatible_chat_completions", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "model"}
            )
            result = adapter.predict(record)
            payload = adapter.render(record)
        self.assertFalse(result.answer)
        self.assertEqual(result.positive_probability, 0.1)
        self.assertEqual(result.selected_probability, 0.9)
        self.assertEqual(payload["response_format"]["json_schema"]["schema"]["properties"]["answer"], {"type": "boolean"})

    def test_openai_compatible_responses_adapter_parses_structured_output(self) -> None:
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
            adapter = OpenAICompatibleResponsesAdapter(
                {
                    "adapter": "openai_compatible_responses",
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

    def test_openai_compatible_responses_adapter_can_request_a_json_object(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1"}}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}):
            adapter = OpenAICompatibleResponsesAdapter(
                {
                    "adapter": "openai_compatible_responses",
                    "provider": "test",
                    "endpoint": "https://example.invalid",
                    "api_key_env": "TEST_KEY",
                    "model_id": "model",
                    "response_format": "json_object",
                }
            )
            payload = adapter.render(record)
        self.assertEqual(payload["text"], {"format": {"type": "json_object"}})

    def test_bedrock_converse_adapter_uses_a_bearer_key_and_native_schema(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1", "two": "2"}}
        response = {
            "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
            "output": {"message": {"content": [{"text": '{"choice":"two","confidence":0.7}'}]}},
        }
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ) as post:
            adapter = BedrockConverseAdapter(
                {
                    "adapter": "bedrock_converse",
                    "provider": "amazon-bedrock",
                    "endpoint": "https://bedrock-runtime.ap-northeast-1.amazonaws.com",
                    "api_key_env": "TEST_KEY",
                    "model_id": "jp.anthropic.claude-haiku-4-5-20251001-v1:0",
                }
            )
            result = adapter.predict(record)
            payload = adapter.render(record)
        self.assertEqual(result.answer, "two")
        self.assertEqual(result.provider_usage["input_tokens"], 11)
        self.assertEqual(payload["outputConfig"]["textFormat"]["type"], "json_schema")
        rendered_schema = json.loads(payload["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
        self.assertEqual(rendered_schema["properties"]["confidence"], {"type": "number"})
        self.assertEqual(post.call_args.args[0], "https://bedrock-runtime.ap-northeast-1.amazonaws.com/model/jp.anthropic.claude-haiku-4-5-20251001-v1:0/converse")

    def test_model_config_cannot_override_the_benchmark_contract(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1"}}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}):
            adapter = OpenAICompatibleChatCompletionsAdapter(
                {
                    "adapter": "openai_compatible_chat_completions",
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

    def test_openai_chat_completions_adapter_preserves_invalid_response_for_diagnosis(self) -> None:
        record = {"state": {"query": "x"}, "question": "q", "criteria": {"one": "1"}}
        response = {"choices": [{"message": {"content": "not JSON"}}]}
        with patch.dict(os.environ, {"TEST_KEY": "secret"}), patch(
            "jev_decision_bench.adapters._post_json", return_value=response
        ):
            adapter = OpenAICompatibleChatCompletionsAdapter(
                {"adapter": "openai_compatible_chat_completions", "provider": "test", "endpoint": "https://example.invalid", "api_key_env": "TEST_KEY", "model_id": "model"}
            )
            with self.assertRaises(AdapterError) as raised:
                adapter.predict(record)
        self.assertEqual(raised.exception.body, '{"choices":[{"message":{"content":"not JSON"}}]}')
        self.assertTrue(raised.exception.invalid_output)
