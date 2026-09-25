from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .util import canonical_json


class AdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        invalid_output: bool = False,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.invalid_output = invalid_output
        self.body = body


@dataclass(frozen=True)
class AdapterResult:
    answer: str | bool
    selected_probability: float
    model_revision: str | None
    provider_usage: dict[str, Any] | None
    cost_usd: float | None
    raw_response: dict[str, Any]
    positive_probability: float | None = None


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: float,
    *,
    api_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "jev-decision-bench/0.1",
        **(api_headers or {"Authorization": f"Bearer {api_key}"}),
    }
    request = Request(
        endpoint,
        data=canonical_json(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310: endpoint is user configuration
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise AdapterError(
            f"HTTP {error.code}: {body[:500]}", retryable=error.code == 429 or error.code >= 500, body=body
        ) from error
    except (URLError, TimeoutError) as error:
        raise AdapterError(f"Transport error: {error}", retryable=True) from error
    except json.JSONDecodeError as error:
        raise AdapterError(f"Provider returned invalid JSON: {error}") from error


class DecisionAdapter:
    adapter_version = "decision-adapter-v1"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        env_name = config["api_key_env"]
        try:
            self.api_key = os.environ[env_name]
        except KeyError as error:
            raise ValueError(f"Missing API key environment variable: {env_name}") from error

    def predict(self, record: dict[str, Any]) -> AdapterResult:
        raise NotImplementedError

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(
            self.config["endpoint"], payload, self.api_key, float(self.config.get("timeout_seconds", 60))
        )


class TypeSafeSystemOneAdapter(DecisionAdapter):
    adapter_version = "typesafe-system-one-v1"

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("task_type", "choice") == "noul":
            return {
                "model": self.config["model_id"],
                "state": record["state"],
                "questions": {
                    "decision": {
                        "type": "noul",
                        "instructions": record["question"],
                        "criteria": record["criteria"],
                    }
                },
            }
        return {
            "model": self.config["model_id"],
            "state": record["state"],
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": record["question"],
                    "criteria": record["criteria"],
                }
            },
        }

    def predict(self, record: dict[str, Any]) -> AdapterResult:
        response = self._post(self.render(record))
        answers = response.get("answers") or response.get("choices")
        answer = answers.get("decision" if record.get("task_type", "choice") == "noul" else "intent") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise AdapterError("TypeSafe response lacks the requested answer")
        if record.get("task_type", "choice") == "noul":
            probability_true = answer.get("noul")
            if not isinstance(probability_true, (int, float)):
                raise AdapterError("TypeSafe response lacks a Noul probability")
            positive_probability = float(probability_true)
            return AdapterResult(
                answer=positive_probability >= 0.5,
                selected_probability=max(positive_probability, 1 - positive_probability),
                positive_probability=positive_probability,
                model_revision=response.get("model") or self.config.get("model_revision"),
                provider_usage=response.get("usage") if isinstance(response.get("usage"), dict) else None,
                cost_usd=None,
                raw_response=response,
            )
        choice = answer.get("choice")
        probabilities = answer.get("probabilities")
        if not isinstance(choice, str) or not isinstance(probabilities, dict):
            raise AdapterError("TypeSafe response lacks choice probabilities")
        probability = probabilities.get(choice)
        if not isinstance(probability, (int, float)):
            raise AdapterError("TypeSafe response lacks probability for selected choice")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        cost = usage.get("cost") if usage else None
        return AdapterResult(
            answer=choice,
            selected_probability=float(probability),
            model_revision=response.get("model") or self.config.get("model_revision"),
            provider_usage=usage,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            raw_response=response,
        )


def _choice_schema(record: dict[str, Any]) -> dict[str, Any]:
    labels = list(record["criteria"])
    return {
        "name": "banking77_choice",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "choice": {"type": "string", "enum": labels},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["choice", "confidence"],
        },
    }


def _choice_instructions() -> str:
    return (
        "Make one bounded classification decision. Return exactly one JSON object and nothing else. "
        "It must contain only `choice` (an exact key from the supplied criteria) and `confidence` "
        "(a number from 0 to 1). Do not return reasoning, tags, prose, Markdown, an `intent` "
        "field, an unknown label, or any other field."
    )


def _choice_input(record: dict[str, Any]) -> str:
    return canonical_json({"state": record["state"], "question": record["question"], "criteria": record["criteria"]})


def _noul_schema(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "bounded_noul",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer": {"type": "boolean"},
                "probability_true": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["answer", "probability_true"],
        },
    }


def _noul_instructions() -> str:
    return (
        "Make one bounded true-or-false decision. Return exactly one JSON object and nothing else. "
        "It must contain only `answer` (a boolean) and `probability_true` (the probability from 0 to 1 "
        "that the answer is true). `answer` must be true when `probability_true` is at least 0.5, "
        "otherwise false. Do not return reasoning, tags, prose, Markdown, or any other field."
    )


def _decision_schema(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("task_type", "choice") == "choice":
        return _choice_schema(record)
    if record.get("task_type") == "noul":
        return _noul_schema(record)
    raise ValueError(f"Unsupported task type: {record.get('task_type')}")


def _decision_instructions(record: dict[str, Any]) -> str:
    if record.get("task_type", "choice") == "choice":
        return _choice_instructions()
    if record.get("task_type") == "noul":
        return _noul_instructions()
    raise ValueError(f"Unsupported task type: {record.get('task_type')}")


def _parsed_llm_decision(record: dict[str, Any], parsed: dict[str, Any]) -> tuple[str | bool, float, float | None]:
    if record.get("task_type", "choice") == "choice":
        choice, confidence = parsed.get("choice"), parsed.get("confidence")
        if not isinstance(choice, str) or not isinstance(confidence, (int, float)):
            raise AdapterError("LLM response lacks choice or confidence", invalid_output=True)
        return choice, float(confidence), None
    answer, probability_true = parsed.get("answer"), parsed.get("probability_true")
    if not isinstance(answer, bool) or not isinstance(probability_true, (int, float)):
        raise AdapterError("LLM response lacks a boolean answer or true probability", invalid_output=True)
    probability_true = float(probability_true)
    if not 0 <= probability_true <= 1 or answer != (probability_true >= 0.5):
        raise AdapterError("LLM Noul response has inconsistent answer or true probability", invalid_output=True)
    return answer, max(probability_true, 1 - probability_true), probability_true


def _model_arguments(config: dict[str, Any], reserved: set[str]) -> dict[str, Any]:
    """Return provider-specific arguments without allowing contract overrides."""
    arguments = config.get("model_config", {})
    if not isinstance(arguments, dict):
        raise ValueError("model_config must be a JSON object")
    conflicts = sorted(set(arguments).intersection(reserved))
    if conflicts:
        raise ValueError(f"model_config cannot override benchmark request fields: {', '.join(conflicts)}")
    return dict(arguments)


class OpenAICompatibleChatCompletionsAdapter(DecisionAdapter):
    adapter_version = "openai-compatible-chat-completions-v2"

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        schema = _decision_schema(record)
        payload: dict[str, Any] = {
            "model": self.config["model_id"],
            "max_tokens": self.config.get("max_tokens", 128),
            "messages": [
                {
                    "role": "system",
                    "content": _decision_instructions(record),
                },
                {
                    "role": "user",
                    "content": _choice_input(record),
                },
            ],
        }
        if self.config.get("response_format", "json_schema") == "json_schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        else:
            payload["response_format"] = {"type": "json_object"}
        payload.update(
            _model_arguments(self.config, {"model", "max_tokens", "messages", "response_format"})
        )
        return payload

    def predict(self, record: dict[str, Any]) -> AdapterResult:
        response = self._post(self.render(record))
        try:
            content = response["choices"][0]["message"]["content"]
            parsed = _parse_openai_content(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise AdapterError(
                f"OpenAI-compatible response lacks valid JSON content: {error}",
                invalid_output=True,
                body=canonical_json(response),
            ) from error
        if not isinstance(parsed, dict):
            raise AdapterError(
                "OpenAI-compatible response JSON is not an object",
                invalid_output=True,
                body=canonical_json(response),
            )
        try:
            answer, selected_probability, positive_probability = _parsed_llm_decision(record, parsed)
        except AdapterError as error:
            raise AdapterError(str(error), invalid_output=True, body=canonical_json(response)) from error
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        return AdapterResult(
            answer=answer,
            selected_probability=selected_probability,
            positive_probability=positive_probability,
            model_revision=response.get("model") or self.config.get("model_revision"),
            provider_usage=usage,
            cost_usd=_configured_cost(usage, self.config),
            raw_response=response,
        )


class OpenAICompatibleResponsesAdapter(DecisionAdapter):
    """Adapter for providers implementing the OpenAI Responses API."""

    adapter_version = "openai-compatible-responses-v1"

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config["model_id"],
            "max_output_tokens": self.config.get("max_output_tokens", self.config.get("max_tokens", 128)),
            "input": [
                {"role": "system", "content": _decision_instructions(record)},
                {"role": "user", "content": _choice_input(record)},
            ],
        }
        response_format = self.config.get("response_format", "json_schema")
        if response_format == "json_schema":
            payload["text"] = {"format": {"type": "json_schema", **_decision_schema(record)}}
        elif response_format == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}
        else:
            raise ValueError(f"Unsupported OpenAI Responses response_format: {response_format}")
        payload.update(
            _model_arguments(self.config, {"model", "max_output_tokens", "input", "text"})
        )
        return payload

    def predict(self, record: dict[str, Any]) -> AdapterResult:
        response = self._post(self.render(record))
        try:
            parsed = _parse_openai_content(_responses_output_text(response))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise AdapterError(
                f"OpenAI Responses response lacks valid JSON content: {error}",
                invalid_output=True,
                body=canonical_json(response),
            ) from error
        if not isinstance(parsed, dict):
            raise AdapterError(
                "OpenAI Responses response JSON is not an object",
                invalid_output=True,
                body=canonical_json(response),
            )
        try:
            answer, selected_probability, positive_probability = _parsed_llm_decision(record, parsed)
        except AdapterError as error:
            raise AdapterError(str(error), invalid_output=True, body=canonical_json(response)) from error
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        return AdapterResult(
            answer=answer,
            selected_probability=selected_probability,
            positive_probability=positive_probability,
            model_revision=response.get("model") or self.config.get("model_revision"),
            provider_usage=usage,
            cost_usd=_configured_cost(usage, self.config),
            raw_response=response,
        )


class BedrockConverseAdapter(DecisionAdapter):
    """Adapter for Bedrock Runtime's provider-neutral Converse API."""

    adapter_version = "bedrock-converse-v1"

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "system": [{"text": _decision_instructions(record)}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": _choice_input(record)}],
                }
            ],
            "inferenceConfig": {"maxTokens": self.config.get("max_tokens", 128)},
        }
        response_format = self.config.get("response_format", "json_schema")
        if response_format == "json_schema":
            schema = _decision_schema(record)
            # Bedrock Converse rejects numeric minimum and maximum constraints.
            # The benchmark still validates confidence is in [0, 1] after the
            # response, so this only relaxes a provider-side rendering detail.
            for property_schema in schema["schema"]["properties"].values():
                property_schema.pop("minimum", None)
                property_schema.pop("maximum", None)
            payload["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": schema["name"],
                            "description": "Return one bounded benchmark classification decision.",
                            "schema": canonical_json(schema["schema"]),
                        }
                    },
                }
            }
        elif response_format != "json_object":
            raise ValueError(f"Unsupported Bedrock Converse response_format: {response_format}")
        model_options = self.config.get("model_config", {})
        if not isinstance(model_options, dict):
            raise ValueError("model_config must be a JSON object")
        if model_options:
            payload["additionalModelRequestFields"] = model_options
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = self.config["endpoint"].rstrip("/")
        model_id = quote(self.config["model_id"], safe=".:_-" )
        return _post_json(
            f"{endpoint}/model/{model_id}/converse",
            payload,
            self.api_key,
            float(self.config.get("timeout_seconds", 60)),
        )

    def predict(self, record: dict[str, Any]) -> AdapterResult:
        response = self._post(self.render(record))
        try:
            text = next(
                part["text"]
                for part in response["output"]["message"]["content"]
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
            parsed = json.loads(text)
        except (KeyError, StopIteration, TypeError, json.JSONDecodeError) as error:
            raise AdapterError(
                f"Bedrock Converse response lacks valid JSON content: {error}",
                invalid_output=True,
                body=canonical_json(response),
            ) from error
        if not isinstance(parsed, dict):
            raise AdapterError(
                "Bedrock Converse response JSON is not an object",
                invalid_output=True,
                body=canonical_json(response),
            )
        try:
            answer, selected_probability, positive_probability = _parsed_llm_decision(record, parsed)
        except AdapterError as error:
            raise AdapterError(str(error), invalid_output=True, body=canonical_json(response)) from error
        raw_usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        usage = (
            {
                "input_tokens": raw_usage.get("inputTokens"),
                "output_tokens": raw_usage.get("outputTokens"),
                "total_tokens": raw_usage.get("totalTokens"),
            }
            if raw_usage
            else None
        )
        return AdapterResult(
            answer=answer,
            selected_probability=selected_probability,
            positive_probability=positive_probability,
            model_revision=self.config.get("model_revision"),
            provider_usage=usage,
            cost_usd=_configured_cost(usage, self.config),
            raw_response=response,
        )


_REASONING_PREFIX = re.compile(r"^\s*<reasoning>.*?</reasoning>\s*", re.DOTALL)


def _parse_openai_content(content: Any) -> Any:
    """Parse a Chat Completions answer without changing its decision value.

    Some reasoning models on OpenAI-compatible gateways place their separate
    reasoning channel in a complete ``<reasoning>...</reasoning>`` prefix of
    ``message.content``.  Removing that transport wrapper exposes the model's
    requested JSON answer.  We deliberately do not extract or repair JSON
    embedded in arbitrary prose: malformed answers remain invalid evidence.
    """
    if not isinstance(content, str):
        return content
    return json.loads(_REASONING_PREFIX.sub("", content, count=1))


def _responses_output_text(response: dict[str, Any]) -> str:
    """Read the first output-text part from a completed Responses result."""
    for output in response["output"]:
        if not isinstance(output, dict) or output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text
    raise KeyError("output_text")


def _configured_cost(usage: dict[str, Any] | None, config: dict[str, Any]) -> float | None:
    if not usage or "input_cost_per_million" not in config or "output_cost_per_million" not in config:
        return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if not isinstance(prompt, (int, float)) or not isinstance(completion, (int, float)):
        return None
    return (prompt * float(config["input_cost_per_million"]) + completion * float(config["output_cost_per_million"])) / 1_000_000


def _adapter_class(config: dict[str, Any]) -> type[DecisionAdapter]:
    adapter = config.get("adapter")
    if adapter == "typesafe_system_one":
        return TypeSafeSystemOneAdapter
    if adapter == "openai_compatible_chat_completions":
        return OpenAICompatibleChatCompletionsAdapter
    if adapter == "openai_compatible_responses":
        return OpenAICompatibleResponsesAdapter
    if adapter == "bedrock_converse":
        return BedrockConverseAdapter
    raise ValueError(f"Unsupported adapter: {adapter}")


def adapter_version(config: dict[str, Any]) -> str:
    """Return the request-rendering version without requiring credentials."""
    return _adapter_class(config).adapter_version


def build_adapter(config: dict[str, Any]) -> DecisionAdapter:
    return _adapter_class(config)(config)


def sleep_before_retry(attempt: int) -> None:
    time.sleep(min(4.0, 0.5 * (2**attempt)))
