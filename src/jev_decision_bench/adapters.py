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

from .task_contracts import TaskOutputError, contract_for_record
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
    answer: str | bool | int
    selected_probability: float
    model_revision: str | None
    provider_usage: dict[str, Any] | None
    cost_usd: float | None
    raw_response: dict[str, Any]
    positive_probability: float | None = None
    score: float | None = None
    level_probabilities: dict[str, float] | None = None
    partial_error: str | None = None


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
        question_name, question = contract_for_record(record).native_question(record)
        return {
            "model": self.config["model_id"],
            "state": record["state"],
            "questions": {question_name: question},
        }

    def predict(self, record: dict[str, Any]) -> AdapterResult:
        response = self._post(self.render(record))
        try:
            decision = contract_for_record(record).parse_native_output(response, record)
        except TaskOutputError as error:
            raise AdapterError(str(error), invalid_output=True, body=canonical_json(response)) from error
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        cost = usage.get("cost") if usage else None
        return AdapterResult(
            answer=decision.answer,
            selected_probability=decision.selected_probability,
            positive_probability=decision.positive_probability,
            score=decision.score,
            level_probabilities=decision.level_probabilities,
            partial_error=decision.partial_error,
            model_revision=response.get("model") or self.config.get("model_revision"),
            provider_usage=usage,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            raw_response=response,
        )


def _normalized_llm_result(record: dict[str, Any], parsed: dict[str, Any]):
    try:
        decision = contract_for_record(record).parse_llm_output(parsed, record)
    except TaskOutputError as error:
        raise AdapterError(str(error), invalid_output=True) from error
    return decision


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
        contract = contract_for_record(record)
        schema = contract.llm_schema(record)
        payload: dict[str, Any] = {
            "model": self.config["model_id"],
            "max_tokens": self.config.get("max_tokens", 128),
            "messages": [
                {
                    "role": "system",
                    "content": contract.llm_instructions(),
                },
                {
                    "role": "user",
                    "content": contract.llm_input(record),
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
            decision = _normalized_llm_result(record, parsed)
        except AdapterError as error:
            raise AdapterError(str(error), invalid_output=True, body=canonical_json(response)) from error
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        return AdapterResult(
            answer=decision.answer,
            selected_probability=decision.selected_probability,
            positive_probability=decision.positive_probability,
            score=decision.score,
            level_probabilities=decision.level_probabilities,
            partial_error=decision.partial_error,
            model_revision=response.get("model") or self.config.get("model_revision"),
            provider_usage=usage,
            cost_usd=_configured_cost(usage, self.config),
            raw_response=response,
        )


class OpenAICompatibleResponsesAdapter(DecisionAdapter):
    """Adapter for providers implementing the OpenAI Responses API."""

    adapter_version = "openai-compatible-responses-v1"

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        contract = contract_for_record(record)
        payload: dict[str, Any] = {
            "model": self.config["model_id"],
            "max_output_tokens": self.config.get("max_output_tokens", self.config.get("max_tokens", 128)),
            "input": [
                {"role": "system", "content": contract.llm_instructions()},
                {"role": "user", "content": contract.llm_input(record)},
            ],
        }
        response_format = self.config.get("response_format", "json_schema")
        if response_format == "json_schema":
            payload["text"] = {"format": {"type": "json_schema", **contract.llm_schema(record)}}
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
            decision = _normalized_llm_result(record, parsed)
        except AdapterError as error:
            raise AdapterError(str(error), invalid_output=True, body=canonical_json(response)) from error
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        return AdapterResult(
            answer=decision.answer,
            selected_probability=decision.selected_probability,
            positive_probability=decision.positive_probability,
            score=decision.score,
            level_probabilities=decision.level_probabilities,
            partial_error=decision.partial_error,
            model_revision=response.get("model") or self.config.get("model_revision"),
            provider_usage=usage,
            cost_usd=_configured_cost(usage, self.config),
            raw_response=response,
        )


class BedrockConverseAdapter(DecisionAdapter):
    """Adapter for Bedrock Runtime's provider-neutral Converse API."""

    adapter_version = "bedrock-converse-v1"

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        contract = contract_for_record(record)
        payload: dict[str, Any] = {
            "system": [{"text": contract.llm_instructions()}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": contract.llm_input(record)}],
                }
            ],
            "inferenceConfig": {"maxTokens": self.config.get("max_tokens", 128)},
        }
        response_format = self.config.get("response_format", "json_schema")
        if response_format == "json_schema":
            schema = contract.llm_schema(record)
            # Bedrock Converse rejects numeric minimum and maximum constraints.
            # The benchmark still validates confidence is in [0, 1] after the
            # response, so this only relaxes a provider-side rendering detail.
            def strip_numeric_bounds(value: Any) -> None:
                if isinstance(value, dict):
                    value.pop("minimum", None)
                    value.pop("maximum", None)
                    for nested in value.values():
                        strip_numeric_bounds(nested)
                elif isinstance(value, list):
                    for nested in value:
                        strip_numeric_bounds(nested)

            strip_numeric_bounds(schema["schema"])
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
            decision = _normalized_llm_result(record, parsed)
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
            answer=decision.answer,
            selected_probability=decision.selected_probability,
            positive_probability=decision.positive_probability,
            score=decision.score,
            level_probabilities=decision.level_probabilities,
            partial_error=decision.partial_error,
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
