from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
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
    answer: str
    selected_probability: float
    model_revision: str | None
    provider_usage: dict[str, Any] | None
    cost_usd: float | None
    raw_response: dict[str, Any]


def _post_json(endpoint: str, payload: dict[str, Any], api_key: str, timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        endpoint,
        data=canonical_json(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "jev-decision-bench/0.1",
        },
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


class ChoiceAdapter:
    adapter_version = "choice-adapter-v1"

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


class TypeSafeDirectAdapter(ChoiceAdapter):
    adapter_version = "typesafe-direct-choice-v1"

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
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
        answer = answers.get("intent") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise AdapterError("TypeSafe response lacks an intent answer")
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


def _model_arguments(config: dict[str, Any], reserved: set[str]) -> dict[str, Any]:
    """Return provider-specific arguments without allowing contract overrides."""
    arguments = config.get("model_config", {})
    if not isinstance(arguments, dict):
        raise ValueError("model_config must be a JSON object")
    conflicts = sorted(set(arguments).intersection(reserved))
    if conflicts:
        raise ValueError(f"model_config cannot override benchmark request fields: {', '.join(conflicts)}")
    return dict(arguments)


class OpenAICompatibleAdapter(ChoiceAdapter):
    adapter_version = "openai-compatible-choice-v2"

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        schema = _choice_schema(record)
        payload: dict[str, Any] = {
            "model": self.config["model_id"],
            "max_tokens": self.config.get("max_tokens", 128),
            "messages": [
                {
                    "role": "system",
                    "content": _choice_instructions(),
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
        choice, confidence = parsed.get("choice"), parsed.get("confidence")
        if not isinstance(choice, str) or not isinstance(confidence, (int, float)):
            raise AdapterError(
                "OpenAI-compatible response lacks choice or confidence",
                invalid_output=True,
                body=canonical_json(response),
            )
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        return AdapterResult(
            answer=choice,
            selected_probability=float(confidence),
            model_revision=response.get("model") or self.config.get("model_revision"),
            provider_usage=usage,
            cost_usd=_configured_cost(usage, self.config),
            raw_response=response,
        )


class OpenAIResponsesAdapter(ChoiceAdapter):
    """Adapter for providers implementing the OpenAI Responses API."""

    adapter_version = "openai-responses-choice-v1"

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config["model_id"],
            "max_output_tokens": self.config.get("max_output_tokens", self.config.get("max_tokens", 128)),
            "input": [
                {"role": "system", "content": _choice_instructions()},
                {"role": "user", "content": _choice_input(record)},
            ],
            "text": {"format": {"type": "json_schema", **_choice_schema(record)}},
        }
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
        choice, confidence = parsed.get("choice"), parsed.get("confidence")
        if not isinstance(choice, str) or not isinstance(confidence, (int, float)):
            raise AdapterError(
                "OpenAI Responses response lacks choice or confidence",
                invalid_output=True,
                body=canonical_json(response),
            )
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        return AdapterResult(
            answer=choice,
            selected_probability=float(confidence),
            model_revision=response.get("model") or self.config.get("model_revision"),
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


def build_adapter(config: dict[str, Any]) -> ChoiceAdapter:
    adapter = config.get("adapter")
    if adapter == "typesafe_direct":
        return TypeSafeDirectAdapter(config)
    if adapter == "openai_compatible":
        return OpenAICompatibleAdapter(config)
    if adapter == "openai_responses":
        return OpenAIResponsesAdapter(config)
    raise ValueError(f"Unsupported adapter: {adapter}")


def sleep_before_retry(attempt: int) -> None:
    time.sleep(min(4.0, 0.5 * (2**attempt)))
