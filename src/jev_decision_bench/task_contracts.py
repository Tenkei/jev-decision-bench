from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .util import canonical_json


@dataclass(frozen=True)
class NormalizedDecision:
    """Provider-independent decision result before runner validation."""

    answer: str | bool
    selected_probability: float
    positive_probability: float | None = None


class TaskOutputError(ValueError):
    """A response is structurally incompatible with a task contract."""


class TaskContract:
    """The stable meaning of one JEV decision shape.

    Experiment specifications supply the data, rubric, and preflight fixture.
    Contracts supply the reusable mechanics for that shape only.
    """

    task_type: str
    record_set: str
    validation_contract_version: str

    def llm_schema(self, record: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def llm_instructions(self) -> str:
        raise NotImplementedError

    def llm_input(self, record: dict[str, Any]) -> str:
        return canonical_json(
            {"state": record["state"], "question": record["question"], "criteria": record["criteria"]}
        )

    def parse_llm_output(self, parsed: dict[str, Any]) -> NormalizedDecision:
        raise NotImplementedError

    def native_question(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def parse_native_output(self, response: dict[str, Any], record: dict[str, Any]) -> NormalizedDecision:
        raise NotImplementedError

    def validate_prediction(self, record: dict[str, Any], result: NormalizedDecision) -> str | None:
        if not 0 <= result.selected_probability <= 1:
            return f"selected probability must be in [0, 1], got {result.selected_probability}"
        return None

    def preflight_record(self, task_spec: dict[str, Any], criteria: dict[str, str]) -> dict[str, Any]:
        fixture = task_spec.get("preflight")
        if not isinstance(fixture, dict) or not isinstance(fixture.get("state"), dict):
            raise ValueError(f"{self.task_type} experiment task must define preflight.state")
        return {
            "decision_id": "preflight",
            "task_type": self.task_type,
            "state": fixture["state"],
            "question": task_spec["question"],
            "criteria": criteria,
        }


class ChoiceContract(TaskContract):
    task_type = "choice"
    record_set = "choice"
    validation_contract_version = "choice-top-label-confidence-v1"

    def llm_schema(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": "bounded_choice",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "choice": {"type": "string", "enum": list(record["criteria"])},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["choice", "confidence"],
            },
        }

    def llm_instructions(self) -> str:
        return (
            "Make one bounded classification decision. Return exactly one JSON object and nothing else. "
            "It must contain only `choice` (an exact key from the supplied criteria) and `confidence` "
            "(a number from 0 to 1). Do not return reasoning, tags, prose, Markdown, an `intent` "
            "field, an unknown label, or any other field."
        )

    def parse_llm_output(self, parsed: dict[str, Any]) -> NormalizedDecision:
        choice, confidence = parsed.get("choice"), parsed.get("confidence")
        if not isinstance(choice, str) or not isinstance(confidence, (int, float)):
            raise TaskOutputError("LLM response lacks choice or confidence")
        return NormalizedDecision(choice, float(confidence))

    def native_question(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return "intent", {
            "type": "choice",
            "instructions": record["question"],
            "criteria": record["criteria"],
        }

    def parse_native_output(self, response: dict[str, Any], record: dict[str, Any]) -> NormalizedDecision:
        answers = response.get("answers") or response.get("choices")
        answer = answers.get("intent") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise TaskOutputError("TypeSafe response lacks the requested answer")
        choice, probabilities = answer.get("choice"), answer.get("probabilities")
        if not isinstance(choice, str) or not isinstance(probabilities, dict):
            raise TaskOutputError("TypeSafe response lacks choice probabilities")
        probability = probabilities.get(choice)
        if not isinstance(probability, (int, float)):
            raise TaskOutputError("TypeSafe response lacks probability for selected choice")
        return NormalizedDecision(choice, float(probability))

    def validate_prediction(self, record: dict[str, Any], result: NormalizedDecision) -> str | None:
        error = super().validate_prediction(record, result)
        if error:
            return error
        if result.answer not in record["criteria"]:
            return f"choice is not in frozen rubric: {result.answer}"
        return None


class NoulContract(TaskContract):
    task_type = "noul"
    record_set = "noul"
    validation_contract_version = "noul-boolean-probability-v1"

    def llm_schema(self, record: dict[str, Any]) -> dict[str, Any]:
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

    def llm_instructions(self) -> str:
        return (
            "Make one bounded true-or-false decision. Return exactly one JSON object and nothing else. "
            "It must contain only `answer` (a boolean) and `probability_true` (the probability from 0 to 1 "
            "that the answer is true). `answer` must be true when `probability_true` is at least 0.5, "
            "otherwise false. Do not return reasoning, tags, prose, Markdown, or any other field."
        )

    def parse_llm_output(self, parsed: dict[str, Any]) -> NormalizedDecision:
        answer, probability_true = parsed.get("answer"), parsed.get("probability_true")
        if not isinstance(answer, bool) or not isinstance(probability_true, (int, float)):
            raise TaskOutputError("LLM response lacks a boolean answer or true probability")
        probability_true = float(probability_true)
        if not 0 <= probability_true <= 1 or answer != (probability_true >= 0.5):
            raise TaskOutputError("LLM Noul response has inconsistent answer or true probability")
        return NormalizedDecision(answer, max(probability_true, 1 - probability_true), probability_true)

    def native_question(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return "decision", {
            "type": "noul",
            "instructions": record["question"],
            "criteria": record["criteria"],
        }

    def parse_native_output(self, response: dict[str, Any], record: dict[str, Any]) -> NormalizedDecision:
        answers = response.get("answers") or response.get("choices")
        answer = answers.get("decision") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise TaskOutputError("TypeSafe response lacks the requested answer")
        probability_true = answer.get("noul")
        if not isinstance(probability_true, (int, float)):
            raise TaskOutputError("TypeSafe response lacks a Noul probability")
        probability_true = float(probability_true)
        return NormalizedDecision(probability_true >= 0.5, max(probability_true, 1 - probability_true), probability_true)

    def validate_prediction(self, record: dict[str, Any], result: NormalizedDecision) -> str | None:
        error = super().validate_prediction(record, result)
        if error:
            return error
        if not isinstance(result.answer, bool) or not isinstance(result.positive_probability, (int, float)):
            return "Noul output requires a boolean answer and true probability"
        if not 0 <= result.positive_probability <= 1:
            return f"true probability must be in [0, 1], got {result.positive_probability}"
        if result.answer != (result.positive_probability >= 0.5):
            return "Noul answer must match the true-probability threshold"
        return None


_CONTRACTS: dict[str, TaskContract] = {
    "choice": ChoiceContract(),
    "noul": NoulContract(),
}


def get_task_contract(task_type: str) -> TaskContract:
    try:
        return _CONTRACTS[task_type]
    except KeyError as error:
        raise ValueError(f"Unsupported task type: {task_type}") from error


def contract_for_record(record: dict[str, Any]) -> TaskContract:
    return get_task_contract(str(record.get("task_type", "choice")))


def contract_for_manifest(manifest: dict[str, Any]) -> TaskContract:
    task = manifest.get("task")
    if isinstance(task, dict) and isinstance(task.get("task_type"), str):
        return get_task_contract(task["task_type"])
    record_sets = manifest.get("record_sets", ["choice"])
    if isinstance(record_sets, list) and len(record_sets) == 1:
        return get_task_contract(str(record_sets[0]))
    raise ValueError("Experiment package must contain exactly one supported task")
