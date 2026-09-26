from __future__ import annotations

import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .adapters import AdapterError, adapter_version, build_adapter, sleep_before_retry
from .task_contracts import NormalizedDecision, contract_for_manifest, contract_for_record
from .util import append_jsonl, canonical_json, order_fields, read_json, read_jsonl, sha256_bytes, write_json


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _path_token(value: object) -> str:
    """Make a stable, human-readable path component from pinned metadata."""
    token = re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower()
    return token or "unknown"


def _run_id(manifest: dict[str, Any], config: dict[str, Any]) -> str:
    experiment = f"{_path_token(manifest['experiment_id'])}-{_path_token(manifest['experiment_version'])}"
    provider = _path_token(config["provider"])
    model = _path_token(config["model_id"])
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment}--{provider}--{model}--{timestamp}--{uuid.uuid4().hex[:8]}"


def _records_path(package_dir: Path, manifest: dict[str, Any]) -> Path:
    return package_dir / f"records.{contract_for_manifest(manifest).record_set}.jsonl"


def _preflight_record(package_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = package_dir / "preflight.json"
    if not path.is_file():
        raise ValueError("Experiment package is missing preflight.json; prepare the experiment again")
    record = read_json(path)
    if contract_for_record(record).task_type != contract_for_manifest(manifest).task_type:
        raise ValueError("Preflight fixture does not match the package task contract")
    actual_hash = sha256_bytes(canonical_json(record).encode("utf-8"))
    expected_hash = manifest.get("preflight_sha256")
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("Preflight fixture hash does not match the experiment manifest")
    return record


def _safe_event_request(record: dict[str, Any], config: dict[str, Any], adapter) -> dict[str, Any]:
    return {
        "decision_id": record["decision_id"],
        "adapter": config["adapter"],
        "endpoint": config["endpoint"],
        "model_id": config.get("model_id"),
        "payload": adapter.render(record),
    }


def _pinned_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reject inline credentials and return the exact safe configuration to persist."""
    forbidden = _inline_api_key_paths(config)
    if forbidden:
        raise ValueError(f"Model config must name an environment variable, not include credentials: {', '.join(forbidden)}")
    return order_fields(
        config,
        (
            "adapter",
            "provider",
            "model_id",
            "model_revision",
            "endpoint",
            "api_key_env",
            "response_format",
            "max_tokens",
            "timeout_seconds",
            "max_retries",
            "model_config",
            "cost_basis",
        ),
    )


def _ordered_run_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return order_fields(
        manifest,
        (
            "run_id",
            "status",
            "started_at",
            "finished_at",
            "preflight_error",
            "experiment_id",
            "experiment_version",
            "experiment_package_hash",
            "task_type",
            "adapter_version",
            "model_config",
            "model_config_sha256",
            "execution",
            "counts",
        ),
    )


def _inline_api_key_paths(value: Any, path: str = "") -> list[str]:
    if not isinstance(value, dict):
        return []
    found: list[str] = []
    for key, child in value.items():
        child_path = f"{path}.{key}" if path else key
        if "api_key" in key.lower() and key != "api_key_env":
            found.append(child_path)
        found.extend(_inline_api_key_paths(child, child_path))
    return found


def _equivalent_runs(
    artifacts_root: Path,
    *,
    experiment_package_hash: str,
    model_config_sha256: str,
    adapter_version: str,
) -> list[tuple[Path, str]]:
    """Find prior runs with an identical task, route configuration, and renderer."""
    runs_dir = artifacts_root / "runs"
    if not runs_dir.is_dir():
        return []
    matches: list[tuple[Path, str]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        manifest_path = run_dir / "run-manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            existing = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            existing.get("experiment_package_hash") == experiment_package_hash
            and existing.get("model_config_sha256") == model_config_sha256
            and existing.get("adapter_version") == adapter_version
        ):
            matches.append((run_dir, str(existing.get("status", "unknown"))))
    return matches


def _reject_equivalent_run(matches: list[tuple[Path, str]]) -> None:
    if not matches:
        return
    existing = "\n".join(f"- {path} ({status})" for path, status in matches)
    raise ValueError(
        "An equivalent run already exists:\n"
        f"{existing}\n"
        "This matches the experiment package, model configuration, and adapter version. "
        "Use score or compare to inspect it, or pass --repeat to create an intentional new trial."
    )


_TERMINAL_PREDICTION_STATUSES = frozenset({"valid", "invalid", "provider_error"})


def _read_resume_predictions(predictions_path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not predictions_path.exists():
        return []
    predictions = read_jsonl(predictions_path)
    record_ids = {record["decision_id"] for record in records}
    seen: set[str] = set()
    for prediction in predictions:
        decision_id = prediction.get("decision_id")
        status = prediction.get("status")
        if not isinstance(decision_id, str) or decision_id not in record_ids:
            raise ValueError("Resume run contains a prediction outside the selected experiment package")
        if decision_id in seen:
            raise ValueError(f"Resume run contains duplicate prediction: {decision_id}")
        if status not in _TERMINAL_PREDICTION_STATUSES:
            raise ValueError(f"Resume run contains a non-terminal prediction status: {status}")
        seen.add(decision_id)
    return predictions


def _prediction_counts(predictions: list[dict[str, Any]], total: int) -> dict[str, int]:
    return {
        "total": total,
        "attempted": len(predictions),
        "valid": sum(prediction["status"] == "valid" for prediction in predictions),
        "invalid": sum(prediction["status"] == "invalid" for prediction in predictions),
        "provider_error": sum(prediction["status"] == "provider_error" for prediction in predictions),
    }


def _has_successful_preflight(events_path: Path) -> bool:
    if not events_path.exists():
        return False
    for event in read_jsonl(events_path):
        if event.get("type") != "preflight":
            continue
        response = event.get("response")
        return not (isinstance(response, dict) and "error" in response)
    return False


def _load_resume_run(
    run_dir: Path,
    *,
    experiment_package_hash: str,
    model_config_sha256: str,
    adapter_version: str,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    manifest_path = run_dir / "run-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Resume run is missing its manifest: {run_dir}")
    run_manifest = read_json(manifest_path)
    mismatches = [
        label
        for label, expected, actual in (
            ("experiment package", experiment_package_hash, run_manifest.get("experiment_package_hash")),
            ("model configuration", model_config_sha256, run_manifest.get("model_config_sha256")),
            ("adapter version", adapter_version, run_manifest.get("adapter_version")),
        )
        if expected != actual
    ]
    if mismatches:
        raise ValueError(f"Resume run does not match the current {'; '.join(mismatches)}")
    if run_manifest.get("status") != "running":
        raise ValueError(f"Only a running run can be resumed, got: {run_manifest.get('status')}")
    predictions = _read_resume_predictions(run_dir / "predictions.jsonl", records)
    return run_manifest, predictions, _has_successful_preflight(run_dir / "events.jsonl")


def run(
    package_dir: Path,
    model_config_path: Path,
    artifacts_root: Path,
    on_progress: Callable[[int, int], None] | None = None,
    *,
    repeat: bool = False,
    resume_run_dir: Path | None = None,
) -> Path:
    manifest = read_json(package_dir / "experiment-manifest.json")
    records = read_jsonl(_records_path(package_dir, manifest))
    if not records:
        raise ValueError("Experiment package has no decision records")
    contract = contract_for_manifest(manifest)
    if any(contract_for_record(record).task_type != contract.task_type for record in records):
        raise ValueError("Experiment records do not match the package task contract")
    config = read_json(model_config_path)
    pinned_config = _pinned_model_config(config)
    model_config_sha256 = sha256_bytes(canonical_json(pinned_config).encode("utf-8"))
    renderer_version = adapter_version(config)
    if resume_run_dir is not None:
        run_dir = resume_run_dir
        run_manifest, existing_predictions, preflight_succeeded = _load_resume_run(
            run_dir,
            experiment_package_hash=manifest["experiment_package_hash"],
            model_config_sha256=model_config_sha256,
            adapter_version=renderer_version,
            records=records,
        )
        run_manifest["counts"] = _prediction_counts(existing_predictions, len(records))
        write_json(run_dir / "run-manifest.json", _ordered_run_manifest(run_manifest))
    else:
        existing_predictions = []
        preflight_succeeded = False
        if not repeat:
            _reject_equivalent_run(
                _equivalent_runs(
                    artifacts_root,
                    experiment_package_hash=manifest["experiment_package_hash"],
                    model_config_sha256=model_config_sha256,
                    adapter_version=renderer_version,
                )
            )
        run_id = _run_id(manifest, config)
        run_dir = artifacts_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run_manifest = {
            "run_id": run_id,
            "started_at": _utc_now(),
            "status": "running",
            "experiment_package_hash": manifest["experiment_package_hash"],
            "experiment_id": manifest["experiment_id"],
            "experiment_version": manifest["experiment_version"],
            "task_type": contract.task_type,
            "model_config": pinned_config,
            "model_config_sha256": model_config_sha256,
            "adapter_version": renderer_version,
            "execution": {"mode": "serial", "logical_decisions_per_request": 1},
            "counts": _prediction_counts([], len(records)),
        }
        write_json(run_dir / "model-config.json", pinned_config)
        write_json(run_dir / "run-manifest.json", _ordered_run_manifest(run_manifest))

    adapter = build_adapter(config)
    events_path = run_dir / "events.jsonl"
    predictions_path = run_dir / "predictions.jsonl"

    # The experiment pins an in-domain synthetic request. It validates route
    # credentials and response shape without exposing or scoring a test item.
    preflight_record = _preflight_record(package_dir, manifest)
    if not preflight_succeeded:
        try:
            started = time.monotonic_ns()
            preflight = _call_with_retries(adapter, preflight_record, int(config.get("max_retries", 2)))
            append_jsonl(
                events_path,
                {
                    "type": "preflight",
                    "at": _utc_now(),
                    "latency_ms": (time.monotonic_ns() - started) / 1_000_000,
                    "response": preflight.raw_response,
                    "request": _safe_event_request(preflight_record, config, adapter),
                },
                sort_keys=False,
            )
        except AdapterError as error:
            append_jsonl(
                events_path,
                {
                    "type": "preflight",
                    "at": _utc_now(),
                    "response": {"error": str(error), "body": error.body},
                    "request": _safe_event_request(preflight_record, config, adapter),
                },
                sort_keys=False,
            )
            run_manifest["status"] = "preflight_failed"
            run_manifest["finished_at"] = _utc_now()
            run_manifest["preflight_error"] = str(error)
            write_json(run_dir / "run-manifest.json", _ordered_run_manifest(run_manifest))
            return run_dir

    completed_ids = {prediction["decision_id"] for prediction in existing_predictions}
    if on_progress:
        on_progress(len(completed_ids), len(records))

    for record in records:
        if record["decision_id"] in completed_ids:
            continue
        run_manifest["counts"]["attempted"] += 1
        started = time.monotonic_ns()
        try:
            result = _call_with_retries(adapter, record, int(config.get("max_retries", 2)))
            latency_ms = (time.monotonic_ns() - started) / 1_000_000
            validation_error = contract_for_record(record).validate_prediction(
                record,
                NormalizedDecision(
                    result.answer,
                    result.selected_probability,
                    result.positive_probability,
                    result.score,
                    result.level_probabilities,
                ),
            )
            status = "invalid" if validation_error else "valid"
            prediction = {
                "decision_id": record["decision_id"],
                "status": status,
                "answer": result.answer if not validation_error else None,
                "selected_probability": result.selected_probability if not validation_error else None,
                "positive_probability": result.positive_probability if not validation_error else None,
                "score": result.score if not validation_error else None,
                "level_probabilities": result.level_probabilities if not validation_error else None,
                "probability_provenance": "native" if config["adapter"] == "typesafe_system_one" else "verbalized",
                "model": {
                    "provider": config["provider"],
                    "id": config["model_id"],
                    "revision": result.model_revision,
                },
                "timing_ms": latency_ms,
                "cost_usd": result.cost_usd,
                "provider_usage": result.provider_usage,
                "validation_error": validation_error,
            }
            run_manifest["counts"][status] += 1
            response = result.raw_response
        except AdapterError as error:
            latency_ms = (time.monotonic_ns() - started) / 1_000_000
            status = "invalid" if error.invalid_output else "provider_error"
            prediction = {
                "decision_id": record["decision_id"],
                "status": status,
                "answer": None,
                "selected_probability": None,
                "positive_probability": None,
                "score": None,
                "level_probabilities": None,
                "probability_provenance": None,
                "model": {"provider": config["provider"], "id": config["model_id"], "revision": config.get("model_revision")},
                "timing_ms": latency_ms,
                "cost_usd": None,
                "provider_usage": None,
                "validation_error": str(error),
            }
            response = {"error": str(error), "body": error.body}
            run_manifest["counts"][status] += 1
        append_jsonl(predictions_path, prediction)
        append_jsonl(
            events_path,
            {
                "type": "decision",
                "at": _utc_now(),
                "prediction_status": prediction["status"],
                "latency_ms": latency_ms,
                "response": response,
                "request": _safe_event_request(record, config, adapter),
            },
            sort_keys=False,
        )
        if on_progress:
            on_progress(run_manifest["counts"]["attempted"], len(records))

    run_manifest["status"] = "completed"
    run_manifest["finished_at"] = _utc_now()
    write_json(run_dir / "run-manifest.json", _ordered_run_manifest(run_manifest))
    return run_dir


def _call_with_retries(adapter, record: dict[str, Any], max_retries: int):
    for attempt in range(max_retries + 1):
        try:
            return adapter.predict(record)
        except AdapterError as error:
            if not error.retryable or attempt == max_retries:
                raise
            sleep_before_retry(attempt)
    raise AssertionError("unreachable")
