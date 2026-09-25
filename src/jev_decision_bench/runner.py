from __future__ import annotations

import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .adapters import AdapterError, adapter_version, build_adapter, sleep_before_retry
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


def _validate_choice(record: dict[str, Any], answer: str, probability: float) -> str | None:
    if answer not in record["criteria"]:
        return f"choice is not in frozen rubric: {answer}"
    if not 0 <= probability <= 1:
        return f"selected probability must be in [0, 1], got {probability}"
    return None


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


def run(
    package_dir: Path,
    model_config_path: Path,
    artifacts_root: Path,
    on_progress: Callable[[int, int], None] | None = None,
    *,
    repeat: bool = False,
) -> Path:
    manifest = read_json(package_dir / "experiment-manifest.json")
    records = read_jsonl(package_dir / "records.choice.jsonl")
    config = read_json(model_config_path)
    pinned_config = _pinned_model_config(config)
    model_config_sha256 = sha256_bytes(canonical_json(pinned_config).encode("utf-8"))
    renderer_version = adapter_version(config)
    if not repeat:
        _reject_equivalent_run(
            _equivalent_runs(
                artifacts_root,
                experiment_package_hash=manifest["experiment_package_hash"],
                model_config_sha256=model_config_sha256,
                adapter_version=renderer_version,
            )
        )
    adapter = build_adapter(config)
    run_id = _run_id(manifest, config)
    run_dir = artifacts_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    run_manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at": _utc_now(),
        "status": "running",
        "experiment_package_hash": manifest["experiment_package_hash"],
        "experiment_id": manifest["experiment_id"],
        "experiment_version": manifest["experiment_version"],
        "model_config": pinned_config,
        "model_config_sha256": model_config_sha256,
        "adapter_version": renderer_version,
        "execution": {"mode": "serial", "logical_decisions_per_request": 1},
        "counts": {"total": len(records), "attempted": 0, "valid": 0, "invalid": 0, "provider_error": 0},
    }
    write_json(run_dir / "model-config.json", pinned_config)
    write_json(run_dir / "run-manifest.json", _ordered_run_manifest(run_manifest))

    # An in-domain synthetic request validates credentials and response shape
    # without exposing or scoring a package record.  It must have a clear
    # answer from the frozen rubric; an intentionally unrelated query can
    # cause a well-behaved bounded classifier to search for a nonexistent
    # "unknown" label instead of testing the response contract.
    preflight_record = {
        "decision_id": "preflight",
        "state": {"query": "My physical card has not arrived."},
        "question": records[0]["question"],
        "criteria": records[0]["criteria"],
    }
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

    if on_progress:
        on_progress(0, len(records))

    for record in records:
        run_manifest["counts"]["attempted"] += 1
        started = time.monotonic_ns()
        try:
            result = _call_with_retries(adapter, record, int(config.get("max_retries", 2)))
            latency_ms = (time.monotonic_ns() - started) / 1_000_000
            validation_error = _validate_choice(record, result.answer, result.selected_probability)
            status = "invalid" if validation_error else "valid"
            prediction = {
                "decision_id": record["decision_id"],
                "status": status,
                "answer": result.answer if not validation_error else None,
                "selected_probability": result.selected_probability if not validation_error else None,
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
