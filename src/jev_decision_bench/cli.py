from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compare import compare
from .prepare import prepare
from .runner import run
from .scoring import score
from .util import read_json


def _artifacts_path(value: str) -> Path:
    return Path(value).resolve()


def _run_progress(completed: int, total: int) -> None:
    """Report completed logical decisions without hiding failures in the count."""
    line = f"Progress: {completed:,}/{total:,} ({completed / total * 100:.1f}%)"
    if sys.stdout.isatty():
        print(f"\r{line}", end="\n" if completed == total else "", flush=True)
    elif completed == 0 or completed == total or completed % 25 == 0:
        print(line, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-decision-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Create a pinned experiment package")
    prepare_parser.add_argument("--experiment", required=True, choices=["banking77-choice-v0"])
    prepare_parser.add_argument("--artifacts-root", default="artifacts", type=_artifacts_path)
    run_parser = subparsers.add_parser("run", help="Execute one model against a prepared package")
    run_parser.add_argument("--package", required=True, type=Path)
    run_parser.add_argument("--model-config", required=True, type=Path)
    run_parser.add_argument("--artifacts-root", default="artifacts", type=_artifacts_path)
    run_parser.add_argument(
        "--repeat",
        action="store_true",
        help="Intentionally create a new trial even when an equivalent run already exists",
    )
    score_parser = subparsers.add_parser("score", help="Score saved run evidence offline")
    score_parser.add_argument("--run", required=True, type=Path)
    score_parser.add_argument("--package", required=True, type=Path)
    compare_parser = subparsers.add_parser("compare", help="Compare completed compatible runs")
    compare_parser.add_argument("--package", required=True, type=Path)
    compare_parser.add_argument("--baseline", required=True, type=Path)
    compare_parser.add_argument("--runs", required=True, type=Path, nargs="+")
    compare_parser.add_argument("--artifacts-root", default="artifacts", type=_artifacts_path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            print(prepare(args.experiment, args.artifacts_root))
        elif args.command == "run":
            run_dir = run(
                args.package,
                args.model_config,
                args.artifacts_root,
                on_progress=_run_progress,
                repeat=args.repeat,
            )
            print(run_dir)
            run_manifest = read_json(run_dir / "run-manifest.json")
            if run_manifest.get("status") == "preflight_failed":
                print(f"Error: {run_manifest['preflight_error']}", file=sys.stderr)
                return 1
        elif args.command == "score":
            metrics = score(args.run, args.package)
            print(args.run / "evaluation.json")
            print(f"accuracy_on_valid={metrics['accuracy_on_valid']}")
        elif args.command == "compare":
            print(compare(args.package, args.baseline, args.runs, args.artifacts_root))
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
