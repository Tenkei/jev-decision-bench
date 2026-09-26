#!/usr/bin/env python3
"""Render comparison charts from the committed result summaries.

Run from the repository root with:
    .venv/bin/python scripts/render_results_charts.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"

EXPERIMENTS = {
    "Choice": "banking77-choice-v0--175e210593f7--20260926T070757Z-4975f004",
    "Noul": "hatecheck-noul-v0--1409f2a86b78--20260926T070758Z-2e6677d0",
    "Score": "trec-score-v0--91f38528cbd7--20260926T140924Z-110f1aa8",
}
MODEL_NAMES = {
    "jev-latest": "JEV",
    "deepseek.v3.2": "DeepSeek V3.2",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5",
    "global.anthropic.claude-sonnet-5": "Claude Sonnet 5",
    "global.openai.gpt-5.6-luna": "GPT-5.6 Luna",
    "global.openai.gpt-5.6-sol": "GPT-5.6 Sol",
}
MODEL_ORDER = list(MODEL_NAMES.values())
COLORS = {
    "JEV": "#2563eb",
    "DeepSeek V3.2": "#9333ea",
    "Claude Haiku 4.5": "#ea580c",
    "Claude Sonnet 5": "#d97706",
    "GPT-5.6 Luna": "#0891b2",
    "GPT-5.6 Sol": "#059669",
}


def _read_comparison(directory: str) -> dict:
    return json.loads((RESULTS / directory / "comparison.json").read_text(encoding="utf-8"))


def _rows(comparison: dict) -> dict[str, dict]:
    rows = [comparison["baseline"], *comparison["runs"]]
    return {MODEL_NAMES[row["model_id"]]: row for row in rows}


def _svg(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            '<style>text{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}.title{font-size:18px;font-weight:700}.subtitle{font-size:12px;fill:#526072}.label{font-size:11px}.value{font-size:11px;font-weight:600}.grid{stroke:#d9e0ea;stroke-width:1}.track{fill:#edf1f6}</style>',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            *body,
            "</svg>",
        ]
    )


def _bar_panels(path: Path, heading: str, subtitle: str, panels: list[tuple[str, dict[str, float], str]], *, percent: bool) -> None:
    width, height = 1260, 290
    panel_width, panel_gap, left = 390, 30, 20
    body = [f'<text class="title" x="20" y="28">{html.escape(heading)}</text>', f'<text class="subtitle" x="20" y="48">{html.escape(subtitle)}</text>']
    for panel_index, (title, values, unit) in enumerate(panels):
        x, y = left + panel_index * (panel_width + panel_gap), 78
        maximum = 100.0 if percent else max(values.values())
        body.extend((f'<text class="title" x="{x}" y="{y}">{html.escape(title)}</text>', f'<line class="grid" x1="{x}" y1="{y + 10}" x2="{x + panel_width - 5}" y2="{y + 10}"/>'))
        for index, model in enumerate(MODEL_ORDER):
            value = values[model]
            row_y, bar_x, bar_width = y + 31 + index * 28, x + 145, 190
            filled = bar_width * value / maximum if maximum else 0
            label = f"{value:.1f}%" if percent else f"{value:,.0f}{unit}"
            body.extend(
                (
                    f'<text class="label" x="{x}" y="{row_y + 11}">{html.escape(model)}</text>',
                    f'<rect class="track" x="{bar_x}" y="{row_y}" width="{bar_width}" height="14" rx="3"/>',
                    f'<rect x="{bar_x}" y="{row_y}" width="{filled:.2f}" height="14" rx="3" fill="{COLORS[model]}"/>',
                    f'<text class="value" x="{bar_x + bar_width + 6}" y="{row_y + 11}">{label}</text>',
                )
            )
    path.write_text(_svg(width, height, body), encoding="utf-8")


def main() -> None:
    comparisons = {name: _rows(_read_comparison(directory)) for name, directory in EXPERIMENTS.items()}
    quality = {
        "Choice": {model: comparisons["Choice"][model]["model_quality"]["metrics"]["accuracy_on_valid"] * 100 for model in MODEL_ORDER},
        "Noul": {model: comparisons["Noul"][model]["model_quality"]["metrics"]["accuracy_on_valid"] * 100 for model in MODEL_ORDER},
        "Score": {model: comparisons["Score"][model]["model_quality"]["metrics"]["score_nearest_tier_accuracy_on_usable"] * 100 for model in MODEL_ORDER},
    }
    reliability = {
        "Choice valid output": {model: comparisons["Choice"][model]["provider_metrics"]["valid_output_rate"] * 100 for model in MODEL_ORDER},
        "Noul valid output": {model: comparisons["Noul"][model]["provider_metrics"]["valid_output_rate"] * 100 for model in MODEL_ORDER},
        "Score consistency": {model: comparisons["Score"][model]["model_quality"]["metrics"]["score_distribution_consistency_rate"] * 100 for model in MODEL_ORDER},
    }
    latency = {
        task: {model: comparisons[task][model]["model_performance"]["latency_ms"]["p50"] for model in MODEL_ORDER}
        for task in EXPERIMENTS
    }
    tokens = {
        task: {model: comparisons[task][model]["model_performance"]["token_usage"]["reported_total_tokens_per_attempted"] for model in MODEL_ORDER}
        for task in EXPERIMENTS
    }
    CHARTS.mkdir(parents=True, exist_ok=True)
    _bar_panels(CHARTS / "quality.svg", "Decision quality", "Accuracy by task type. Higher is better; tasks are shown separately and are not a combined ranking.", [(task, values, "") for task, values in quality.items()], percent=True)
    _bar_panels(CHARTS / "reliability.svg", "Output reliability", "Choice and Noul use valid-output rate; Score uses score-distribution consistency. Higher is better.", [(task, values, "") for task, values in reliability.items()], percent=True)
    _bar_panels(CHARTS / "latency.svg", "Operational performance: p50 latency", "Milliseconds per attempted decision. Lower is better; results reflect the tested network path and provider route.", [(task, values, " ms") for task, values in latency.items()], percent=False)
    _bar_panels(CHARTS / "tokens.svg", "Operational performance: token usage", "Provider-reported total tokens per attempted decision. Lower is not necessarily lower cost across providers.", [(task, values, "") for task, values in tokens.items()], percent=False)


if __name__ == "__main__":
    main()
