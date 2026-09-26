# Analyzing and sharing results

Use this guide after completing compatible JEV and LLM runs from the same
prepared experiment package. It covers offline scoring, comparison, publication
artifacts, and re-evaluation from saved evidence.

## 1. Score each completed run

Scoring makes no provider calls. It joins saved normalized predictions to the
prepared package, uses that experiment's recommended evaluation policy by
default, and writes a separate evaluation artifact. The run evidence is never
rewritten, so a run can later be evaluated with a different policy.

```bash
.venv/bin/jev-decision-bench score --package "$PACKAGE" --run "$JEV_RUN"
.venv/bin/jev-decision-bench score --package "$PACKAGE" --run "$LLM_RUN"
```

To use an explicit policy—for example, a different Noul operating threshold—
pass `--evaluation`. The exact policy is copied beside the score output.

```bash
.venv/bin/jev-decision-bench score \
  --package "$PACKAGE" \
  --run "$LLM_RUN" \
  --evaluation configs/evaluations/hatecheck-threshold-0.7.json
```

Metrics depend on the task contract. Choice reports decision quality and
top-label calibration; Noul adds binary discrimination and calibration metrics;
Score evaluates the final ordinal score, score-distribution consistency,
malformed responses, and the complete level distribution. See
[Methodology](methodology.md#experiment-metrics) for metric definitions.

## 2. Compare compatible runs

Only completed runs made from the same experiment-package hash can be compared.
Pass JEV explicitly as the baseline.

```bash
.venv/bin/jev-decision-bench compare \
  --package "$PACKAGE" \
  --baseline "$JEV_RUN" \
  --runs "$LLM_RUN"
```

Comparison always reads saved run evidence directly. It reports provider health
and model performance—latency, reported token use, and known cost—even before
a run has been scored. When all selected runs have scores under the same policy,
it also includes decision-quality and calibration metrics. To require a
particular policy, pass `--evaluation`; comparison then stops if any selected
run lacks that matching score artifact.

```bash
.venv/bin/jev-decision-bench compare \
  --package "$PACKAGE" \
  --baseline "$JEV_RUN" \
  --runs "$LLM_RUN" \
  --evaluation configs/evaluations/hatecheck-threshold-0.7.json
```

Comparison writes `comparison.json` under:

```text
artifacts/comparisons/<experiment-id>-<version>--<package-hash>--<comparison-id>/
```

Quality and provider-health deltas are candidate minus baseline. Latency, known
cost, and reported token-use metrics are ratios to the baseline, so values below
`1` are lower. Token ratios are operational evidence, not cross-provider cost
equivalence: providers can use different tokenizers and account for cached or
reasoning tokens differently.

## 3. Export a result for publication

Export creates two forms of the same scored comparison:

- A small summary below `results/` that should be committed with the project.
- A sanitized `.tar.gz` audit archive below `artifacts/exports/` that should be
  uploaded as a release asset or external archive.

```bash
.venv/bin/jev-decision-bench export \
  --package "$PACKAGE" \
  --comparison artifacts/comparisons/<comparison-id>
```

The tracked summary contains:

```text
results/<comparison-id>/
  README.md
  comparison.json
  experiment-manifest.json
  evaluation-config.json
  run-manifests/
```

The audit archive additionally contains the prepared decision records and each
run's normalized `predictions.jsonl`. It can therefore be rescored and
compared without contacting a model provider. It excludes raw provider events,
logs, credential values, and downloaded source copies.

A run manifest preserves the adapter, model, endpoint, response format, token
limit, retries, and provider-specific settings such as temperature or reasoning
effort. Sensitive configuration fields are redacted, while an `api_key_env`
name is retained.

Use `--output-dir` to choose the archive destination, `--results-root` for the
tracked-summary location, and `--overwrite` to replace both outputs for the
same comparison ID.

After unpacking the archive, use its `reproduce.sh` script to rebuild scores
and a comparison offline. The script invokes the current
`jev-decision-bench` executable (or `JEV_DECISION_BENCH=/path/to/executable`
when supplied), so the saved evidence can be re-evaluated under future scoring
logic. To reproduce the originally published figures exactly, use the benchmark
revision recorded in the exported experiment manifest.

## Artifact layout

```text
artifacts/
  experiments/<experiment-id>/<experiment-version>/<package-hash>/
    experiment-manifest.json
    records.<record-set>.jsonl
    preflight.json
    rubric.json
    source/
  runs/<run-id>/
    run-manifest.json
    events.jsonl
    predictions.jsonl
  evaluations/<run-id>/<evaluation-id>--<evaluation-config-hash>/
    evaluation-config.json
    scores.json
  comparisons/<experiment-id>-<version>--<package-hash>--<comparison-id>/
    comparison.json
  exports/
    <comparison-id>.tar.gz
results/
  <comparison-id>/
    README.md
    comparison.json
    experiment-manifest.json
    evaluation-config.json
    run-manifests/
```

Artifacts are intentionally ignored by Git. `events.jsonl` retains request
payloads and provider responses for auditability, but never writes API keys or
authorization headers. Treat it as potentially sensitive because it contains
dataset text and model outputs.

The tracked configuration is editable input; the sanitized run snapshot and
its SHA-256 are the evidence of the exact endpoint, model, decoding, retry,
and pricing configuration that actually ran. Inline API-key fields are
rejected; only an `api_key_env` name may be persisted.

## Next

See [Methodology](methodology.md) for the evaluation rules and
[Architecture](architecture.md) for how packages, runs, scores, comparisons,
and exports map to the codebase.
