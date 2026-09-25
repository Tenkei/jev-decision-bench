# Running experiments

See the [Experiment index](index.md) to select a versioned workload and find
its task type, dataset, JEV interface, and implementation status. Run only an
experiment marked ready; deferred entries describe planned work, not runnable
commands.

This guide runs a prepared experiment against native JEV and a conventional
LLM. `banking77-choice-v0` sends 3,080 intent-routing decisions;
`hatecheck-noul-v0` sends 3,728 binary policy decisions. Each model run also
makes one unscored preflight request.

## 1. Hardware and network

The v0 benchmark uses remote APIs. It does **not** require a GPU or a local
model server.

Latency metrics include your network path to the provider. Run both models from
the same machine and network if you intend to compare p50/p95 latency.

## 2. Create the local Python environment

From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-deps -e .
```

The benchmark has no third-party runtime dependency today. Use
`.venv/bin/python -m pip`, rather than a bare `pip`, so the command remains
correct after moving or renaming the repository.

Verify the command is available:

```bash
.venv/bin/jev-decision-bench --help
```

## 3. Configure the model routes

The tracked configuration files are editable starting points. Do not put
secrets in either JSON file; provide them only through the named environment
variable.

### JEV

Set the API key named by `api_key_env` in
`configs/typesafe-direct.json`:

```bash
export TYPESAFE_API_KEY='replace-with-your-key'
```

Discover the model identifiers available to your TypeSafe account, then replace
`replace-with-typesafe-model-id` in `configs/typesafe-direct.json` with the
exact returned name:

```bash
curl --fail-with-body --silent --show-error \
  --header "Authorization: Bearer $TYPESAFE_API_KEY" \
  https://api.typesafe.ai/v1/models | python3 -m json.tool
```

Edit this field in the tracked configuration, using one of the returned model
names:

```json
{
  ...,
  "api_key_env": "TYPESAFE_API_KEY",
  "model_id": "replace-with-typesafe-model-id",
  "model_revision": null,
  ...,
}
```

### OpenAI-compatible LLM

`configs/openai-compatible.json` is the editable configuration for any
provider offering an OpenAI-compatible Chat Completions endpoint. Set its
endpoint, model ID, key environment-variable name, and any known model revision
or token pricing.

`configs/gemini.json` is only an example of that route. Its `api_key_env`
field identifies the environment variable that must hold the provider key:

```bash
grep '"api_key_env"' configs/gemini.json
#   "api_key_env": "GEMINI_API_KEY",

export GEMINI_API_KEY='replace-with-your-key'
```

For another provider, inspect its selected configuration in the same way and
export the variable named by `api_key_env`. Never put the key itself in JSON.

The default response mode is `json_schema`. Keep `temperature` at `0` for the
first comparison. The adapter asks the LLM for exactly this result shape:

```json
{
  "choice": "one BANKING77 intent ID",
  "confidence": 0.0
}
```

`confidence` must be between `0` and `1`. It is recorded as a verbalized
top-label probability; it is not presented as a native JEV distribution.

### AWS / Amazon Bedrock

For a self-service Bedrock API-key setup and a matching OpenAI-compatible
configuration, see [AWS / Amazon Bedrock](aws.md). It creates IAM resources and
may incur inference charges, so run its commands only from an appropriately
authorized AWS identity.

## 4. Prepare the immutable experiment package

Preparation downloads the pinned source files, checks their SHA-256 values and
row count, compiles the experiment's decision records, and writes an immutable
package below `artifacts/`.

```bash
.venv/bin/jev-decision-bench prepare --experiment banking77-choice-v0
```

For the HateCheck Noul experiment, use:

```bash
.venv/bin/jev-decision-bench prepare --experiment hatecheck-noul-v0
```

The command prints the package directory. Save it for every later stage:

```bash
export PACKAGE='artifacts/experiments/banking77-choice/v0/<package-hash>'
```

Do not edit a prepared package. Re-running `prepare` with the same inputs fails
instead of overwriting it; that is intentional.

## 5. Run JEV and the LLM

Each command creates a new run directory under `artifacts/runs/` and prints its
path. Its name identifies the prepared experiment, provider, and model:
`banking77-choice-v0--amazon-bedrock--global-openai-gpt-5-6-luna--<timestamp>--<nonce>`.
Requests are serial, one decision per provider call, so the results have
directly interpretable per-decision latency.

Before calling a provider, the runner checks for an existing run with the same
experiment-package hash, model-configuration hash, and adapter version. It
stops rather than spending on an accidental duplicate. To intentionally repeat
the treatment—for example, to measure run-to-run variation—add `--repeat` to
the command.

If a terminal disconnects or the process is interrupted, continue the same
compatible run rather than starting again:

```bash
.venv/bin/jev-decision-bench run \
  --package "$PACKAGE" \
  --model-config configs/openai-compatible.json \
  --resume artifacts/runs/<interrupted-run-id>
```

Resume verifies the experiment package, model configuration, and adapter
version; it skips every decision that already has a terminal prediction. A
successful recorded preflight is not repeated.

While a run is active, the CLI displays completed test cases and percentage,
for example `Progress: 1,542/3,080 (50.1%)`. Failed or invalid decisions still
advance this count because they are completed, recorded test cases.

Run JEV first:

```bash
.venv/bin/jev-decision-bench run \
  --package "$PACKAGE" \
  --model-config configs/typesafe-direct.json
```

Save the printed path:

```bash
export JEV_RUN='artifacts/runs/<jev-run-id>'
```

Then run the conventional LLM:

```bash
.venv/bin/jev-decision-bench run \
  --package "$PACKAGE" \
  --model-config configs/openai-compatible.json
```

Replace the configuration path with a provider-specific file when you choose
one; for example, `configs/gemini.json` is a configured Gemini example.

```bash
export LLM_RUN='artifacts/runs/<llm-run-id>'
```

Before scoring test records, the runner performs one unscored, in-domain
synthetic preflight call to verify credentials and response shape. It does not
reuse a dataset record or include a gold answer. A failed preflight produces a
`preflight_failed` run and no BANKING77 requests.

Transient HTTP failures—timeouts, 429s, and 5xx responses—are retried up to
the configured `max_retries`. Invalid labels, malformed JSON, and invalid
confidence values are recorded as invalid outputs and are never repaired or
retried.

## 6. Score each run offline

Scoring makes no provider calls. It joins saved predictions to the prepared
package, uses that experiment's recommended evaluation policy by default, and
writes a separate immutable evaluation artifact. The run evidence is never
rewritten, so the same run can be evaluated again with a different policy.

```bash
.venv/bin/jev-decision-bench score --package "$PACKAGE" --run "$JEV_RUN"
.venv/bin/jev-decision-bench score --package "$PACKAGE" --run "$LLM_RUN"
```

To use an explicit policy—for example, a different Noul operating threshold—
pass `--evaluation`. Score snapshots this JSON configuration beside the result.

```bash
.venv/bin/jev-decision-bench score \
  --package "$PACKAGE" \
  --run "$LLM_RUN" \
  --evaluation configs/evaluations/hatecheck-threshold-0.7.json
```

The report includes coverage, accuracy and macro-F1 on valid predictions,
top-label Brier/ECE, p50/p95 latency, known cost, and explicit output-status
counts. Provider failures are not silently counted as incorrect predictions.

## 7. Compare the completed runs

Only completed runs made from the same experiment-package hash can be compared.
Pass JEV explicitly as the baseline; every run in `--runs` is reported with
its unchanged absolute metrics and metrics relative to that baseline.

```bash
.venv/bin/jev-decision-bench compare \
  --package "$PACKAGE" \
  --baseline "$JEV_RUN" \
  --runs "$LLM_RUN"
```

The command writes `comparison.json` under
`artifacts/comparisons/<comparison-id>/`. Deltas are candidate minus baseline,
so negative Brier/ECE values are better. Latency and cost are ratios to the
baseline, so values below `1` are faster or cheaper. Cost ratios are `null`
when either run has no known cost.

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
  comparisons/<comparison-id>/
    comparison.json
```

Artifacts are intentionally ignored by Git. `events.jsonl` retains request
payloads and provider responses for auditability, but never writes API keys or
authorization headers. Treat it as potentially sensitive because it contains
the public dataset text and model outputs.

Every run also writes a sanitized `model-config.json` and its SHA-256 in
`run-manifest.json`. The tracked configuration is editable input; the run
snapshot is the evidence of the exact endpoint, model, decoding, retry, and
pricing configuration used in that run. Inline API-key fields are rejected;
only an `api_key_env` name may be persisted.

Common invocation settings such as `max_tokens`, timeout, and retry count are
top-level fields in the configuration. Adapters map the shared
`max_tokens` value to the provider's wire field—for example,
`max_output_tokens` for Responses—so it has one meaning across models.
`model_config` is reserved for provider-specific API arguments, such as
`temperature`, `reasoning_effort`, or a provider service tier. It is passed
through without allowing it to replace the model, prompt, token limit, or
output contract. Omit an argument that the selected model does not support.

The first implementation does not run derived BANKING77 Noul membership
experiments, Score tasks, parallel execution, or native batching. Those
capabilities require a new experiment version and separate documentation before
being used in a headline comparison.

## Next

After both model runs are complete, use the scoring and comparison commands
above to publish the matched result artifacts.
