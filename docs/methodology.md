# Methodology

This document defines how an experiment is prepared, run, and compared. Its
main rule is simple:

> An experiment is an immutable decision package. A model run is one
> repeatable attempt by one configured system to answer that package. An
> evaluation is a separately versioned interpretation of saved predictions.

This separation prevents a model-specific prompt, label map, negative sample,
or metric setting from quietly changing the task after results exist.

## Terms

| Term | Meaning |
| --- | --- |
| **Experiment** | A versioned benchmark condition: dataset split, benchmark items, rubric, task contract, and preflight fixture. Example: `banking77-choice-v0`. |
| **Benchmark item** | One fixed state, question, allowed answers, gold answer, and evaluation metadata. It is shared by every model. |
| **Model configuration** | A provider, model revision, adapter, decoding settings, endpoint, and cost configuration. |
| **Run** | One model configuration answering every benchmark item in one frozen experiment. |
| **Evaluation** | A versioned scoring policy applied offline to one completed run. A run may have multiple evaluations. |
| **Comparison** | A report that joins completed runs only on the same experiment version and record IDs. |

## Fair comparison rules

- JEV and every LLM receive the same state, question, label vocabulary, label
  definitions, and allowed `other`/review path where one exists.
- Conventional LLMs return schema-constrained JSON. Their reported
  probabilities are recorded as **verbalized probabilities**, not treated as
  equivalent to JEV's native decision distribution.
- The benchmark does not grant one system tools, retrieval, demonstrations, or
  chain-of-thought that the other system does not receive.
- Public datasets are useful for reproducibility, but do not establish
  contamination resistance. A future private, human-adjudicated test set will
  supply the headline evaluation.

## Experiment Steps

**Pin** names the immutable artifact that owns the step. **Configuration**
states how often its settings are defined: once per experiment, once for each
model run, or once for a comparison.

| Step | Pin | Configuration | Output |
| --- | --- | --- | --- |
| Select source data and split | Experiment version | Per experiment | Dataset source revision, hashes, row IDs |
| Normalize source rows | Experiment version | Per experiment | Canonical source records |
| Write label definitions and answer rubric | Experiment version | Per experiment | Fixed rubric version and hash |
| Compile benchmark items | Experiment version | Per experiment | Model-independent Choice items in v0 |
| Generate Noul negatives | Deferred Noul experiment | Per experiment | Frozen candidate pairs, sampler version, seed, and hash |
| Recommend an evaluation policy | Experiment source | Per experiment | Editable default evaluation configuration; not part of the package hash |
| Render a benchmark item to a provider request | Model run, referencing an experiment version | Per model | Native JEV request or LLM JSON-schema request |
| Call the provider and record evidence | Model run, referencing an experiment version | Per model | Raw request/response, usage, timing, errors |
| Validate model results | Experiment version for rules; model run for result | Per model | Standardized results or explicit failures |
| Score a completed run | Evaluation configuration and saved run evidence | Per evaluation | Immutable evaluation configuration snapshot and aggregate scores |
| Compare runs | Comparison manifest, referencing experiment and run IDs | Per comparison | Matched model table and report |

## Experiment parameters

Every parameter name identifies its owner. There is no unqualified `version`:
an experiment version, dataset revision, pipeline version, and model revision
are separate facts.

### Experiment and pipeline parameters

These parameters describe how source data becomes decision records and how the
records are evaluated. They are saved in `experiment-manifest.json`.

| Parameter | Example for `banking77-choice-v0` | Purpose |
| --- | --- | --- |
| `experiment_id` | `banking77-choice` | Names the benchmark condition independent of its revision. |
| `experiment_version` | `v0` | Identifies one reproducible definition of the BANKING77 experiment. |
| `pipeline_version` | Source-control revision of this benchmark | Identifies the preparation and validation code. |
| `normalizer_version` | `banking77-normalizer@…` | Defines stable source IDs and permitted input normalization. |
| `rubric_version` | `banking77-intents@…` | Defines the 77 fixed intent IDs and their reviewed definitions. |
| `record_sets` | `choice` | Names the decision views compiled from the source rows. |
| `negative_sampler_version` | `not_applicable` | Reserved for the deferred Noul experiment. |
| `decision_contract_version` | Canonical Choice record schema | Defines the model-independent state, question, criteria, and gold answer. |
| `task.contract_version` | Allowed labels, probability requirements, failure behavior | Defines how provider output becomes a valid prediction or an explicit failure. |
| `preflight_sha256` | SHA-256 over the compiled synthetic request | Pins the unscored route-and-shape check used by every new run. |
| `experiment_package_hash` | SHA-256 over manifest and compiled records | Identifies the complete experiment package used by every run. |

### Dataset parameters

These parameters identify the upstream data independently of the benchmark
pipeline. They are included in `experiment-manifest.json` and source metadata.

| Parameter | Example for `banking77-choice-v0` | Purpose |
| --- | --- | --- |
| `dataset_name` | `BANKING77` | Human-readable source dataset name. |
| `dataset_source` | `PolyAI-LDN/task-specific-datasets` | Upstream repository or distribution location. |
| `dataset_license` | `CC-BY-4.0` | Terms governing use and redistribution. |
| `dataset_revision` | Upstream Git commit or release identifier | Resolves the exact upstream content. |
| `dataset_split` | `banking_data/test.csv` | Specifies the source rows eligible for evaluation. |
| `dataset_file_hashes` | SHA-256 for `test.csv` and `categories.json` | Detects changed or corrupted source files. |
| `dataset_row_count` | `3,080` | Guards against an unexpectedly incomplete or expanded split. |

### Model-run parameters

These parameters describe one system invocation. They are saved in
`run-manifest.json`, never used to redefine the experiment package.

| Parameter | Example | Purpose |
| --- | --- | --- |
| `run_id` | Unique generated ID | Identifies one non-overwriting execution attempt. |
| `provider` | `typesafe`, `openai`, or another provider | Identifies the service that executed the model. |
| `endpoint` | Provider API base URL | Identifies the transport target. |
| `model_id` | Provider model identifier | Identifies the configured model. |
| `model_revision` | Provider-reported snapshot, if available | Distinguishes a mutable model alias from a specific released revision. |
| `adapter_version` | `jev-native@…` or `llm-json@…` | Identifies the canonical-record renderer and response normalizer. |
| `response_mode` | Native JEV decision API or JSON-schema output | Records the system's response transport and contract. |
| `decoding_parameters` | Temperature, seed, token limit | Makes generation settings explicit for conventional LLMs. |
| `execution_parameters` | Concurrency, timeout, transport-retry policy | Records execution behavior without changing logical decisions. |
| `cost_basis` | Provider-reported cost or published tariff revision | Explains how cost metrics were obtained. |
| `model_config_sha256` | SHA-256 of the sanitized model config | Pins the exact endpoint, model, decoding, retry, and cost settings used by the run. |

### Evaluation parameters

Evaluation settings are not part of the prepared package or run. Each experiment
ships a recommended `default-evaluation.json`, but `score --evaluation` may use
another compatible policy to re-evaluate the same saved predictions.

| Parameter | Purpose |
| --- | --- |
| `evaluation_id` / `evaluation_version` | Names one scoring-policy revision. |
| `task_type` | Prevents applying a Choice policy to Noul evidence, or vice versa. |
| `positive_threshold` | For Noul, derives the evaluated binary answer from saved true probabilities. |
| `calibration_bins` | Sets the number of ECE bins. |
| `evaluation_config_sha256` | Identifies the exact policy snapshot written with scores. |

## Experiment metrics

All experiments report the common metrics below. Decision-specific metrics are
added rather than collapsed into a single score.

| Applies to | Metrics | Purpose |
| --- | --- | --- |
| Every experiment | Exact-decision accuracy, valid-output rate, p50/p95 latency, cost per decision | Measures correctness, response-contract reliability, elapsed time, and reported or configured cost. Invalid output and provider failures remain separate from wrong answers. |
| `Choice` | Macro-F1, confusion matrix | Shows performance across labels, including errors between similar classes. |
| `Noul` | Precision, recall, AUROC, Brier score, expected calibration error, confidence-based risk/coverage | Measures binary decision quality and the usefulness of confidence. |
| `Score` | Exact-tier accuracy, ordinal mean absolute error, calibration over rubric levels | Respects the ordered meaning of rubric levels. |

## Lifecycle

An experiment moves through four stages. `prepare` produces the experiment
package; each `run` references that package; `score` works from saved run
evidence; and `compare` joins only compatible scored runs.

```text
prepare experiment once
        │
        ▼
immutable experiment package
        │
        ├── run JEV ──────────────► validate ─┐
        ├── run LLM A ────────────► validate ─┼──► evaluate ─► compare
        └── run LLM B ────────────► validate ─┘     ▲
                                                evaluation policy
```

### 1. Prepare an experiment

Preparation produces an experiment package before any model is called.

1. **Resolve the source.** Record the dataset URL, source revision, license,
   split name, expected row count, and SHA-256 hashes. Refuse to proceed if
   a later download does not match the pinned input.
2. **Normalize source rows.** Produce stable source IDs and preserve the
   original text and original label. Normalization rules are versioned, not
   changed after a run.
3. **Freeze the rubric.** Map every label to a reviewed definition and fix its
   order. For ordinal `Score` tasks, also fix the level order and its meaning.
4. **Compile task benchmark items.** An item contains the state,
   question, typed answer space, gold answer, and source reference. It contains
   no provider-specific prompt syntax.
5. **Compile the preflight fixture.** The experiment supplies an in-domain,
   unscored state used only to prove the selected route can return a valid
   response shape.
6. **Write `experiment-manifest.json`.** Include hashes for every package artifact.
   This manifest is immutable once the first model run starts; a change creates
   a new experiment version.

### 2. Run one model configuration

Every model receives the exact same benchmark items from the prepared package.

1. **Resolve the model configuration.** Record provider, endpoint, model and
   revision, adapter version, temperature, seed when supported, token limit,
   concurrency, timeout, retry policy, and cost basis.
2. **Preflight without test scoring.** Execute the package's pinned synthetic
   fixture to verify credentials and response shape. A preflight failure stops
   the run and is recorded; it cannot alter the experiment package.
3. **Render each decision.** The adapter turns the shared benchmark item into
   the model's native transport while preserving the fair-comparison rules:
   JEV receives its native decision request and an LLM receives the same
   bounded task through a strict JSON schema.

4. **Execute and append evidence.** Store every raw request, raw response,
   provider usage record, wall-clock duration, and retry/error event under a
   new run ID. A run never overwrites an earlier run.
5. **Normalize and validate.** Convert only valid provider output to the common
   prediction contract. Invalid labels, malformed probability vectors, and
   incomplete responses become explicit failures—not repaired answers.
6. **Finalize the run manifest.** Record totals for attempted, successful,
   invalid, failed, and unattempted decision records. An incomplete run remains
   incomplete; it is not silently compared as if missing records were wrong.
   Save a sanitized `model-config.json` beside the manifest and record its
   SHA-256. Configuration may name an API-key environment variable, but cannot
   contain an API key value.

### 3. Score and compare — repeatable from saved evidence

1. **Evaluate one run.** Join model results to the gold answers in benchmark
   items by stable decision ID. Apply a chosen compatible evaluation policy
   without calling a provider.
2. **Publish evaluation artifacts.** Snapshot the policy and emit aggregate
   metrics, calibration data, confusion matrices, costs, and latency summaries as JSON.
3. **Compare runs.** Require the same experiment-manifest hash and an explicit
   baseline. Preserve absolute metrics and add deltas or ratios relative to the
   baseline; omissions and failures remain separate from accuracy.

## Shared data formats

Every provider receives the same benchmark item and produces a model result in
the same shape. These formats keep the task fixed while allowing JEV and an
LLM to use different APIs.

### Benchmark item

This is one immutable question from the experiment package. It includes all
information needed to evaluate the answer, but no provider-specific request
syntax.

```json
{
  "decision_id": "banking77-choice-v0:test:000123",
  "source_id": "banking77:test:000123",
  "task_type": "choice",
  "state": {"query": "…"},
  "question": "Which banking intent best matches this query?",
  "criteria": {
    "card_arrival": "A requested physical card has not arrived.",
    "…": "The remaining frozen BANKING77 intent definitions."
  },
  "gold": "card_arrival",
  "metadata": {
    "dataset_row_index": 123,
    "record_set": "choice"
  }
}
```

### Model result

After validation, every provider response is saved in this common shape. An
invalid response is recorded as a failure rather than being repaired.

```json
{
  "decision_id": "banking77-choice-v0:test:000123",
  "status": "valid",
  "answer": "card_arrival",
  "selected_probability": 0.82,
  "probability_provenance": "native",
  "model": {"provider": "…", "id": "…", "revision": "…"},
  "timing_ms": 0,
  "cost_usd": null
}
```

`probability_provenance` is `native` for JEV's decision API and `verbalized`
for an LLM that writes top-label confidence in JSON. A missing or invalid
probability is retained as such; it is never made up by the evaluator.

## Next

Read the [Architecture](architecture.md) to see how packages, task contracts,
runs, and evaluations map to the codebase. Then use the
[Experiment index](index.md) to select a versioned workload.
