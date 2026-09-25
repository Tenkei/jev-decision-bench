# Architecture

`jev-decision-bench` separates the decision problem, a model execution, and
the evaluation of that execution. This makes it possible to compare providers
fairly, preserve raw evidence, and re-evaluate a completed run without calling
a model again.

```mermaid
flowchart TD
    source["Versioned experiment source<br/>spec · rubric · source pins<br/>preflight fixture · recommended evaluation"]
    package["Immutable experiment package<br/>records · rubric · preflight"]
    runner["Runner"]
    contract["Task contract<br/>Choice · Noul · future Score"]
    adapter["Model adapter"]
    provider["Provider API"]
    run["Run evidence<br/>pinned route config · events · predictions"]
    policy["Evaluation policy"]
    evaluation["Immutable evaluation artifact<br/>evaluation-config.json · scores.json"]

    source -->|prepare| package
    package -->|run| runner
    runner -->|resolves task type| contract
    runner -->|executes records| adapter
    contract -->|renders and validates| adapter
    adapter -->|request| provider
    provider -->|response| adapter
    adapter --> run
    run -->|score| evaluation
    policy -->|configures| evaluation

    classDef immutable fill:#d1fae5,stroke:#047857,color:#064e3b,stroke-width:2px
    class package,run,evaluation immutable
```

## Main boundaries

| Component | Responsibility | Does not own |
| --- | --- | --- |
| Experiment source | Dataset pins, rubric, task type, preflight fixture, recommended evaluation policy | Provider prompts or credentials |
| Prepared package | Immutable compiled records and hashes | Model route or evaluation result |
| Task contract | Semantics of Choice, Noul, or Score: schemas, native JEV shape, parsing, validation, and scorer selection | HTTP transport or dataset download |
| Adapter | API transport: request serialization, authentication, response extraction, retries at the runner boundary | Whether an answer is a valid Choice or Noul decision |
| Run | One model route answering one exact package | Scoring policy |
| Evaluation | Offline metric policy applied to saved predictions | Provider calls or changes to run evidence |

## Experiment package

`prepare` downloads only the source files pinned by the experiment spec,
verifies their hashes, and writes a package such as:

```text
artifacts/experiments/<experiment-id>/<version>/<package-hash>/
  experiment-manifest.json
  records.choice.jsonl or records.noul.jsonl
  rubric.json
  preflight.json
  source/
```

The package hash covers the compiled records, rubric, preflight fixture, and
manifest. The recommended `default-evaluation.json` remains in the versioned
experiment source, rather than the package, because evaluation policy may
change without changing the decision task.

`preflight.json` is an in-domain, unscored fixture. It proves that a configured
provider route can produce a structurally valid decision before the run sends
any test records.

## Task contracts

Task contracts are registered by task type in `src/jev_decision_bench/task_contracts.py`.
They are the only layer that interprets a decision's meaning.

| Contract | Current result shape | Key validation |
| --- | --- | --- |
| Choice | `choice` and `confidence` | Choice must be a frozen rubric key; confidence must be in `[0, 1]`. |
| Noul | `answer` and `probability_true` | Answer must agree with the recorded true probability at the execution threshold. |
| Score | Planned | Will define ordered levels and ordinal validation without altering Choice or Noul paths. |

For native JEV, a contract renders the appropriate question type. For an LLM,
it supplies the bounded JSON schema and instructions. Adapters then send those
rendered requests through their provider-specific APIs.

Adding a task type therefore means adding a contract and an experiment; it
does not require task-type branches throughout the runner or adapters.

## Run evidence

`run` snapshots the sanitized route configuration and links it to the exact
experiment-package hash. It appends one prediction and one audit event per
logical decision. Raw provider requests and responses are retained in
`events.jsonl`; normalized terminal outcomes are retained in
`predictions.jsonl`.

The runner rejects accidental duplicate runs with the same package, route
configuration, and adapter version. A stopped run can be resumed only when all
three still match.

## Evaluation and comparison

`score` reads saved predictions plus the package's task contract. By default
it uses the experiment's `default-evaluation.json`; `--evaluation` selects a
different compatible policy. It creates a separate artifact:

```text
artifacts/evaluations/<run-id>/<evaluation-id>--<evaluation-config-hash>/
  evaluation-config.json
  scores.json
```

This lets the same Noul probabilities be evaluated at another threshold, or
with another calibration-bin count, without rerunning the model. The policy is
snapshotted with each result, so old and new evaluations remain comparable and
auditable.

`compare` accepts only completed runs from the same package and evaluation
policy. It preserves their absolute metrics and adds deltas or ratios relative
to an explicitly selected baseline.

## Extension points

To add an experiment, create its versioned source directory with a pinned
dataset spec, rubric, preflight fixture, default evaluation policy, and a
compiler registration. To add a new provider, implement an adapter that
transports the generic rendered decision request. To add `Score`, implement
its contract and evaluator, then add a Score experiment; existing adapters,
runs, and evaluations stay unchanged.

## Next

Read the [Methodology](methodology.md) for the reproducibility rules, then use
[Running experiments](running-experiments.md) to prepare and execute a model run.
