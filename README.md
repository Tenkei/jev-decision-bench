# jev-decision-bench

`jev-decision-bench` evaluates conventional language models on tasks designed around
[TypeSafe JEV](https://typesafe.ai/)'s bounded-decision interface. It is not a
general LLM leaderboard and it does not ask JEV to take benchmarks designed for
free-form text generation.

The central question is:

> Given the same state, decision rubric, and permitted answers, how accurately,
> reliably, quickly, and cheaply can JEV and conventional LLMs make a
> code-consumable decision?

## Methodology

Most LLM evaluations begin with a benchmark designed for general-purpose
language models and then ask every system, including JEV, to take it. This
project deliberately reverses that direction.

We first design the workload around the kinds of decisions JEV is intended to
make: a bounded state, an explicit rubric, and a finite answer space. We then
ask conventional LLMs to take the *same* decision tasks through a normalized,
schema-constrained adapter. The purpose is not to test whether JEV can imitate
free-form generation. It is to measure how well LLMs can perform the work JEV
is designed to do.

This method does not presume anything about JEV's private implementation or
that JEV will win. It makes the comparison operationally fair: all systems see
the same evidence, labels, and decision criteria; only their native interfaces
and reported uncertainty differ.

## What this benchmark measures

Every example is a **decision item**: a state, a question, a finite answer
space, and a gold label. The answer space determines the native JEV interface:

| Decision geometry | JEV interface | Example |
| --- | --- | --- |
| Two mutually exclusive outcomes | Noul | Is this query about a card that has not arrived? |
| Unordered, closed-set alternatives | Choice | Which banking intent best matches this query? |
| Ordered rubric levels | Score | How urgent is this request: low, medium, high, or critical? |

The benchmark treats all three as bounded decisions, but preserves their
semantics. A `Score` is an ordered rubric position, not an arbitrary number;
using a `Score` for categories with no meaningful order is invalid. A two-level
`Score` is equivalent to a `Noul` and is not scored as a separate capability.

## Experiment overview

The benchmark is organized by decision task family. `Dataset used` identifies
the selected source where one exists; `TBD` means the family is part of the
benchmark design but no dataset has been adopted yet.

| Task family | JEV task type | Dataset used | Description | Status |
| --- | --- | --- | --- | --- |
| Intent / queue routing | `Choice` / `Noul` | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | Correctly route a noisy support or operations request, either by choosing one queue or by testing membership in a candidate queue; include an `other`/review option where the task permits it. | Choice implementation ready; Noul membership is deferred to the next milestone. |
| Semantic attribute extraction | Multiple `Noul` / `Choice` | TBD | Extract several controlled features from the same input—for example topic, affected product, stated deadline, and request type—without generating prose. | Backlog — dataset selection required. |
| Policy & guardrail decisions | `Noul` | TBD | Evaluate eligibility, safety, contradiction, and action-blocking judgments. | Backlog — dataset selection required. |
| Tool / action approval | `Choice` / `Noul` | TBD | Pick or approve the right next action amid distractors; actions are never executed. | Backlog — dataset selection required. |
| Severity / priority assessment | `Score` | TBD | Evaluate ordered judgments such as urgency, risk, or business impact. | Backlog — dataset selection required. |
| Search relevance / ranking | `Noul` / `Score` | TBD | Determine query-document relevance and rank candidates. | Backlog — dataset selection required. |
| Generic requirement verification | `Noul` / `Choice` | TBD | Determine whether any artifact or state satisfies a named, explicit requirement, independent of its domain. | Backlog — dataset selection required. |
| OOD uncertainty & abstention | All three | TBD | Measure whether confidence falls and review is selected when evidence or policy is missing. | Backlog — dataset selection required. |

BANKING77 has no intrinsic ordinal label, so it is intentionally not used for a
`Score` experiment. See [Experiment index](docs/index.md) for the exact v0
contracts and frozen-protocol requirements. See [Methodology](docs/methodology.md)
for the implementation lifecycle and artifact contracts. See
[Running experiments](docs/running-experiments.md) for the environment setup and
step-by-step runbook.

## Fair comparison rules

- JEV and every LLM receive the same state, question, label vocabulary, label
  definitions, and allowed `other`/review path where one exists.
- Conventional LLMs return schema-constrained JSON. Their reported
  probabilities are recorded as **verbalized probabilities**, not treated as
  equivalent to JEV's native decision distribution.
- The benchmark does not grant one system tools, retrieval, demonstrations, or
  chain-of-thought that the other system does not receive.
- Dataset versions, prompts, label definitions, model revisions, parameters,
  raw responses, timing, and costs are recorded in a run manifest.
- Public datasets are useful for reproducibility, but do not establish
  contamination resistance. A future private, human-adjudicated test set will
  supply the headline evaluation.

## Metrics

All experiments report exact-decision accuracy, valid-output rate, p50/p95
latency, and cost per decision. Metrics appropriate to the decision geometry
are added rather than collapsed into a single score:

- **Choice:** macro-F1 and confusion matrix.
- **Noul:** precision, recall, AUROC, Brier score, expected calibration error,
  and confidence-based risk/coverage.
- **Score:** exact-tier accuracy, ordinal mean absolute error, and calibration
  over rubric levels.

## Status

The repository currently documents the benchmark contract and planned first
experiments. The Choice-only BANKING77 implementation can prepare a pinned
package, call configured providers, score saved evidence, and compare runs; no
live model result has been published.

## Choice-only BANKING77 commands

```bash
PYTHONPATH=src .venv/bin/python -m jev_decision_bench.cli prepare \
  --experiment banking77-choice-v0

PYTHONPATH=src .venv/bin/python -m jev_decision_bench.cli run \
  --package artifacts/experiments/banking77-choice/v0/<package-hash> \
  --model-config configs/typesafe-direct.json

PYTHONPATH=src .venv/bin/python -m jev_decision_bench.cli score \
  --package artifacts/experiments/banking77-choice/v0/<package-hash> \
  --run artifacts/runs/<run-id>
```

Select a tracked model configuration, add only the provider details required
for that route, and set the API-key environment variable named by `api_key_env`.
Configuration files never contain API keys.

## Documentation

The repository follows a small documentation set: this README explains the
project, while the `docs/` pages describe the portfolio, method, execution,
and provider-specific setup.

- [Experiment index](docs/index.md) — registered experiments, task types,
  datasets, and status.
- [Methodology](docs/methodology.md) — experiment packages, benchmark items,
  model runs, and result contracts.
- [Running experiments](docs/running-experiments.md) — environment setup,
  preparation, execution, scoring, and comparison.
- [AWS / Amazon Bedrock](docs/aws.md) — self-service AWS credentials and
  Bedrock setup.
