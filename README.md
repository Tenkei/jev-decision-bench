# jev-decision-bench

`jev-decision-bench` evaluates conventional language models on tasks designed around
[TypeSafe JEV](https://typesafe.ai/)'s **bounded-decision** interface. It is not a
general LLM leaderboard and it does not ask JEV to take benchmarks designed for
free-form text generation.

The central question is:

> Given the same state, decision rubric, and permitted answers, how accurately,
> reliably, quickly, and cheaply can JEV and conventional LLMs make a
> code-consumable decision?

## A Benchmark You Run Yourself

This project is not a generic online benchmark suite or a hosted leaderboard. It is an
easy-to-run, reproducible toolkit for evaluating your own model configurations
and versioned datasets against the same bounded-decision contracts used for
JEV, so you can make an evidence-based comparison with JEV in your environment.

See the [Architecture](docs/architecture.md) for how experiment packages,
runs, scores, and comparisons fit together.

## Project structure

```text
configs/       Tracked, editable model-route configurations; API keys stay in the environment.
experiments/   Versioned experiment specifications, rubrics, and source-data pins.
src/           Benchmark preparation, adapters, execution, scoring, and comparison code.
docs/          Methodology, experiment index, run guide, and provider setup guides.
artifacts/     Generated experiment packages, runs, scores, comparisons, and logs.
tests/         Contract and regression tests for the benchmark implementation.
```

## Experiment overview

The benchmark is organized by decision task family. `Dataset` identifies the
source where one has been selected; `TBD` means the family is part of the
benchmark design but no dataset has been adopted yet. See the
[experiment index](docs/index.md) for implementation and source details.

| Task family | JEV task type | Dataset | Description |
| --- | --- | --- | --- |
| Intent / queue routing | `Choice` / `Noul` | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | Route a request to the appropriate queue or determine whether it belongs in a candidate queue. |
| Semantic attribute extraction | Multiple `Noul` / `Choice` | TBD | Extract controlled attributes, such as topic, affected product, stated deadline, or request type. |
| Policy & guardrail decisions | `Noul` | [HateCheck](https://github.com/paul-rottger/hatecheck-data) | Decide whether content or an action satisfies an explicit policy or guardrail. |
| Tool / action approval | `Choice` / `Noul` | TBD | Choose or approve the appropriate next action from a bounded set; actions are never executed. |
| Severity / priority assessment | `Score` | TBD | Evaluate ordered judgments such as urgency, risk, or business impact. |
| Search relevance / ranking | `Noul` / `Score` | [TREC Deep Learning](https://trec.nist.gov/data/deep2019.html) | Determine query-document relevance and rank candidates. |
| Generic requirement verification | `Noul` / `Choice` | TBD | Determine whether an artifact or state satisfies a named, explicit requirement. |
| OOD uncertainty & abstention | All three | TBD | Measure whether confidence falls and review is selected when evidence or policy is missing. |

## What Is a Decision?

A decision is a bounded, code-consumable answer selected from an explicit
answer space—not a request to generate free-form text. Each benchmark item
contains the state, question, permitted answers, and gold label used for
evaluation. Its answer-space geometry determines the native JEV interface:

| Decision geometry | JEV interface | Example |
| --- | --- | --- |
| Two mutually exclusive outcomes | Noul | Is this query about a card that has not arrived? |
| Unordered, closed-set alternatives | Choice | Which banking intent best matches this query? |
| Ordered rubric levels | Score | How urgent is this request: low, medium, high, or critical? |

> [!NOTE]
> The benchmark treats all three as bounded decisions, but preserves their
> semantics. A `Score` is an ordered rubric position, not an arbitrary number;
> using a `Score` for categories with no meaningful order is invalid. A two-level
> `Score` is equivalent to a `Noul` and is not scored as a separate capability.

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

See the [Methodology](docs/methodology.md) for more information.

## Documentation

The repository follows a small documentation set: this README explains the
project, while the `docs/` pages describe the portfolio, method, execution,
and provider-specific setup.

- [Experiment index](docs/index.md) — registered experiments, task types,
  datasets, and status.
- [Methodology](docs/methodology.md) — experiment packages, benchmark items,
  model runs, and result contracts.
- [Architecture](docs/architecture.md) — code boundaries, immutable artifacts,
  task contracts, adapters, and re-evaluation.
- [Running experiments](docs/running-experiments.md) — environment setup,
  package preparation, and model execution.
- [Analyzing and sharing results](docs/analyzing-results.md) — offline scoring,
  comparison, reproducible exports, and publication artifacts.
- [AWS / Amazon Bedrock](docs/aws.md) — self-service AWS credentials and
  Bedrock setup.
