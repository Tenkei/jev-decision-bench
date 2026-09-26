# Experiment index

This is the portfolio and status register for benchmark experiments. Each row
identifies a separately versioned, model-independent decision workload.

| ID | Task type | JEV task type | Dataset | Status |
| --- | --- | --- | --- | --- |
| `banking77-choice-v0` | Intent / queue routing | `Choice` | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets), official test split (3,080 queries; 77 intents) | Ready for live runs — no model result recorded yet. |
| [`hatecheck-noul-v0`](../experiments/hatecheck-noul/v0/README.md) | Policy & guardrail decisions | `Noul` | [HateCheck](https://github.com/paul-rottger/hatecheck-data), full 3,728-case functional test suite | Ready for live runs — binary hateful-content decisions with functionality slices. |

New experiments are added here only after their dataset, rubric, decision
records, preflight fixture, and recommended evaluation policy are defined.

## Next

Read [Running experiments](running-experiments.md) to prepare the selected
package, run each model, score the saved evidence, and compare compatible runs.
