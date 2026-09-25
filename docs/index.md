# Experiment index

This is the portfolio and status register for benchmark experiments. Each row
identifies a separately versioned, model-independent decision workload.

| ID | Task type | JEV task type | Dataset | Status |
| --- | --- | --- | --- | --- |
| `banking77-choice-v0` | Intent / queue routing | `Choice` | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets), official test split (3,080 queries; 77 intents) | Ready for live runs — no model result recorded yet. |
| `banking77-noul-v0` | Intent / queue routing | `Noul` | Same frozen BANKING77 test split | Deferred — requires a frozen confusion map, compiler, adapter mode, and scorer. |

New experiments are added here only after their dataset, rubric, decision
records, and evaluator have been pinned.

## Next

Read [Running experiments](running-experiments.md) to prepare the selected
package, run each model, score the saved evidence, and compare compatible runs.
