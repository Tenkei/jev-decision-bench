# BANKING77 Choice v0

`banking77-choice-v0` evaluates intent routing for banking-support queries. It
uses BANKING77's official 3,080-query test split and its 77 fixed intent
labels. It is a reproducible public classification workload, not a
contamination-resistant headline evaluation.

The compiler pins `test.csv` and `categories.json` to the source revision in
`spec.json`, verifies their SHA-256 hashes and expected row count, then keeps
the test-row index as the record's source reference. The rubric freezes every
intent key and its reviewed definition.

## Decision contract

Each record supplies a customer query, one routing question, and all 77
intent definitions:

```json
{
  "state": {"query": "My physical card has not arrived."},
  "question": "Which banking intent best matches this query?",
  "criteria": {
    "card_arrival": "A requested physical card has not arrived.",
    "…": "The remaining frozen BANKING77 intent definitions."
  }
}
```

The gold choice in this example is `card_arrival`. JEV receives its native
Choice request. Conventional LLMs return a schema-constrained rubric key and
top-label confidence; the runner rejects an unknown rubric key or a confidence
outside `[0, 1]`.

`default-evaluation.json` recommends accuracy, macro-F1, top-label Brier
score, expected calibration error, and a 77-label confusion matrix. The saved
run evidence can be re-evaluated under another compatible policy without
calling a model again.

BANKING77 is recorded as `CC-BY-4.0` in the pinned experiment specification.
