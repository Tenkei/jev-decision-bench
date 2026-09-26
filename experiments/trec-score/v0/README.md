# TREC Score v0

This experiment asks how well one candidate passage answers one search query.
It uses NIST's four-level 2019 TREC Deep Learning Passage Ranking judgments:
not relevant, related but not an answer, highly relevant, and perfect.

## Dataset and preparation

The compiler downloads the official 2019 test-query file, the corresponding
top-1000 candidate bundle, and the NIST qrels. It selects 400 judged pairs per
grade by SHA-256 ordering of `query_id:passage_id`, yielding 1,600 records.
NIST's complete qrels include pooled passages outside the published reranking
bundle; only judged pairs whose passage text is present in that bundle are eligible.
The state contains only the query and candidate passage. Retrieval rank, source
system score, and other candidates are deliberately excluded.

## Decision contract

Each record supplies a query, one candidate passage, and the four ordered NIST
relevance definitions. The score is the continuous position across those
levels; probabilities record the model's distribution over them. The TREC
judgments make this a genuine Score task because every level expresses a
distinct relevance relationship.

```json
{
  "state": {
    "query": "what is the capital of france",
    "passage": "Paris is the capital and most populous city of France."
  },
  "question": "How well does this passage answer the search query?",
  "criteria": [
    "Not Relevant: the passage does not address the query or its information need.",
    "Related: the passage is on the same general topic but does not answer the query.",
    "Highly Relevant: the passage substantially answers the query, but is not a complete ideal answer.",
    "Perfect: the passage is a complete, direct answer to the query."
  ]
}
```

A valid normalized response carries both the continuous score and one
probability for every level:

```json
{
  "score": 2.99,
  "probabilities": {"0": 0.0, "1": 0.0, "2": 0.01, "3": 0.99}
}
```

## Evaluation

`default-evaluation.json` recommends exact-tier accuracy, ordinal mean
absolute error, top-label Brier score, and expected calibration error. A
material disagreement between a reported score and its level distribution is
retained as a partial answer and excluded from strict quality metrics.

## Dataset notes

MS MARCO source data does not declare a dataset reuse license in this package;
the source is recorded as `NOASSERTION`. Review its terms before redistributing
source or derived text.
