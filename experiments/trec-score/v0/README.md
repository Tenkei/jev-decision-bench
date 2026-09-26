# TREC Score v0: Deep Learning passage relevance

This experiment asks how well one candidate passage answers one search query.
It uses NIST's four-level 2019 TREC Deep Learning Passage Ranking judgments:
not relevant, related but not an answer, highly relevant, and perfect.

The compiler downloads the official 2019 test-query file, the corresponding
top-1000 candidate bundle, and the NIST qrels. It selects 400 judged pairs per
grade by SHA-256 ordering of `query_id:passage_id`, yielding 1,600 records.
NIST's complete qrels include pooled passages outside the published reranking
bundle; only judged pairs whose passage text is present in that bundle are eligible.
The state contains only the query and candidate passage. Retrieval rank, source
system score, and other candidates are deliberately excluded.

The TREC judgment definitions make this a genuine ordered Score task: each
level describes a distinct relevance relationship, and fractional scores express
the model's distribution across those relationships. MS MARCO source data does
not declare a dataset reuse license in this package; the source is recorded as
`NOASSERTION`. Review its terms before redistributing source or derived text.
