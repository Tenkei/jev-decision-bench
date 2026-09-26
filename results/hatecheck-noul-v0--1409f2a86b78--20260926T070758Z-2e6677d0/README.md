# HateCheck Noul comparison

This comparison evaluates 3,728 binary moderation decisions from package
`1409f2a86b78`. JEV is the baseline. The complete machine-readable evidence
is in [comparison.json](comparison.json). See the
[HateCheck Noul experiment](../../experiments/hatecheck-noul/v0/README.md)
for the dataset, decision contract, and evaluation details.

## Decision quality and confidence

Higher is better for accuracy, macro-F1, and AUROC. Lower is better for Brier
and ECE.

| Model | Accuracy | Macro-F1 | AUROC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| JEV | 98.2% | 97.9% | 0.9976 | 0.0222 | 0.0666 |
| DeepSeek V3.2 | 97.2% | 96.7% | 0.9855 | 0.0281 | 0.0602 |
| Claude Haiku 4.5 | 95.1% | 94.2% | 0.9934 | 0.0412 | 0.0925 |
| Claude Sonnet 5 | 97.4% | 97.0% | 0.9956 | 0.0230 | 0.0502 |
| GPT-5.6 Luna | 100.0% | 99.9% | 1.0000 | 0.0018 | 0.0154 |
| GPT-5.6 Sol | 99.8% | 99.7% | 1.0000 | 0.0023 | 0.0153 |

## Response reliability and performance

Valid rate is the share of records satisfying the Noul output contract.
Latency is milliseconds. Tokens are reported total tokens per attempted
decision.

| Model | Valid rate | p50 latency | p95 latency | Tokens / decision |
| --- | ---: | ---: | ---: | ---: |
| JEV | 100.0% | 234 | 285 | 382 |
| DeepSeek V3.2 | 100.0% | 1,316 | 2,961 | 195 |
| Claude Haiku 4.5 | 96.2% | 716 | 915 | 367 |
| Claude Sonnet 5 | 100.0% | 1,735 | 3,207 | 285 |
| GPT-5.6 Luna | 73.2% | 900 | 1,250 | 196 |
| GPT-5.6 Sol | 100.0% | 860 | 1,745 | 208 |

## Interpretation

The GPT-5.6 routes have the strongest decision and probability quality. Luna's
quality is highest on its valid responses, but its 73.2% valid-output rate is
an important qualification. Sol combines near-perfect quality with complete
contract compliance.
