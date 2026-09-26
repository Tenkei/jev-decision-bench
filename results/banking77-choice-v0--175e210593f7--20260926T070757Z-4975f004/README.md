# BANKING77 Choice comparison

This comparison evaluates 3,080 intent-routing decisions from package
`175e210593f7`. JEV is the baseline. The complete machine-readable evidence
is in [comparison.json](comparison.json). See the
[BANKING77 Choice experiment](../../experiments/banking77-choice/v0/README.md)
for the dataset, decision contract, and evaluation details.

## Decision quality

Higher is better for accuracy and macro-F1. Lower is better for Brier and ECE.

| Model | Accuracy | Macro-F1 | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| JEV | 81.0% | 80.4% | 0.131 | 0.104 |
| DeepSeek V3.2 | 79.5% | 78.8% | 0.149 | 0.098 |
| Claude Haiku 4.5 | 77.9% | 77.1% | 0.151 | 0.101 |
| Claude Sonnet 5 | 80.5% | 79.7% | 0.119 | 0.027 |
| GPT-5.6 Luna | 85.6% | 84.7% | 0.107 | 0.091 |
| GPT-5.6 Sol | 86.1% | 85.5% | 0.101 | 0.072 |

All runs produced valid answers for every record.

## Performance

Latency is milliseconds. Tokens are reported total tokens per attempted
decision; they are provider-reported and may not be directly comparable across
providers.

| Model | p50 latency | p95 latency | Tokens / decision |
| --- | ---: | ---: | ---: |
| JEV | 512 | 575 | 2,867 |
| DeepSeek V3.2 | 1,494 | 3,050 | 1,297 |
| Claude Haiku 4.5 | 903 | 1,138 | 2,326 |
| Claude Sonnet 5 | 1,739 | 3,431 | 2,038 |
| GPT-5.6 Luna | 894 | 1,931 | 1,608 |
| GPT-5.6 Sol | 863 | 2,127 | 1,148 |

## Interpretation

GPT-5.6 Sol has the strongest decision-quality results in this comparison,
with Luna close behind. Sonnet has notably low calibration error despite lower
accuracy than the two GPT-5.6 routes.
