# TREC Score comparison

This comparison evaluates 1,600 ordered search-relevance decisions from
package `91f38528cbd7`. JEV is the baseline. The complete machine-readable
evidence is in [comparison.json](comparison.json). See the
[TREC Score experiment](../../experiments/trec-score/v0/README.md)
for the dataset, decision contract, and evaluation details.

## Final-score quality and consistency

Accuracy maps the reported continuous score to its nearest rubric tier. MAE
measures the distance from the gold tier. Consistency is the share of usable
responses where the reported score equals the probability-weighted score.
Malformed responses lack a usable Score answer. Higher is better for accuracy
and consistency; lower is better for MAE and malformed rate.

| Model | Tier accuracy | Score MAE | Consistency | Malformed |
| --- | ---: | ---: | ---: | ---: |
| JEV | 47.4% | 0.671 | 100.0% | 0.75% |
| Claude Haiku 4.5 | 44.6% | 0.727 | 59.3% | 0.13% |
| Claude Sonnet 5 | 47.4% | 0.672 | 14.4% | 0.06% |
| DeepSeek V3.2 | 42.9% | 0.765 | 65.0% | 0.00% |
| GPT-5.6 Luna | 45.2% | 0.698 | 98.6% | 0.00% |
| GPT-5.6 Sol | 45.8% | 0.691 | 99.8% | 0.00% |

## Probability-distribution quality

Brier score, log loss, and ECE assess the full reported level distribution;
lower is better. Entropy describes distribution spread rather than a quality
ranking: a higher value means a less concentrated distribution.

| Model | Brier | Log loss | ECE | Entropy |
| --- | ---: | ---: | ---: | ---: |
| JEV | 0.771 | 4.608 | 0.153 | 0.519 |
| Claude Haiku 4.5 | 0.791 | 2.973 | 0.145 | 0.744 |
| Claude Sonnet 5 | 0.683 | 2.097 | 0.083 | 0.925 |
| DeepSeek V3.2 | 0.911 | 13.245 | 0.206 | 0.300 |
| GPT-5.6 Luna | 0.869 | 2.803 | 0.191 | 0.505 |
| GPT-5.6 Sol | 0.813 | 2.457 | 0.167 | 0.580 |

## Performance

Latency is milliseconds. Tokens are reported total tokens per attempted
decision.

| Model | p50 latency | p95 latency | Tokens / decision |
| --- | ---: | ---: | ---: |
| JEV | 239 | 290 | 484 |
| Claude Haiku 4.5 | 937 | 1,130 | 649 |
| Claude Sonnet 5 | 2,047 | 3,495 | 499 |
| DeepSeek V3.2 | 2,851 | 7,020 | 364 |
| GPT-5.6 Luna | 1,479 | 2,078 | 516 |
| GPT-5.6 Sol | 1,791 | 2,509 | 426 |

## Interpretation

Score quality should be read together with consistency. Sonnet's usable-score
quality and probability-distribution metrics are strong, but only 14.4% of its
responses make its score and distribution agree. Sol and Luna are nearly
fully consistent; JEV is fully consistent and fastest in this comparison.
