# TROT: Triage of Observation Tokens for Efficient LLM Agents

📄 **Paper:** [OpenReview](https://openreview.net/forum?id=OgkJ87Gu35&noteId=OgkJ87Gu35)

## Abstract

LLM coding agents accumulate large observation histories, including conversations, reasoning traces, tool calls, and tool outputs, causing context windows to grow and inference costs to increase. We present TROT (Triage of Observation Tokens), a hierarchical framework that selectively compresses and removes stale observations using deterministic rules, a lightweight learned gate, and an auxiliary LLM only when needed. On SWE-bench Verified across two model families, TROT achieves the highest task success while substantially reducing context size. It attains a 50.6% resolve rate with 55% token savings on Sonnet-4.6 and a 46.1% resolve rate with 46% token savings on Devstral-24B, while requiring 6× fewer auxiliary LLM calls than uniform LLM-based reflection.

![TROT pipeline](figures/pipeline.png)

*Figure: TROT pipeline. Each observation passes through three layers in cost order: a deterministic compressor (zero cost), a learned gate (near-zero cost), and an auxiliary LLM used selectively. The agent sees the result, and the history compressor feeds back into the next step.*

## Method

TROT handles observations in three layers, cheapest first:

1. **Deterministic rules (zero cost).** Veto compression for critical content such as error traces and first-time outputs, and apply pattern-based compression to known waste: stale file re-reads, submit boilerplate, and edit confirmations.
2. **Learned gate (near-zero cost).** Predicts downstream utility from trajectory-state features and chooses SKIP, SUMMARY, or FULL for each observation.
3. **Auxiliary LLM (selective).** Summarizes only the observations the gate marks for summary.

## Baselines and Main Results

We compare against four baselines and the unmodified agent on SWE-bench Verified (89 held-out tasks), using the same agent loop, prompt, tools, and harness for every method.

| Baseline | Description |
|----------|-------------|
| Baseline | Unmodified agent, full observations |
| TACO (Ren et al. 2026) | Self-evolved compression rules with critical-observation bypass |
| AgentDiet (Xiao et al. 2025) | Auxiliary Haiku reflection on observations above a token threshold |
| Truncation | Keeps the start and the error tail of each observation |
| Sliding Window (Lindenbauer et al. 2025) | Masks observations older than M steps with a placeholder |

Resolve rate is the percentage of tasks with a patch that passes the repository's tests. Token savings are relative to the baseline.

| Method | Sonnet-4.6 Resolve % | Tokens | Savings | Devstral-24B Resolve % | Tokens | Savings |
|--------|---------------------:|-------:|--------:|-----------------------:|-------:|--------:|
| Baseline | 45.9 | 647K | | 44.9 | 652K | |
| TACO | 47.1 | 713K | −10% | 43.8 | 592K | +9% |
| AgentDiet | 42.4 | 455K | +30% | 44.9 | 570K | +13% |
| Truncation | 38.8 | 527K | +19% | 41.6 | 580K | +11% |
| Sliding Window | 37.6 | 478K | +26% | 38.2 | 403K | +38% |
| **TROT (ours)** | **50.6** | **289K** | **+55%** | **46.1** | **349K** | **+46%** |

TROT is the only method that achieves both the highest resolve rate and the largest token savings on both models.

## Citation

```bibtex
@inproceedings{trot2026,
  title={TROT: Triage of Observation Tokens for Efficient LLM Agents},
  author={Paul Trust},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
