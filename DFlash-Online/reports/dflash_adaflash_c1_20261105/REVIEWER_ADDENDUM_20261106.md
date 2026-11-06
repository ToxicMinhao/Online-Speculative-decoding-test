# Reviewer clarification: AdaFlash clipping and the A58–K+ comparison

## Scope and provenance

This addendum supplements the [completed C1 report](REPORT.md) and the [bounded clipping diagnostic](../dflash_adaflash_clipping_20261106/REPORT.md). Their original measurements and report bytes remain unchanged. This is a documentation update, with no new training, generation or checkpoint modification.

- A58 campaign execution: `681336c5e5c5caa0aa67560e46ffe83ce81e9351`; original result publication: `74059492c9a7751b3845b75798f4033e285bc994`.
- Clipping diagnostic execution: `64cb56509d1c4ed3bd2ceb8365a346603edb2957`; result publication: `680413210f88cb6e126056b3bf3f23fcc25b0407`.
- Official loss source examined: AdaFlash `478bf0ff7a3a5232b13d0f14e752d5fe1da0b718`, [rkl_model.py](https://github.com/ZinYY/AdaFlash/blob/478bf0ff7a3a5232b13d0f14e752d5fe1da0b718/pipeline/training/rkl_model.py#L168-L207).
- Review status: analysis of retained execution evidence; no independent GPU replication. Implementation/audit authorship overlap is disclosed in the original reports.

## What is the same, and what differs?

Both diagnostic branches use **0.8 teacher-argmax cross-entropy (CE) + 0.2 reverse Kullback–Leibler divergence (RKL)**, at temperature 1. Teacher targets, response-only masks, positional weights, response averaging, saved responses/features, anchor bindings, initial weights, fresh AdamW state, learning rate and random state are matched. Both use the same underlying BF16 probability and optimizer path. Neither introduces a K/K+ selection mask or an adaptive-length head.

**The only intended algorithmic change between these two branches is the clipping definition.** The completed A58 campaign used the paper-defined vocabulary-entry rule specified by its handoff. The diagnostic also evaluates the repository's normalized whole-position rule within our framework.

Let `e_iv = q_iv (log q_iv − log p_iv)`, where `q` is the draft distribution and `p` the frozen target distribution. Let `w_i` be the existing validity/position weight and `D = sum_i w_i + 1e-6`, over the entire response's sampled blocks.

| Rule | Reverse-KL contribution |
|---|---|
| A58 / paper-defined | `sum_i w_i sum_v min(e_iv, 0.01) / D` |
| Repository rule | `sum_i w_i min((sum_v e_iv) / D, 0.01)` |

The first caps each vocabulary entry **before summation**. The second sums the vocabulary, divides by the full response denominator, then caps the **whole position**. Although both use 0.01, they cap different quantities. The repository rule is equivalent in exact arithmetic to capping raw position RKL at `0.01 × D` before the weighted response reduction. A strictly capped position loses its direct RKL gradient, while CE still contributes. The paper rule retains uncapped entry contributions, including negative entries.

The numerical adaptation remains explicit: repository-rule normalization, cap decisions and rescaling use FP32, then return to the shared BF16 mixture. Thus this is a controlled comparison of clipping definitions under common arithmetic, **not bitwise reproduction of the authors' kernels or entire training pipeline**. Clipping is part of the loss; changing its definition changes the objective even though the CE/RKL ingredients and nominal mixture remain the same.

## What the quick test established

The first six retained event-1 presentations formed three fixed pairs. Each rule took one disposable step per pair from identical released weights and fresh optimizer/RNG state. Total work was **six AdamW steps**, twelve training presentations and twelve post-step source-fit forwards, plus separately counted preparation/startup. There was no full event or complete-response generation. Execution took **4 min 42 s**.

| Pair | BF16 update-vector difference / paper update norm | Paper-rule prefix gain | Repository-rule prefix gain |
|---|---:|---:|---:|
| 1 | 79.48% | +2,298 | +2,452 |
| 2 | 75.29% | +1,581 | +1,475 |
| 3 | 73.56% | +1,200 | +1,005 |
| Total prefix gain | — | **+5,079** | **+4,932** |

Update magnitudes were nearly equal, but their directions differed substantially: cosine similarity was 0.685–0.730. The resulting BF16 parameter values differed at **10.74–12.18% of trainable coordinates**. That coordinate fraction is distinct from the 74–79% relative vector distance above.

Local usefulness was mixed. The repository rule lost fewer old-prefix occurrences overall (**455 versus 589**) but repaired fewer old first errors (**1,071 versus 1,131**). Prefix gains count occurrences across overlapping sampled training blocks, not independent tokens, held-out requests or native verification calls. The repository rule helped more in one pair and less in two; these three pairs do not establish a generally superior loss.

All six state restorations and paired input/prediction/RNG checks passed. The first paper-rule pair reproduced the common logged historical A58 first-update metrics. There were 56 local CPU test passes, 29 focused A800 test passes and 247 retained-evidence audit checks. These support the diagnostic's implementation and accounting, not a full-horizon equivalence claim. [Detailed evidence and limits](../dflash_adaflash_clipping_20261106/REPORT.md).

**Retention clarification:** the original C1 report's statement that retired feature reconstruction was unavailable is superseded narrowly by this diagnostic. Six response-feature sets were reconstructed from exact retained native trace inputs, with posterior/commit validation. The original feature tensors remain retired; their byte-for-byte equality cannot be checked. This does not restore intermediate event 1–57 optimizer states or general exact continuation capability.

## What the existing A58 versus K+ result supports

The completed 1,024-request endpoint comparison remains valid for the checkpoints actually evaluated:

| Endpoint measure | Paper-defined A58 | K+58 |
|---|---:|---:|
| Verification calls | 45,083 | 43,267 |
| Sum of per-request three-repeat mean latency | 816.640 s | 783.742 s |

K+58 used **4.02813% fewer verification calls** and **4.02845% less complete-request time**, with a **95% paired-request bootstrap interval of 3.32888%–4.73465%** for the time reduction. Exact output tokens and termination matched across methods. The interval describes variation across this reused request panel; it does not establish training-seed robustness. Training costs are separate.

The supported statement is:

> Under our fixed concurrency-one DFlash-Online protocol, K+58 outperformed the paper-defined AdaFlash OPD-loss A58 checkpoint by about 4.03% in both verification calls and complete-request latency. A bounded paired diagnostic found that the official repository's clipping definition produces substantially different updates, with mixed same-response fitting results. The full-training outcome under that alternative clipping rule remains unmeasured.

We **cannot** claim that K+ is 4.03% faster than the official released AdaFlash implementation or full system. We also cannot claim that clipping caused A58's deficit, that repository clipping would remove it, or that K+'s advantage would necessarily remain unchanged. The quick test establishes update sensitivity, not 58-event performance, held-out transfer or adaptation-inclusive speedup.

## Decision status

The user requested this clarification before reviewer discussion and explicitly paused further training. No new event, baseline replacement, tuning or additional run is authorized or launched. The reviewer/user will decide whether any further comparison is needed.
