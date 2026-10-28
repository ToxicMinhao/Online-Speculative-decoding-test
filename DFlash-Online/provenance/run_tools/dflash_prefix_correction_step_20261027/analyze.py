"""Summarize retained local results; no model execution or mechanism changes."""
import csv
import json
from pathlib import Path

root=Path(__file__).parent
raw=root/'raw'
r=json.loads((raw/'execution_001/RESULT.json').read_text())
assert r['status']=='completed'
k,c=r['post_k'],r['post_c']
summary={key:r[key] for key in ('status','comparison','correction','budget','ordinary_K_wall_s',
    'ordinary_K_peak_cuda','correction_peak_cuda','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','wall_s')}
summary.update(costs=r['core']['costs'],startup=r['startup'],
    K_violation={key:k[key] for key in ('V','numerator','denominator','totals')},
    C_violation={key:c[key] for key in ('V','numerator','denominator','totals')},
    repair_retention=c['repair_totals'],core_work=r['core']['work'],native_forward_calls=r['core']['forward_calls'])
(root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
rows=[]
for method,state in [('K',k),('K_plus_correction',c)]:
 for row in state['responses']:
  rows.append({'method':method,**{key:value for key,value in row.items() if isinstance(value,(str,int,float))}})
with (root/'PER_RESPONSE.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
co=r['correction'];cost=r['core']['costs'];kt=k['totals'];ct=c['totals']
report=f'''# Single-step K-plus-correction diagnostic

**The diagnostic meets its prespecified local advancement criterion.** The original ordinary K step reproduced every retained per-response prefix count exactly. One post-K correction reduced lost old-prefix tokens from **{kt['old_prefix_tokens_lost']} to {ct['old_prefix_tokens_lost']}** and increased net matching-prefix gain from **+{kt['net_matching_prefix_change']:,} to +{ct['net_matching_prefix_change']:,}**. All **{c['repair_totals']['useful_repairs_retained']}** useful K repairs survived; **{c['repair_totals']['useful_repairs_newly_obtained']}** additional original first errors were repaired.

This is a local, training-conditioned comparison of the same two responses and anchors. It does **not** establish better held-out drafting or serving speed. The result supports considering the subsequent one-event test stated in the handoff; no such test, tuning or extra correction was run.

## Local comparison

| Metric | Ordinary K | K + one correction | Denominator / meaning |
|---|---:|---:|---|
| Newly earlier errors | {kt['new_earlier_errors']} | {ct['new_earlier_errors']} | 774 sampled blocks; new prefix shorter than its pre-update prefix |
| Lost old-prefix tokens | {kt['old_prefix_tokens_lost']} | {ct['old_prefix_tokens_lost']} | 5,236 pre-update matching-prefix occurrences |
| Useful original first-error repairs | {kt['first_errors_repaired_with_prefix_intact']} | {ct['first_errors_repaired_with_prefix_intact']} | 632 original first-error blocks; all earlier correct positions must remain correct |
| Total matching-prefix tokens | {kt['new_matching_prefix_tokens']} | {ct['new_matching_prefix_tokens']} | Sum over the same 774 blocks |
| Net matching-prefix gain versus pre-update | +{kt['net_matching_prefix_change']} | +{ct['net_matching_prefix_change']} | Pre-update total 5,236 |
| Strict negative margins on protected positions | {kt['strict_margin_violations']} | {ct['strict_margin_violations']} | 5,236 fixed protected occurrences |
| Wrong-at-tie protected positions | {kt['wrong_at_tie_positions']} | {ct['wrong_at_tie_positions']} | Native BF16 argmax is wrong at zero margin |
| All protected argmax errors | {kt['protected_argmax_errors']} | {ct['protected_argmax_errors']} | Negative margins plus wrong ties |
| Global hinge V | {k['V']:.9f} | {c['V']:.9f} | Weighted numerators {k['numerator']:g} / {c['numerator']:g}; common denominator {k['denominator']:g} |

Repair identity accounting: **527 retained, 0 lost, 6 newly obtained**. The correction removes 40 lost old-prefix occurrences and adds 67 net prefix occurrences beyond K. These are overlapping sampled-block occurrences, not independent tokens or requests. Primary improvement is established by the prefix counts, not by the lower hinge alone. Three wrong ties remain after correction; this surrogate gives zero direct derivative at zero margin and does not guarantee complete preservation.

Relative to post-K, **17 blocks improve, 8 regress and 749 tie** in matching-prefix length. They gain 77 and lose 10 occurrences, giving the net +67. Six K-damaged old prefixes are fully restored, and the correction introduces no damage to previously intact original prefixes. The 8 regressions concern post-K prefix progress; retaining every original first-error repair does not mean that every block keeps its entire post-K prefix.

### Per response

| Response | Blocks | Useful repairs K → C | Earlier errors K → C | Lost prefix K → C | Net prefix K → C |
|---|---:|---:|---:|---:|---:|
'''
for a,b in zip(k['responses'],c['responses']):
 report+=f"| {a['record_id']} | {a['blocks']} | {a['first_errors_repaired_with_prefix_intact']} → {b['first_errors_repaired_with_prefix_intact']} | {a['new_earlier_errors']} → {b['new_earlier_errors']} | {a['old_prefix_tokens_lost']} → {b['old_prefix_tokens_lost']} | +{a['net_matching_prefix_change']} → +{b['net_matching_prefix_change']} |\n"
report+=f'''
## What ran and what stayed unchanged

The same pinned Qwen3-8B target and Qwen3-8B-DFlash-b16 draft, qualified BF16 K/AdamW recipe, Flex attention, constant LR 3e-4, weight decay 0.01, clipping 1, response accumulation 2, original seed-17 RNG and startup were reused. Released weights, empty Adam, original modes/counters and RNG passed the exact initial-state check. The original first presentations are `train_00072` and `train_00121`; all anchors and conditioning are retained. K's original response normalization and objective remain unchanged.

Original correct-prefix membership and triangular weights were captured from K's existing pre-update forward. The correction uses one global denominator across both responses. Its gradients are accumulated separately in FP32; the global squared norm, coefficient and correction vector use FP32. Parameters are corrected from actual post-K BF16 weights and rounded once back to BF16. No FP32 master parameters, extra Adam step, epsilon, clipping, line search or resampling was introduced.

Global gradient squared norm: **{co['gradient_norm_squared_fp32']:.12f}**; coefficient `-V / ||g||²`: **{co['coefficient_fp32']:.12f}**. K's update norm is **{co['K_update_norm']:.9f}**. Proposed correction norm is **{co['proposed_correction_norm']:.9f}** ({100*co['proposed_correction_to_K_norm_ratio']:.4f}% of K); BF16-realized correction norm is **{co['realized_correction_norm']:.9f}** ({100*co['realized_correction_to_K_norm_ratio']:.4f}% of K). Norm reporting uses FP64; decision arithmetic remains FP32. The correction follows the full K step rather than shrinking or replacing it. Exact Adam moments, counters, RNG, modes, caches and existing gradients remain unchanged around correction/diagnostic replay.

The complete ordinary-K first-step reproduction matches 527 useful repairs, 47 earlier errors, 153 lost tokens and +1,740 net prefix tokens, including all eight metric checks separately for each response. K's update norm also agrees with the retained reference. No discrepancy, nonfinite failure or retry occurred in this GPU run.

## Work and costs

| Work | Actual |
|---|---:|
| Core training presentations / Adam steps | 2 / 1 |
| Direct corrections | 1 |
| Core response forwards / backward equivalents | 6 / 4 |
| Separate startup response forwards / backwards / Adam | 2 / 2 / 0 |
| Complete-response generations / exports / automatic retries | 0 / 0 / 0 |

All 774 sampled blocks were reused. There are 98 initial projection chunks across the two response forwards. K evaluates its chunk objective 196 times including recomputation; its 392 head calls include current and frozen-teacher projections. Post-K uses 196 objective/head calls including recomputation; post-correction uses 98 without backward. Separate startup uses 392 head calls including both projections and recomputation. Draft/layer counts and attempted/completed work remain in [SUMMARY.json](SUMMARY.json) and the raw work log. No new target-model forward or feedback generation occurred.

| Cost | Seconds | Scope |
|---|---:|---|
| Separate disposable startup | {r['startup']['wall_s']:.3f} | Includes loading; zero Adam, exact restoration |
| Fresh K loading and initial-state verification | {r['loading_and_initialization_s']:.3f} | Before the measured step |
| Ordinary K step with prefix recording | {r['ordinary_K_wall_s']:.3f} | Two training presentations and one Adam step |
| Post-K violation forwards and gradients | {cost['post_k_s']:.3f} | New correction work |
| Direct parameter correction | {cost['correction_s']:.3f} | No optimizer update |
| Post-correction scoring | {cost['post_c_s']:.3f} | Diagnostic-only recheck |
| Diagnostic process wall time | {r['wall_s']:.3f} | Includes state copies/checks and reporting |
| Supervisor elapsed | 163.000 | Includes preflight, remote CPU tests and postflight |

Post-K violation work plus applying the correction cost **{cost['post_k_s']+cost['correction_s']:.3f}s** here. The {cost['post_c_s']:.3f}s recheck is separate diagnostic work. These phase timers do not cover every CPU snapshot/check overhead; they do not establish an optimized online update cost or any serving speedup. The {cost['prefix_chunk_capture_s']:.3f}s capture timer covers only per-chunk observation, not all input copying and snapshots, and is nested inside K's timer. Do not sum nested timers again.

Peak GPU allocation/reservation: ordinary K phase **{r['ordinary_K_peak_cuda']['allocated_bytes']/1e9:.3f}/{r['ordinary_K_peak_cuda']['reserved_bytes']/1e9:.3f} GB**; correction/recheck phase **{r['correction_peak_cuda']['allocated_bytes']/1e9:.3f}/{r['correction_peak_cuda']['reserved_bytes']/1e9:.3f} GB**. The latter includes the resident K model. Peak allocated memory rose by **{(r['correction_peak_cuda']['allocated_bytes']-r['ordinary_K_peak_cuda']['allocated_bytes'])/1e9:.3f} GB**. No separate fresh uninstrumented timing baseline was run.

## Correctness and retained evidence

Five additive source/test files were installed in A800's `/data/sources/minhao/DFlash-Online`. Historical objectives and author training/inference files were unchanged. 48 local CPU checks and 21 remote new-module checks passed; these suites overlap. The initial CPU recorder failure was fixed by making observation explicitly no-grad; it was an instrumentation error, not a change to K or the correction formula. Its failure record remains [preserved](CPU_FOCUSED_INITIAL.log). The existing qualified `donated_buffer=False` pin was retained before compilation.

The diagnostic completed on A800 at 10:59:44 UTC on October 27, exit 0. All 565 sealed source/input artifacts passed postflight, and all 53 retrieved evidence files passed size/hash verification. Both GPUs are idle. No prior artifacts were removed to admit this run. After verified retrieval, only the upload/evidence transfer containers were retired on both hosts; extracted evidence and source remain, with no replay capability lost. About 37.79GB remains on A800 and 82.42GB on Fenrir. [Storage record](STORAGE_MAINTENANCE.json).

- [Prespecified handoff](HANDOFF.md), [numerical contract](NUMERICS.json), [implementation review](REVIEW.md).
- [Complete result](raw/execution_001/RESULT.json), [exact reproduction gate](raw/execution_001/REPRODUCTION.json), [budget receipt](raw/execution_001/BUDGET.json).
- [Per-response table](PER_RESPONSE.csv), [per-block audit table](analysis/PER_BLOCK.csv), [independent audit](FINAL_AUDIT.md).
- [Post-correction optimizer/state equality](raw/execution_001/CORRECTION_STATE_UNCHANGED.json), [postflight](raw/POSTFLIGHT.json).

**Stop at this local result.** It demonstrates a useful correction on this one fitted batch under BF16. It does not establish transfer across requests, robustness across batches, a full-event gain or deployment acceleration.
'''
(root/'REPORT.md').write_text(report)
print(json.dumps(summary['comparison'],indent=2))
