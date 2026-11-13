# Completed frozen event-seven replay: interpretation and publication review

Status: analyzed against retrieved, hash-verified scalar tables. This reviewer authored the primary analysis/report code; this is an authorship-overlapping interpretation review, not the independent scalar audit or an independent GPU replication. Execution commit: `10d89bf8ba657cd905d7ad4fb31bf65a0a851656`.

## Main finding

**The fitted retention advantage is unresolved.** Native accepts through 740/815 equal-weight slot equivalents (90.797546%); RECENT accepts through 745/815 (91.411043%). RECENT rescues 13 and loses 8, leaving five additional retained slot equivalents: **+0.613497 percentage points**, request-cluster 95% interval **[−0.494453, +1.668574] pp**.

Native still misses 75 slot equivalents: **9.202454 pp of headroom**, comprising 19 earlier breaks and 56 first mismatches at the protected tag. RECENT has 14 earlier breaks and 56 exact-tag mismatches. Native retention therefore does not exhaust the available headroom. The five-slot net point gain is much smaller than those 75 available misses, but the uncertainty interval—not a new threshold—determines the conclusion.

This result does **not** demonstrate that the mechanism fits training repairs successfully but then fails specifically at transfer. Its training-input advantage is itself unresolved. The earlier PR44 serving/held-out result stands unchanged: 8 held-out rescues versus 11 losses; acceptance through 275/431 versus native 278/431; no demonstrated serving advantage.

## Where the paired changes occur

The following partition is computed from paired scalar rows without importing the primary reduction. Slot equivalents first average each slot's one or two eligible presentations.

| Changed endpoint outcome | Raw occurrences | Slot equivalents |
|---|---:|---:|
| Rescue: native's first mismatch was exactly at the tag | 13 | 7 |
| Rescue: native broke earlier, but its tag was already directly correct | 8 | 4 |
| Rescue: native broke earlier and its tag was also wrong | 4 | 2 |
| Loss: RECENT's first mismatch is exactly at the tag | 12 | 6 |
| Loss: RECENT breaks earlier, but its tag remains directly correct | 2 | 1 |
| Loss: RECENT breaks earlier and its tag is also wrong | 2 | 1 |
| Total rescues | 25 | 13 |
| Total losses | 16 | 8 |

An accepted-through rescue after an earlier native break is not automatically a repair of the tagged token: in four rescued slot equivalents, that token was already directly correct under native. Conversely, a tag can remain directly correct while an earlier error makes it unusable. Exact-tag rescue rows have the old preceding matches intact under both endpoints and now match the tag under RECENT. These are paired final-endpoint descriptions, not evidence identifying which update caused or repaired the transition.

Across all original slots, direct tag decisions show **9 gains and 8 losses**, a net one slot equivalent (**+0.122699 pp**), compared with five net accepted-through gains. Mean margin rises by **0.012807**; the below-floor fraction drops from 10.552147% to 9.815951%. These descriptive margin changes alone do not establish useful retention or explain serving behavior.

## Denominators and singleton

- All 128 fitted requests, 815 slots and 1,629 occurrences are retained.
- Pass 0 contains 814 occurrences; pass 1 contains 815.
- The singleton is `train_00786`, canonical start 91, proposal slot 13, presentation 166 (pass 1). Native first fails exactly at the tag; RECENT accepts through it. It contributes one full rescued slot equivalent, not one half.
- Consequently, raw occurrence counts (25 rescues, 16 losses) yield +0.552486 pp over 1,629, while the prespecified equal-slot estimate is +0.613497 pp over 815. These are distinct valid descriptive reductions; the equal-slot result is primary.
- The 814 twice-eligible slots have the same recorded correct-prefix lengths and direct-correctness outcomes in both eligible presentations for each endpoint. The two passes are not independent request samples.
- There are 21 distinct slots with a retention disagreement, spanning 19 requests. Twenty requests have any three-category disagreement, and 16 have any direct-correctness disagreement.
- The primary interval jointly resamples all slots and both arms within 128 request clusters, 10,000 draws, PCG64 seed 1707, dividing by each draw's resampled slot count. It measures fitted-request composition sensitivity, not training-seed variation.

## Work and cost scope

Both arms completed 256 full-response forwards and 8,170 vocabulary-head chunks: **512 forwards and 16,340 chunks total**, with zero backwards, Adam steps or new generations.

| Measured interval | NATIVE7 seconds | RECENT7 seconds |
|---|---:|---:|
| CUDA-synchronized scoring forwards | 81.445874 | 80.252544 |
| Model loading and initial integrity checks | 26.619107 | 26.467217 |
| Final integrity check | 5.752227 | 5.706855 |
| Worker total | 127.471636 | 126.067765 |

The scoring intervals total 161.698418 s; worker totals sum to 253.539401 s. They are nested host wall-clock measurements that wait for GPU completion, not pure GPU-kernel busy time and not additive across rows. Peak allocation is 5,531,260,928 B and peak reserved memory 6,874,464,256 B for each worker. Final campaign wall/preparation and exact occupied-card definition should come from the supervisor receipts, separately.

## Publication checks

The generated scalar tables expose only request/slot/presentation identities, margins, outcomes and counts. They omit prompt/answer text, canonical and rival token IDs, tensors and private filesystem paths. Publication should include redacted endpoint/input identities, installed-tensor and sampler-restoration receipts, full work counts, the independently reduced scalar audit, and the single reproduction command. Retain the original generated report and raw records unchanged; publication-only explanations may clarify the above distinctions.

**Final interpretation:** native leaves 9.202454 pp of measured headroom. RECENT rescues 13 slot equivalents and loses 8, but its +0.613497 pp advantage remains unresolved. This cannot yet isolate a fit–transfer failure, and does not justify a coefficient change, more training, or a new algorithmic claim. Stop at this report.
