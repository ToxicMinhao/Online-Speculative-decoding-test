# Native prefix-loss result audit — execution 002

**The prespecified local E-versus-N screen passes.** Independent reduction agrees with the report: **+0.015625 accepted proposals per fixed block**, conditional paired 95% interval **[+0.00244140625, +0.02880859375]**. This is 64 additional prefix positions over 4,096 paired block/windows; it is not a 1.5625% speedup or a complete-response serving result.

The audit passes **32,821 checks** with no failures. A separate streaming audit reconciles all **991,193 native ledger lines**. Actual execution source is `2549f02d9b1edba8fb86f8e91a497e68c224d586`; original failed source `fdccdf5482f2125f212476547522068f953c564a` remains preserved.

## What changed

| Population / state | Acquired prefix positions | Lost old-prefix positions | Net |
|---|---:|---:|---:|
| Independent N | 177 | 136 | +41 |
| Independent E | 194 | 89 | +105 |
| Independent C | 163 | 114 | +49 |
| Independent H | 154 | 101 | +53 |
| Fitting N | 2,948 | 139 | +2,809 |
| Fitting E | 2,119 | 153 | +1,966 |
| Fitting C | 1,051 | 162 | +889 |
| Fitting H | 383 | 359 | +24 |

E's independent advantage comprises **17 more acquired positions and 47 fewer lost positions**. It has 57 old-blocker repairs versus N's 52, and 40 earlier breaks versus 43. It also displaces eight native blocker repairs. Repair counts use blocks; the acquisition/loss table uses prefix-position occurrences.

The independent population has 4,096 paired block/windows and 3,577 pre-update blocker opportunities. Among 64 equally weighted request means, E improves 22, regresses 12 and ties 30. Its window differences are positive in 11, negative in three and tied in two; both pass means are positive (+0.0078125 and +0.0234375). These are descriptive views of the same data, not independent confirmation.

E fits the original responses less well: **843 fewer net fitting-prefix positions than N**, consisting of 829 fewer acquired and 14 more lost. Thus, the independent result does not follow from better fitting on the sampled responses. Source evidence comprises 8,454 overlapping fitting anchors, all kept and eligible, with zero padded/zero-supervision anchors; 123,450 valid proposal occurrences and 6,356 old-blocker opportunities. Five evaluated states do not multiply those exposure denominators.

## What remains unresolved

E−H is +0.0126953 with interval [−0.0053711,+0.0288086], and C−H is −0.0009766 with interval [−0.0158691,+0.0144043]. Both are unresolved, not equivalence. H is an actual zero-current-gradient Adam update with history and weight decay; it is not static P.

E's BF16-realized displacement norm is 89.57%–97.00% of N's across the fixed windows (median 94.53%). The screen therefore does not distinguish a role-specific directional benefit from changed update size, clipping or interaction with Adam history. C remains a descriptive control. Its numerical advantage over N is not established by its interval and does not justify switching the primary arm.

This result earns consideration of the handoff's separately designed scale control and short actual continuation. It does not establish a deployable upgrade, fresh-input robustness, long-horizon behavior, full-response acceptance or latency gains. No additional run is performed or authorized by this audit.

## Validity and work

All six repaired qualification cases pass, exact origin restoration passes, and final native weights/optimizer/counters/RNG agree with the retained NATIVE7 endpoint. All 20,480 prescribed state outcomes completed with no invalid, unknown or unattempted rows. Only N advances the trajectory.

Current work: **182 Adam applications, 492 response forwards and 332 response backwards**. This includes six qualification applications, separately recorded startup and all 160 source-evaluation forwards. The H optimizer branch needs no update forward/backward, but receives the same source evaluation as other states: 32 response-level source forwards. Native work separately records 23,425 draft blocks, 23,484 target calls, 448,601 context forwards and 84 loads. No new complete-response generations occurred.

Current occupied-card time is **6,800.969879 s**. Adding the retained failed attempt's **203.070064 s and five Adam applications** gives **7,004.039942 s (116.734 minutes), 187 cumulative Adam applications**, within the unchanged 10,800-second cap and approved 187-application limit. Current campaign wall time is 6,872.811598 s; nested phase timers must not be summed. The prior failure review is unchanged.

## Interpretation review

All **11/11 ARS categories** are assessed in `RESULT_INTERPRETATION_002.json`: Simpson's paradox, ecological fallacy, Berkson's paradox, collider bias, base-rate neglect, regression to the mean, survivorship bias, look-elsewhere effects, forking paths, causal attribution and reverse causality. No contradiction with the prespecified local-screen claim was found. Reused development data, multiple historical explorations, conditional uncertainty, one native trajectory and update-size confounding remain explicit limitations.

This is a same-team review by the production reducer's author using independent arithmetic and receipt reconstruction. It is not an independent model replication. Raw proposal/token reconstruction and tensor equality depend on the retained native correctness receipts unless separately audited by another reviewer.
