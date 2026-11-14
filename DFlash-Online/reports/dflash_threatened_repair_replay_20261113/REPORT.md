# K+ threatened-repair replay: selected repairs rescued, independent benefit not established

## Material Passport

- Status: **completed and audited; local advancement screen not passed**.
- Executed source: `d51718a7fa993a1494e593d15a8539090227d8b8`; sealed execution envelope: `ece7048e0ba024e27241d908605f8ef3622e21ba`.
- One native KPLUS6→7 trajectory, sixteen disposable comparisons, no new complete-response generations or serving-latency benchmark.
- The original failed qualification is preserved in [QUALIFICATION_REPORT.md](QUALIFICATION_REPORT.md). The approved retry includes its work and time.

**Targeted replay changed the selected decisions strongly, but did not demonstrate an advantage on independent prefixes.** It rescued 48 of 54 selected threatened tag/window cases, versus 17 rescued by position-matched uniform replay. On the independent probes, both primary point estimates were negative and both intervals included zero. This fixed recipe does not qualify for a longer rollout under the prespecified screen; neither equivalence nor universally harmful replay is established.

## Fixed experiment and validity

We restored the original complete KPLUS6 state at Adam step 768, including weights, persistent Adam moments, counters and RNG, and ran the original 128 two-response updates to step 896. The 128 saved responses, two-pass order, natural anchor draws, frozen teacher features, native K+ objective, BF16/Flex arithmetic, eight-block vocabulary chunks, LR `3e-4`, weight decay `.01` and clipping remained unchanged. Final native weights, optimizer tensors, counters and RNG exactly matched historical NATIVE7.

At windows 8,16,…,128, P was the complete pre-update state and N the actual native Adam endpoint. T replayed up to four currently accepted repair tags that N would reject exactly at the tagged position. U used the same number and positional weights, sampled deterministically from currently accepted tags without excluding threats. Both added `0.10 × selected-row weighted forward KL` after ordinary response-averaged gradients and before one native clip/Adam application. Both used the same padded union of original full-response layouts. Only N advanced; disposable branches restored N exactly. One identical U/T selection reused an exact branch result.

Memory contained 815 repaired tags across 128 responses and 129 immutable original layouts. Independent evaluation used four input-only selected starts from each of 64 reused development requests, disjoint from event 1–7 feedback. All sixteen windows were retained. Probe outcomes were collected only after training closed and could not influence selection. Full histories, native target conditioning, seed-plus-fifteen proposals, BF16 and termination boundaries were preserved.

All 129 focused CPU tests and the repaired six-case GPU qualification passed. The earlier adapter failure omitted the separately stored teacher tensor after batch normalization; the fix used that existing detached tensor with unchanged predecessor alignment. It changed no algorithm, arithmetic or tolerance. Disabled, zero-coefficient and padded-zero paths reproduced native behavior; active replay reached trainable parameters. All branch restorations, duplicate checks and native reconstruction checks passed. There were **16,384 valid conceptual probe rows**, from **16,128 physical probes plus 256 exact aliases**, with zero invalid or unattempted rows.

## Independent drafting results

Units are **accepted draft proposals per fixed block**, not verifier calls or speedup. The primary reduction averages four starts within request/window, all sixteen windows within request, then weights the 64 requests equally. Both intervals use the same 10,000 paired whole-request bootstrap draws, PCG64 seed 1707, with linear percentiles.

| Primary comparison | Difference | Paired 95% interval |
|---|---:|---:|
| T − N: targeted replay versus native K+ | **−0.012451** | **[−0.026611, +0.000494]** |
| T − U: targeted versus position-matched uniform replay | **−0.005371** | **[−0.017090, +0.005615]** |

Absolute prefix levels were P **5.375000**, N **5.385010**, U **5.377930**, T **5.372559**. Across request averages, T/N had 15 improvements, 24 regressions and 25 ties; T/U had 17, 19 and 28. Across fixed windows, T/N was positive in five and negative in eleven; T/U was positive in six, negative in seven and tied in three. All windows were active, so the secondary active-only mean equals the primary. These are conditional intervals for one native trajectory, one fixed uniform control and reused development requests—not training-seed uncertainty or simultaneous interval coverage.

The exact acquisition/loss decomposition shows why the observed point estimate was worse:

| Independent prefix occurrences, relative to P | N | U | T |
|---|---:|---:|---:|
| Newly acquired prefix positions | 177 | 127 | 143 |
| Lost old-prefix positions | 136 | 115 | 153 |
| Net prefix change | +41 | +12 | −10 |
| Native blocker repairs displaced by replay | — | 18 | 21 |

There were 4,096 paired block/windows. T acquired **34 fewer** prefix positions and lost **17 more** than N, producing `−34 − 17 = −51` net positions, or `−51/4096`. Against U, T acquired 16 more but lost 38 more, giving −22. Every integer decomposition passed. Blocker counts are distinct from position counts. This is a descriptive trade-off on these fixed probes; the primary intervals do not establish population-level harm.

## Opportunity and selected-memory effect

The complete funnel contains 13,040 tag/window rows: 202 current-response exclusions, 2,003 cases not accepted through the tag at P, and 10,835 eligible cases. Among eligible cases, N retained 10,725, created **84 exact-tag first rejections**, and caused 26 earlier breaks (25 with the tag still directly correct, one with it wrong). Earlier breaks were recorded but not treated.

The support gate passed: **50 distinct selected T tags, 41 memory requests, sixteen active windows, and fifteen windows with different U/T selections**. The full threat population contained 75 distinct tags from 52 requests. T selected 54 tag/window occurrences; U/T overlapped in two occurrences. These are separate occurrence and unique-identity denominators.

On the same 54 T-selected cases, P accepted all, N accepted none, U accepted 17 and T accepted **48 (88.9%)**. Relative to U, T rescued 32 and lost one. Of N's 54 rejections, 26 were wrong native argmax ties; T left no wrong ties on this set. Thus the minimum manipulation gate passed convincingly. It is a fitted-memory intervention check, not transfer evidence or a guarantee from lower KL.

All sixteen T branches changed BF16 weights relative to N. Added replay-gradient norms ranged from 0.243 to 3.380, versus native norms of 0.279–0.513; combined clipping activated in five T branches and no N branches. The intervention was therefore neither absent nor entirely rounded away. The fixed 0.10 loss coefficient does not imply 10% gradient influence, and these observations do not establish gradient strength or clipping as the cause of the independent outcome.

## Work, costs and retained evidence

| Work | Completed |
|---|---:|
| Native trajectory Adam steps | 128 |
| Disposable scientific U/T Adam steps | 31; one exact branch reused |
| Native duplicate-check Adam steps | 2 |
| Repaired qualification Adam steps | 6 |
| Current-attempt Adam total | 167 |
| Preserved failed-attempt Adam steps | 3 |
| **Cumulative Adam applications** | **170 / 171 allowed** |
| Current response forwards / backwards, including startup and checks | 4,878 / 545 |
| Scientific response forwards / backwards alone | 4,852 / 521 |
| New complete-response generations | 0 |

Scientific forward work comprised 256 native, 62 disposable native recomputations, 203 sparse replay, 4,128 P/N memory scans and 203 selected-union postchecks. The current native engine ledger additionally records 19,329 draft blocks, 19,388 target calls, 359,497 context forwards and 67 loads, including reconstruction and initialization. The failed attempt's 12 completed and one failed response forward, ten backwards, 119 draft blocks, 117 target calls, 178 context forwards and two loads remain separate and preserved. Current execution had zero failed or unknown operations.

Current campaign wall time was **119.41 minutes**. Current occupied-card time was 118.24 minutes; including the failed attempt's 185.987 seconds, cumulative occupied-card time was **121.34 minutes**, below the 180-minute cap. The original 1–2-hour estimate was provisional; actual cumulative card time slightly exceeded its upper estimate. Training/qualification took 74.11 minutes and independent native probes 37.39 minutes. Training loading was 31.17 seconds, disposable startup 20.07 seconds, six-case qualification 135.25 seconds and the native trajectory including diagnostics 4,229.26 seconds. These nested timers must not be added as independent costs or interpreted as kernel timings.

Physical U/T replay-gradient scopes recorded 140.71/142.60 seconds, respectively; complete branch updates, gradient measurements, state copies and memory scans add work. The padded matched experiment is not a measured optimized deployment policy. Additional replay exposure and computation differ from N. There is no serving or training-inclusive speedup claim.

Training peak allocated/reserved memory was **21.16/24.77 GiB**; sampled native-engine use reached about **54.65 GiB**. Both cards were idle afterward, with **349.216 GB** disk free. No model or historical evidence was deleted. The verified private delivery contains **50,302 files / 447,860,866 bytes**; weights, recovery tensors, teacher packets and private token/text records remain outside Git. Intermittent SSH failures delayed monitoring and the first retrieval, but triggered no scientific restart. The successful single-stream retrieval authenticated every file.

## Interpretation and decision

What held: exact native continuation, sufficient threat/support coverage, correct replay activation and strong selected-memory rescue. What failed: neither independent primary lower bound exceeded zero. The observed independent trade-off also shows less acquisition and more loss than N; source rescue alone was insufficient.

This weakens “there were no actionable threats” and “the replay could not change the selected decisions” as explanations for this test. It does not establish the underlying cause of failed transfer, an optimal coefficient, broad memorization or domain effects, or failure of all replay mechanisms. PR45's earlier fitted advantage was unresolved; this experiment supplies a stronger selected-case intervention result on a differently selected population, at different states and with different aggregation, so the two aggregates must not be pooled.

**Stop this fixed recipe at the local screen.** The single observation that would change the next design decision is a separately prespecified intervention demonstrating positive independent T/N and T/U contrasts, with useful acquisition retained. No further experiment, tuning or rollout is authorized here.

The MIR selection-and-rehearsal concept was adapted under the user's scoped exception after [source review](MIR_SOURCE_REVIEW.md); this is not an author-supported MIR reproduction. Separate CPU audits checked raw scoring, state/selection/work receipts, both reductions and all eleven ARS interpretation categories. Reviewers are separate agents of the same model family, with authorship overlap disclosed; retained GPU receipts were audited, not independently rerun.

Evidence: [primary results](PRIMARY_RESULTS.json), [all request/window effects](PER_REQUEST.csv), [per-window effects](PER_WINDOW.csv), [full probes](PROBE_OUTCOMES.csv), [selection funnel](WINDOW_SELECTION.csv), [memory outcomes](MEMORY_OUTCOMES.csv), [repair decomposition](REPAIR_TRADEOFF.csv), [state parity](STATE_PARITY.json), [updates](UPDATE_RECEIPTS.csv), [costs](WORK_AND_COSTS.json), [audit index](audit/README.md), [publication provenance](PUBLICATION.json). The original failure report remains unchanged.
