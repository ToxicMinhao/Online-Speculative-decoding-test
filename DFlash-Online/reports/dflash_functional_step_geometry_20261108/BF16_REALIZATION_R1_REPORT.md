# FSG R1: norm-feasible BF16 realization qualification

## Material Passport

- ARS-Codex3.24.2 experiment-agent run and validate; native qualification executed, same-team review with implementation overlap.
- Qualification execution: `95aeb869189da2d7d4f2109b93c05fd3bd9b7944` ([PR33](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/33)).
- Scope: one supplementary four-case qualification. Historical numerical failures and the previous all-fallback result remain unchanged.

## Result

**The revised implementation qualifies on both retained natural windows.** Both FSG cases commit an endpoint distinct from the prestate and native update, within the unchanged realized BF16 norm bound and protected-KL rule. Both SCALAR cases run their own norm matching and actual final-candidate check, then restore native because the KL difference lies inside the existing numerical deadband. This is valid fallback, not a correctness failure.

The conditional pilot's activity/correctness gates are satisfied. Its separate native bindings and188.444GB storage admission remain prerequisites; the pilot has not yet run. Neither an accepted correction nor lower fitted protected KL establishes held-out drafting benefit.

## Numerical outcomes

| Dataset / arm | Native norm | Raw BF16 FSG norm | Feasibility scale | Final proposed norm | Candidate / native protected KL | Outcome |
|---|---:|---:|---:|---:|---:|---|
| gsm8k / FSG | 1.4583258594 | 1.4584726072 | 0.9924452901 | 1.4583258578 | 0.051821266858 / 0.072695263441 | Accepted nontrivial correction |
| gsm8k / SCALAR | 1.4583258594 | 1.4584726072 | 0.9924452901 | 1.4583258578 | 0.072695263446 / 0.072695263441 | Native fallback: numerical ambiguity |
| mathqa / FSG | 1.1423660015 | 1.1424440155 | 0.9965760112 | 1.1423660001 | 0.095945269270 / 0.129428663372 | Accepted nontrivial correction |
| mathqa / SCALAR | 1.1423660015 | 1.1424440155 | 0.9965760112 | 1.1423660001 | 0.129428663388 / 0.129428663372 | Native fallback: numerical ambiguity |

Every feasibility search performed exactly24 iterations and selected the feasible lower bracket. The final write uses the original continuous displacement and the same arithmetic as probes. No norm tolerance, KL deadband, lambda, margin floor, precision or head layout changed. At a=0 the exact prestate is used; neutral/degenerate bypass behavior remains separately covered by the retained native proofs and focused CPU tests.

SCALAR's absolute target-norm mismatches were2.137e-12 (GSM8K) and2.097e-12 (MathQA), with relative mismatches1.465e-12 and1.836e-12. Its hypothetical FSG endpoint received no KL veto; only its own scalar endpoint was scored. These near-perfect norm matches do not establish equivalence or useful learning.

## Correctness and work

- Fresh warmed ordinary gradients, actual native Adam endpoint, preserved advanced Adam moments/counters, RNG/mode/state restoration and exact export checks pass for all four cases. Each case independently restores its dataset's original complete KPLUS4 state.
- The independent same-layout formula, exact native/neutral controls and maximum-size resource tests are authenticated retained evidence, not newly rerun tests. The old cross-layout coordinate failure stays a failed sensitivity diagnostic.
- All4 fresh serving generations match the retained canonical token IDs and termination. GSM8K FSG/SCALAR use19/19 verification calls; MathQA uses22/20. These are single qualification requests, not paired pilot estimates or evidence of a serving improvement.
-233server CPU checks passed before native work.197local FSG checks also passed; these overlapping counts are not independent experiments.

| New qualification work | Completed | Ceiling |
|---|---:|---:|
| Native Adam applications |4|4|
| Ordinary response forwards / backwards |8/8|8/8|
| Additional forwards / backwards |32/16|32/16|
| Serving generations |4|4|
| New feedback / teacher preparation |0/0|0/0|

Additional work includes4discarded startup forwards/backwards,4warmed ordinary-reference forwards/backwards, and24geometry/candidate forwards with8auxiliary backwards. Startup states are discarded; neither they nor the four updates initialize the pilot. Failed/unknown model work is zero. Parameter-only feasibility and scalar scans are separately recorded and add no model forwards or Adam updates.

## Costs and expected pilot duration

| Dataset / arm | Feasibility scans s | Scalar matching s | Full qualification case s | Peak allocated / reserved GB |
|---|---:|---:|---:|---:|
| gsm8k / FSG | 26.569 | 0.000 | 141.977 | 22.770 / 25.822 |
| gsm8k / SCALAR | 26.326 | 14.915 | 151.014 | 22.770 / 25.982 |
| mathqa / FSG | 26.301 | 0.000 | 143.251 | 38.422 / 40.332 |
| mathqa / SCALAR | 26.221 | 14.882 | 156.000 | 38.422 / 40.647 |

The qualification runner took945.407s (15m45s). The recorded supervisor interval04:10:56–04:32:18UTC is21m22s, including CPU/preflight and the disclosed metadata-only preflight repair. The final source/input postflight passed in54.422s. Full case/hook timers include validation observers and nested work; they must not be multiplied or added as pure algorithm overhead.

The planned pilot is provisionally **12–18hours after launch**, excluding storage preparation and network outages. Feasibility scans alone extrapolate to roughly7.5h over1,024filtered steps; SCALAR matching adds roughly2.1h. Ordinary training, other geometry work, the16diagnostic windows, shared feedback, loading/publication and3,072endpoint completions add to that. Only one window per dataset was measured; activity/length differences can change actual duration. This is an estimate, not an execution result.

## Provenance and preparation issues

An initial extraction-stage storage check rejected launch before model work. After retiring identified disposable downloads/caches,43.0206GBfree admitted the unchanged40GiBqualification requirement. The first preflight then found one missing33,390-byte completed-retrieval manifest. All10,742other sealed dependencies existed. The exact retained manifest was copied and checked against its presealed identity; failed pre/postflight and exit records were preserved. No input tokens, anchors, recovery weights, code, tolerance or scientific allowance changed, and no model case was repeated.

All206compact records (11,756,986B) were retrieved and independently rehashed without mismatch. Raw private tokens, model weights and optimizer payloads remain outside Git. Four exports and exact case/state/serving evidence remain on A800. The publication includes compact case measurements; the private bundle contains full receipts and work ledgers.

## Interpretation and next authorized scope

R1 solves the observed realization obstacle on these two windows: a nontrivial continuous direction can now survive BF16 rounding while respecting the strict norm bound. It does not yet show the direction improves held-out drafting. The current SCALAR fallbacks are outcomes of its unchanged final check and remain visible.

Proceed only with the already agreed KPLUS/FSG/SCALAR events5–6 pilot after separate admission. Keep all panels, hyperparameters,16diagnostics and budgets fixed. No extra precision exploration, cases, sweeps or58-event run is authorized by this result. If FSG and SCALAR both improve over KPLUS but FSG has no demonstrated advantage over SCALAR, directional geometry remains unestablished; that does not establish equivalence. If SCALAR improvement is itself unresolved, the result is inconclusive.

Statistical scope: these four fixed qualification cases are not an efficacy sample. No p-values, bootstrap claims, independent-seed claims or aggregate performance extrapolation are drawn from them. Historical timing is used only for the provisional runtime estimate, not a learning comparison.

Same-team audit:641 assertions pass, including independent compact-file rehashes, raw work accounting, native/state receipts, feasible-bracket arithmetic, own-state SCALAR matching and pooled/per-response KL decisions. This is an evidence/code audit, not an independent GPU replication.

## Pilot preparation after qualification

The standing completed-experiment retention policy was applied only after a separate audit:4,250 exact paths/4,243 inodes, with161,369,784,320 allocated bytes in the allowlist. All paths were removed as recorded, with no selected final-weight or active dependency removed. Settled A800 free space is195,702,054,912B; this exceeds the188,444,277,887B pilot bound, subject to the fresh launch admission after source/package staging. The retired data comprise completed teacher/replay tensors, nonessential initial/recovery copies and four obsolete KP event-1 exports. Final selected weights, original KPLUS4 recovery, current feedback dependencies, canonical outputs, compact verification traces and raw results remain. These retired tensors are not archived; their corresponding optimizer continuation, feature replay and intermediate-checkpoint reevaluation capabilities are lost. Exact manifests and preserved CPU-decoded metadata are private execution evidence.

A metadata-only real-input check caught an inherited canonical-schema mismatch before pilot execution. The old canonical files lacked the prompt/provenance fields required by the new evaluator. A dedicated binder authenticates all512 original native records against PR32 sealed hashes, full traces, exact prompt/output token IDs, termination and pinned model/runtime identities, then supplies truthful metadata to the strict evaluator. It does not change canonical tokens or generate new references. Original compact canonical files remain unchanged. This is a pilot input-binding repair, not a mechanism or scientific-setting change.
