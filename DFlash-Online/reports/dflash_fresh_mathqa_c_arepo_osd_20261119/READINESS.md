# Fresh MathQA C58 / released AdaFlash loss / OSD: readiness

## Material Passport

ARS experiment-agent, bounded run. Research Stage 1 / project None is unchanged.
Current status: **CPU preparation passed; final staging, archive delivery and native qualification pending.** No GPU/model work has run for this study.

The primary contrast is C58 versus AREPO58 complete-request time; C/O and AREPO/O are secondary. This reuses the finished fixed checkpoints and changes neither loss nor weights. It is a fresh-input mathematical transfer test, not another training replication or AdaFlash's full high-concurrency system.

## What must match

The earlier PR54 startup Python-RNG mismatch was repaired before C58 training completed. There is no optimizer construction or training-startup equality test in this serving-only study. Historical verification counts from different requests are inapplicable. Only the new adapter removes that gate; the old PR54 evaluator remains unchanged. Export identity, fresh canonical output tokens and termination, native count identities and repeat consistency still must match. A mismatch stops this study; no general matching waiver is applied.

## Data and frozen inputs

MathQA test revision `4f958013c164a40745a5b771dfb005df308deb4a` has 2,985 rows. The source-bound inventory excludes 1,799 known project uses/reservations, leaving **1,186** eligible rows. The prescribed SHA-256 canonical-JSON rank with seed20261119 seals the first **1,024**. Source IDs, Unicode-NFKC/whitespace-normalized prompts and exact tokenized identities are checked; original prompt rendering is preserved. Historical audit receipts are authenticated and extended, not all old raw carriers reread. There are no declared missing essential manifests. Unknown inaccessible histories, semantic near-duplicates and model pretraining are outside this exclusion claim.

The two fixed PR54 MathQA warmups are already used and disjoint from the panel. Their original native-verifier witnesses authenticate; no warmup-reference regeneration is planned. O58 will produce all fresh canonical outputs before scoring. It remains a speculative native greedy target-verifier producer, not standalone target-only decoding.

All26 target/tokenizer/released-draft/endpoint files rehash correctly on A800. The three serving exports each contain58 BF16 tensors with matching shapes:

| Endpoint | Original scientific execution | Weight SHA-256 |
|---|---|---|
| C58 | `b4459820b3822c3d760bb3fd880e4d139fa2bbc2` | `705c6b8b81895837ffbe1fb926aa266078d6af2c9904a77642b797a81dc1f937` |
| AREPO58 | `7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a` | `fab2e033b9c2cc8c30d36e4fe676fc65eeb8087b2acc9eabad9279f0d92b19e7` |
| O58 | retained original OSD, authenticated PR54 control | `796ec9d486046d43a2b91bd0ddf32b33eb3183b7d924e9dd9ab6bc4c6b0d0856` |

Each weight file is2,097,259,104bytes and stays read-only. Original initialization/training histories remain the completed online recipes, not common-feedback loss-only training.

## Exact work and cost plan

Preserve BF16/FA3, greedy five-layer DFlash, block16, output2048 orEOS, context4096, concurrency1, disabled capture and qualified cache handling. Two-card qualification uses2 old warmups ×3arms ×2repeats ×2cards =24generations. Both cards must match the same witnesses before parallel untimed O58 canonical shards. All scored timing then uses one fixed physical A800, with the other idle and no reference tests, hashing, compression or transfers.

| Work | Planned |
|---|---:|
| Native startup/output/parity qualification |24 |
| Fresh O58 canonical outputs |1,024 |
| Scored outputs,3arms×1,024×3repeats |9,216 |
| Warmups,2×144activations |288 |
| Total complete generations |10,552 |
| Adam steps / new trained checkpoints |0 /0 |

Schedule:16groups of64perrepeat; lexicographic six arm permutations indexed by(16r+b)mod6. Each permutation appears8times. This is aggregate balance, not per-request position balance. Failed/partial work remains charged; no automatic retry.

Historical PR54 MathQA evaluation took12,971.762s. The working forecast is about4.1hours raw and5.6hours including declared1.3×+15min contingency, conditional on actual native checks. Caps remain8charged elapsed hours,12occupied-card hours,30GBadditional peakwrites and400GBshared free reserve. Readiness, archive delivery and analysis costs are reported separately with total session accounting. Initial A800 free496.078GB; bothcards idle. Fenrir has limited headroom, so the10.288GB PR54 compact archive is verified as a compressed stream without a full unpacked duplicate. No historical artifacts are deleted.

## Checks and mandatory gates

The new evaluator's12 CPU tests,10 runtime tests and2 package-assembly tests pass, including a full9,504-row synthetic native-record roundtrip, exact schedule, missing/duplicate rows, invalid canonical/weights/termination, repeat inconsistency and paired bootstrap. Independent same-team review confirms the new labels and primary contrast. The inactive direct-allocation reference separately passes16tests/100scalar-oracle cases; see[CPU_REFERENCE.md](CPU_REFERENCE.md). The archive reader separately passes23 corruption/coverage and keeper tests; the final precommit run passes63 distinct focused tests in total. These checks do not qualify nativeGPU execution.

Remaining before launch: complete member-hash-verified PR54 archive delivery and keeper reconciliation, publish/verify source staging, finish assembled CPU preflight and resource admission. Native24-generation qualification must pass before canonical preparation/scoring. No claim of completed endpoint evidence follows from this readiness report.

## Analysis and scope

Pair whole requests, average three times within each arm/request, pool totals, and use10,000PCG64 draws(seed20261119), lexicographic request-ID order and linear2.5/97.5percentiles. Native acceptance is summed nativeaccepted/summedcalls; stop-clipped acceptance is separate. Primary time gain is100(1−sumC/sumAREPO). Secondary intervals are descriptive/unadjusted. Repeats are not independent requests/seeds. A null or regression completes a valid study; no tuning, enlargement, training or subsequent event follows.
