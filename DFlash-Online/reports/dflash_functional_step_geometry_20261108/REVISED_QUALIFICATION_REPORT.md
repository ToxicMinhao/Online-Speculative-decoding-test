# Revised FSG qualification: formula check passes; ordinary-gradient control stops continuation

**Qualification remains incomplete. The new independent same-layout check passed exactly on the bounded GSM8K fixture, and the maximum-shape resource check passed. The next SCALAR case failed the unchanged exact ordinary-gradient control before its Adam step. No pilot was started.**

Execution: [`1c72903a811de457cbda10da178aac8cc25d19e1`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/1c72903a811de457cbda10da178aac8cc25d19e1), [PR #33](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/33). See [machine-readable summary](REVISED_QUALIFICATION_SUMMARY.json) and [ordinary-gradient differences](REVISED_ORDINARY_GRADIENT_DIFFERENCES.csv). These results use ARS experiment-agent execution/validation and disclosed same-team reviews, not an independent GPU replication.

## The revised contract was implemented

The production auxiliary path remains native BF16, with eight-block vocabulary-head chunks and its actual final partial chunk. The strict realized update-norm and actual protected-KL guards remain unchanged. No FP32-head substitution, tolerance increase, new optimizer update or algorithm retuning was introduced.

An independent implementation now computes protected prefixes, response weights, detached sensitivity inputs, output energy and auxiliary gradients. It has its own prefix recurrence and formula implementation; it does not invoke the production formula helper as its reference. The native comparison uses identical physical head shapes. Separate small CPU tests compare the mathematics against FP64 calculations.

The existing numerical criteria remain coordinate-wise absolute error at most `1e-5 + 0.016 * abs(reference)` and global relative L2 error at most `0.016`. **Historical cross-layout comparisons remain failed under their original criteria.** They are retained as layout-sensitivity evidence under the explicitly revised protocol; they have not been erased or relabeled passes.

## Completed and outstanding checks

| Check | Result |
|---|---|
| CPU correctness and regression suite on A800 | 158 tests passed |
| Source, input and original-state admission/postcheck | All 3,520 sealed files passed both checks |
| Previously completed GSM8K strength-zero/native-fallback cases | Authenticated and reused; no repeated Adam steps |
| Independent native GSM8K formula check | Passed; all 58 parameter gradients exactly equal |
| Maximum-shape resource fixture and state restoration | Passed |
| GSM8K SCALAR ordinary-gradient exactness | **Failed before Adam or scalar construction** |
| GSM8K SCALAR own-state proposal, norm/KL and final-state checks | Unexecuted |
| MathQA native cases and independent formula check | Unexecuted |
| Eight native serving checks | Unexecuted |
| Main events-5–6 pilot | Not started |

The independent numerical fixture uses the first nine retained anchors, producing head chunks of eight blocks and one block without resampling. Pre-update hidden states, proposals, protected masks and weights match exactly. Both implementations give output energy `0.28226443087590225`; all **1,048,626,432 gradient coordinates agree exactly**, with zero tolerance violations and zero relative L2 discrepancy. This qualifies that bounded GSM8K comparison only, not every shape or the unexecuted MathQA check.

The separate resource fixture uses two responses, each expanded to **512 anchors and 2,272 input positions**, with four simultaneous native parameter-gradient vectors. It repeats retained anchors deterministically and adds masked synthetic context/zero teacher features. It tests storage and execution capacity, **not source fitting, transfer, or acceptance of a representative FSG correction**. Peak GPU allocated/reserved memory was **47.018 / 48.564 GB** (43.789 / 45.229 GiB). Both helpers restored committed weights, RNG, modes, ordinary gradients and optimizer tensor identities/versions; the encompassing scientific-state equality check also passed.

Implementation: [independent oracle](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/1c72903a811de457cbda10da178aac8cc25d19e1/DFlash-Online/experiments/functional_step_geometry/same_layout_oracle.py#L99), [native comparison/resource checks](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/1c72903a811de457cbda10da178aac8cc25d19e1/DFlash-Online/experiments/functional_step_geometry/qualification_checks.py#L130).

## The new failure is a different control

SCALAR must first perform exactly the same ordinary K+ learning operation as the native control. Its own FSG computation and scalar norm matching come afterward. The assertion stopped execution at that initial ordinary-gradient comparison, before SCALAR's optimizer step and before any SCALAR auxiliary forward.

| Recorded comparison | Observation |
|---|---:|
| Parameter tensors differing | 49 / 58 |
| Gradient coordinates differing exactly | 155,364,420 / 1,048,626,432 |
| Maximum absolute difference | 0.00048828125 |
| Global absolute discrepancy L2 norm | 0.001644496259489 |
| Nonfinite coordinates | 0 |
| Ordinary response observation records | Both equal after removing only the event label |

These are **exact-equality differences**, not coordinates outside the auxiliary test's relative tolerance. The ordinary-gradient gate was not relaxed by the reviewer's change to the auxiliary oracle. The comparator did not retain the full reference-gradient norm, so a global relative error cannot be reconstructed reliably from its compact receipt and is not reported.

The equal response observations cover recorded losses, masks/counts and exposure; both presentations used the original retained anchors. They do not prove equality of every unrecorded intermediate logit or backward tensor. The first completed presentation row also matches the ordinary replay exactly. The second response's observation was saved before the boundary failed; the second completed presentation row is consequently absent.

**Confirmed:** the new formula check passes; recorded scientific state restores exactly; the ordinary-gradient control fails. The failed SCALAR case performs no native Adam step, no scalar projection and no serving generation.

**Not established:** a formula bug, a state-restoration bug, a defective GPU kernel, or harmless numerical noise. Source review found no demonstrated error in the inspected production formula or helper restoration. There is a confirmed qualification-procedure difference: the resumed reference gradients were captured during the fresh process’s first two training backwards (also designated startup), while SCALAR ran afterward, following the auxiliary/resource work. The original qualification had discarded a separate warm-up before capturing its ordinary reference. This asymmetry needs to be controlled; it has not yet been proved to cause the mismatch. Compiler/kernel caches and allocator state are not covered by the scientific-state snapshot; their influence remains a hypothesis. Some late-layer tensors agree while discrepancies spread through earlier layers, which is consistent with a backward numerical perturbation but does not localize its cause.

Control source: [reference-gradient replay](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/1c72903a811de457cbda10da178aac8cc25d19e1/DFlash-Online/experiments/functional_step_geometry/qualify.py#L439), [unchanged exact pre-Adam assertion](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/1c72903a811de457cbda10da178aac8cc25d19e1/DFlash-Online/experiments/functional_step_geometry/qualify.py#L193). The retained failure remains in `qualification_003/qualification/gsm8k/SCALAR/NATIVE_GRADIENTS.json`; no complete model-sized gradient dump was written.

## Work and costs

| Work | New in this attempt | Cumulative original qualification | Original ceiling |
|---|---:|---:|---:|
| Ordinary response forwards | 2 | 8 | 16 |
| Ordinary response backwards | 2 | 8 | 16 |
| Additional response forwards | 16 | 34 | 96 |
| Additional response backwards | 8 | 20 | 32 |
| Adam steps | **0** | 3 | 8 |
| Serving generations | **0** | 0 | 8 |

The new additional work consists of two forward/backward ordinary-gradient replays without Adam, five forwards/two backwards for the independent numerical check, and nine forwards/four backwards for the resource fixture. Including SCALAR's two ordinary presentations, this attempt used **18 response-level forwards and 10 backwards**. Chunk projections and checkpoint recomputation remain separately recorded in raw work receipts. Earlier separately authorized numerical diagnostics remain disclosed separately; they are not hidden in these original qualification totals.

The supervisor ran **320 seconds** (01:03:35–01:08:55 UTC, November 9), exit 1. The native qualification runner used **175.956 seconds**; the independent formula and resource helpers used **2.966 and 24.898 seconds**. These nested times are not additive training-overhead estimates. Startup, CPU tests, dependency checks and export accounting remain separate in the retained receipts. One exact copy of the already-retained ordinary endpoint was exported for later serving checks; it is not a newly trained endpoint.

All **69 compact files / 8,573,794 bytes** were copied to Fenrir and independently verified against the remote inventory. Original failures and recovery dependencies are preserved. A800 postflight recorded **55.835 GB free**, above the qualification reserve; both GPUs were idle after the controlled stop. This is not storage admission for the larger pilot. After verifying compact retrieval, only the redundant 21,435,734-byte source-transfer archive was removed on each host, with exact deletion receipts; published/extracted source and scientific dependencies remain. A800 free space after cleanup was 55.857 GB. No scientific replay capability was lost.

## Decision and smallest unresolved check

**Do not launch the pilot. Do not loosen the ordinary-gradient check, norm guard or KL guard.** The reviewer's revised auxiliary check resolved the GSM8K formula comparison, but it does not waive this separate ordinary-update control.

The smallest useful follow-up is a bounded, zero-Adam comparison of ordinary K+ and SCALAR's pre-Adam gradients under identical startup/state, with the auxiliary/resource operations isolated from that comparison. It should test whether those intervening operations change subsequent ordinary backwards before attributing the difference to SCALAR. No such follow-up or retry was executed here.

The stopped two-presentation SCALAR attempt remains charged. Only eight ordinary forwards/backwards remain under the original ceiling—exactly the unexecuted MathQA cases—so repeating SCALAR requires an explicitly accounted budget revision or a genuinely resumable retained computation. Compact receipts are not resumable gradient tensors. We have not reset counters or relabeled failed work to make a retry fit.

Representative mechanism activity is still unresolved: the authenticated GSM8K FSG case falls back to native, and MathQA has not run. The synthetic expanded-resource candidate is not evidence of a naturally accepted production correction. It is premature to conclude either that all representative windows fall back or that FSG improves learning.

## Material Passport

- Workflow: ARS experiment-agent run/validate; numerical and execution evidence **ANALYZED**, overall qualification **INCOMPLETE**.
- Source: execution commit `1c72903a811de457cbda10da178aac8cc25d19e1`; subsequent report publication is documentation-only.
- Evidence: revised sealed plan, raw work ledger, native independent-oracle/resource receipts, exact-state receipts, SCALAR failure, per-response observation records, pre/postflight integrity and same-team source/work/numerical reviews.
- Review limits: reviewers share the project team and have implementation authorship overlap; no independent GPU replication. All 11 ARS interpretation categories are considered in the retained work review.
- Scientific limits: no new Adam step, no new feedback or serving generation, no accepted new SCALAR result, no MathQA result, no pilot, no FSG efficacy claim. The historical cross-layout failures remain failures.
