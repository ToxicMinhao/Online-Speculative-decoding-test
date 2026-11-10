# K+ verification-transition replay

This isolated diagnostic asks whether a **local advance can lead to a more expensive continuation**. It freezes each dataset's ordinary KPLUS4 and KPLUS6. It does not train a model or implement the proposed frontier-allocation algorithm.

The adopted handoff is in `reports/dflash_verification_transition_replay_20261110/HANDOFF.txt`. Start with `contract.py` (fixed scope), `data.py` (selection and binding), `campaign.py` (four-cell experiment), and `native.py` (successor reconstruction). The ordinary serving runner and external runtime are unchanged.

## Conditional comparison

At the first round with different *actual committed progress*, retain successors u4 and u6. Earlier committed prefixes must agree, even if earlier proposals differed.

| Destination draft | Source u4 | Source u6 |
|---|---|---|
| KPLUS4 | C44: authenticated native suffix | C46: crossed suffix |
| KPLUS6 | C64: crossed suffix | C66: authenticated native suffix |

Only the two crossed cells need new suffix execution. `R4=C46−C44` and `R6=C66−C64` measure the restart effect with the draft fixed. Positive values mean u6 costs more calls. This is an expensive **farther** restart only for a genuine KPLUS6 advance, with the first-rejected-position repair check reported. `D=C66−C44=R4+M6=M4+R6` must equal the original total-call difference.

Terminal successors have zero remaining calls. No-divergence requests have NOT_APPLICABLE cells and zero transition effects. Incomplete cells remain missing. Equal total calls do not imply equal transitions. Boundaries with at most16 canonical tokens remaining stay in the main sample and are also distinguished from the prespecified interior subset.

## Native state and the only startup gate

The first hash-selected request per dataset supplies both checkpoints' neutral successor tests after round1 and ceil(R/2), deduplicated and nonterminal only. No request substitution is allowed. At most8 neutral suffixes are available.

A successor owns its request/termination/output-budget metadata, valid source target KV, request mapping/allocator state, native seed/positions and original committed target-feature ingestion segments. The constructor restores source target state and replays those same ingestion chunks under the destination draft weights. Source draft cache is retained only for neutral bitwise comparison; it is never transplanted into a changed draft.

Neutral reconstruction must reproduce **all15 proposals per remaining round**, committed progress, canonical output, termination and call count. Inspectable valid-prefix cache bytes must agree. The code preserves full original `max_new_tokens` and accumulated output IDs, so suffix generation consumes the original remaining budget. There is no offset memoization, training/Flex approximation or alternate canonical-cache experiment.

CPU fixtures exercise cache destruction followed by actual reconstruction in a small deterministic native stub. They do **not** establish GPU/FA3 equivalence; that remains the required native gate. A failed constructor stops the diagnostic as technically incomplete.

## Budget and admission

The exact selection uses SHA256 of `kplus_transition_replay_v1\n{dataset}\n{request_id}`, tie-breaking by ID,64 of each original256-request panel. IDs are written before outcomes are loaded. Execute ascending rank, interleaving GSM8K and MathQA.

All attempts, including failed/startup work, share one durable budget authority across the host and scheduler:

- At most256 original untimed completions,256 crossed suffixes,8 neutral suffixes.
- At most32,768 newly executed draft blocks or standalone verifier rounds.
- At most4 cumulative occupied GPU-hours; one serving GPU is used. A30-second cleanup reserve stops dispatch before the ceiling, and the supervisor terminates all registered owned native process groups on timeout. It never kills pre-existing jobs.
- Zero training, backwards, feedback collection or timing sweeps.

A normal proposal plus verification counts once. Target calls and context-forward/token counts are separate. CUDA graph setup forwards are charged. Pending work after a crash remains unknown, never completed. Loading and context rebuilding count toward occupied GPU time. The full budget remains visible when work is incomplete.

Before loading, `preflight.py` verifies the sealed input/selection bytes, four weight hashes, target/tokenizer files, seven pinned runtime files, exact execution commit/clean tree and request limits. Storage admission reserves8GiB plus2GiB for compact traces/logs; live successor tensors stay in host memory and are released per request. Actual availability and native memory feasibility remain launch checks.

## Entry points

Preparation is CPU-only and already completed for this implementation:

```bash
python -m experiments.verification_transition.data --inventory /private/INVENTORY.json --output /private/prepared
```

After implementation review and A800 admission, the eventual native command is:

```bash
python -m experiments.verification_transition.run \
  --inputs /private/prepared/INPUTS.json \
  --output /data/sources/minhao/runs/dflash_verification_transition_replay_20261110/diagnostic_001
```

Keep `SEAL.json` and `SELECTED_IDS.json` beside `INPUTS.json`. Source the required A800 environment helper first and use the pinned interpreter. The command includes the neutral gate and stops on correctness or resource failure; it does not retry. It is documented here, **not executed as part of the implementation stage**.

CPU checks:

```bash
python -m unittest discover -s tests -p 'test_verification_transition*.py'
```

## Evidence and interpretation

Private raw records are saved before validation. Public reports contain selected IDs, four-cell tables, repair/boundary flags, all missing work, source/checkpoint identities and the work ledger, with no prompt text/token arrays/tensors.

Analysis uses10,000 paired full-request resamples per dataset (PCG64 seed20261110), recomputing advance/interior strata within each draw. It includes unconditional and conditional effects, medians/spread, empty strata, contributor shares and leave-one-request-out signs. Sparse or conflicting evidence remains descriptive/inconclusive. Incomplete execution suppresses inferential conclusions rather than treating completed requests as a replacement panel.

Even repeated interior positive R4/R6 only supports proposing a separate short allocation pilot with K+ and matched random controls. R4-positive/R6-resolved cases do not establish that KPLUS6 still needs this mechanism. Untimed replay provides no serving-speedup claim; this decomposition is not a unique attribution of learning gains.
