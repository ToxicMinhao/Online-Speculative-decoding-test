# Ordinary OSD at the matched lower learning rate

This completes the fixed event-4 O/K × learning-rate comparison. Only O_HALF is
trained, from each original O step-zero state. Ordinary all-valid-position OSD
is unchanged; LR is 1.5e-4. Native BF16 Adam, response normalization, shuffling,
sampling, full-block attention and checkpoint publication remain shared code.
Recorded presentation orders and anchors check natural draws; they never replace
those draws. Earlier source prompts remain context and dataset answers are unused.

The original eight coding pools and exact PR21 panel are reused. F, K4 and
K_HALF4 counts remain archived measurements after full record/source validation
and a 48-generation compatibility bridge. O4 and O_HALF4 receive 1,024 fresh
generations: two seeds, two methods, 128 requests and two consecutive repeats.
There is no new latency comparison. Timings are operational costs only.

## Fixed sequence and budget

1. CPU audit of original states, decoded endpoint/export equality, pools,
   ordinary schedules, source/runtime pins and every PR21 native record/repeat.
2. Seal published source, exact configurations, archive and all1,072 serving slots.
3. Admit real peak disk/host/device availability. Run F/K/K_HALF on the first
   eight requests for seed17 then29, checkpoint-batched, once each (48).
4. For seed17 then29, train four O_HALF events in fresh processes with own Adam
   and RNG continuation. Each has256presentations/128steps; total2,048/1,024.
5. After each seed's event4, evaluate original O4 then O_HALF4, two consecutive
   repeats per request. Each method's first8requests is its included16-generation
   validity gate. Full tokens, termination and repeat counts must agree.
6. Reconstruct ratios from summed calls and the fixed joint-request bootstrap.
   Preserve mixed/negative outcomes and stop after the report.

No new feedback, target-feature preparation, reference model, additional
training qualification, warmup generation, optimizer preview or post-step
correction belongs to this campaign. Any correctness/nonfinite/resource failure
stops without automatic retry or replacement requests.

## Commands

Use the workspace A800 rules and its environment helper. Inventory is CPU only:

```bash
CUDA_VISIBLE_DEVICES='' python -m experiments.o_half_matched_lr.preflight \
  ORIGINAL_CODING_RUN --previous PR21_RUN --inventory-output NEW_PREPARATION

python -m experiments.o_half_matched_lr.seal \
  --code CLEAN_PUBLISHED_SOURCE --inventory RETRIEVED_REUSE_AUDIT.json \
  --archive RETRIEVED_ARCHIVE.json --pins ORIGINAL_LOCAL_SEALED_DIRECTORY \
  --output NEW_LOCAL_SEAL --remote-root ACTUAL_NEW_RUN \
  --remote-code /data/sources/minhao/DFlash-Online

bash experiments/o_half_matched_lr/run_a800.sh ACTUAL_NEW_RUN
# Only explicit complete-boundary recovery, with no unknown scientific work:
bash experiments/o_half_matched_lr/run_a800.sh ACTUAL_NEW_RUN --resume

CUDA_VISIBLE_DEVICES='' python -m experiments.o_half_matched_lr.report ACTUAL_NEW_RUN
```

Run focused CPU tests with `python -m unittest discover -s tests -p 'test_o_half_*.py'`.
Do not substitute the old K-specific trainer or historical four-method evaluator.

## Retention and resource scope

Keep the original artifacts unchanged. For new O_HALF trajectories retain genuine
initial states and complete event2/event4 states/exports. New event1/event3 tensors
are retired only after their immediate successor passes full saved/live/export
checks, with exact identities and lost resume capability recorded. All receipts
and metadata remain. This implements the handoff's declared rolling policy.

Fresh additional-write peak is75,394,485,270bytes:23weight-sized tensor slots,
8GiB compiler allowance,10%slack and the unchanged4GiB compact/workspace plus8GiB
reserve. The last16 identical-shape worker caches occupied at most61MB each;
the new allowance covers eight workers without changing cold-cache semantics.
Resume validates chronological boundaries first and never credits unremoved
pre-existing artifacts as free space. Host CPU state checking requires80GiB.

## Interpretation

Primary is K_HALF versus O_HALF, conditional on two fitted seeds and the reused
128-request development panel. The same10,000 PCG64 request-bootstrap draws
(seed20261102) carry both seeds and all methods. Secondary intervals are
descriptive/unadjusted. Report the complete2×2 matrix, common-F interaction,
within-seed gain/harm decomposition and the prespecified A/B/C/D/E decision.
Equal LR does not imply equal optimizer displacement. This does not isolate
selection from normalization, establish seed-population robustness, or provide
fresh confirmatory evidence after the panel helped select K_HALF.

Original O/K execution:2705d992b133196be162229e1d5a08dd9c4f9cbb.
PR21 execution:d10521a1aa7d256bb75b37a6aae07586ae7cc8b3.
PR21 results:649e0b34066e217a3f918d82012d2a631c447ad0.
The actual new execution commit is recorded in its sealed PLAN and every report.
