# K+ immediate-successor shuffle control

Continue the authenticated original PR32 GSM8K KPLUS4 state at Adam step512 through one event to640. The incumbent TRUE export is the authenticated PR37 KPLUS5 export pinned again by PR38. New K and SHUFFLED trajectories each use the same128 canonical feedback responses, teacher packets, two-pass order and native anchor draws. Training seed17, BF16 AdamW, inherited7424-step horizon and the existing objective/serving settings remain fixed.

SHUFFLED changes only the detached high/low priority assignments. Within each response and first-error proposal position, recipients are sorted by canonical anchor offset; donors are sorted by SHA256 of compact UTF-8 JSON `[20261111,response_id,position,anchor_offset]`, then offset. Only valid immediate successors are eligible, including canonical EOS. Singleton or single-class groups keep the same bit values. Fixed points remain. Donor order depends on group membership, not labels or pass number. All blocks of one response share the assignment calculation, including different eight-block vocabulary chunks.

The extra student projection reads the current forward's detached hidden states. The immutable assignment tensor enters checkpointed loss calls explicitly, so backward recomputation cannot see another response's labels. The existing BF16 KL and FP32 response-wide K+ reducer remain unchanged. Ordinary K receives no redistribution.

## Execution and accounting

1. Seal actual code, model/runtime, parent, feedback, TRUE export and panel identities. No outcomes enter panel selection. Reserve peak storage before model work.
2. Run three disposable two-response windows: native TRUE, identity assignment and real shuffle. Identity requires exact losses, unclipped/clipped gradients, updated weights, Adam state/counters, RNG and next anchor draw. Real shuffle uses the same state and has independent scalar checks of masks, hash map, classes and weights. Three temporary Adam steps and six presentations, with six separate startup forwards/backwards and no extra serving warmups.
3. Train K and SHUFFLED independently:256 new Adam steps and512 response presentations in total. Discarded startup adds four forwards/backwards and zero Adam steps. Detached vocabulary projections and recomputation are counted separately.
4. Stop before redundant serving if the shuffle never changes a label or the final TRUE/SHUFFLED exports are identical. Preserve the outcome.
5. Use256 preselected GSM8K test requests (maximum remaining eligible pool295). Time each real arm twice in counterbalanced eight-request blocks:1536 completions. The first-eight correctness gate is included. Reuse221 authenticated canonical references; prepare35 missing references separately. No feedback collection.
6. On64 hash-preselected starts, run at most one parent draft block each. The parent worker runs the original2048-token-cap prefill/first native round and stops scheduling only after the round returns. This is an untimed partial diagnostic, not a serving speedup or complete-response generation. Child proposals come from the timed native traces.

Runtime entry: `python -m experiments.successor_shuffle.runtime RUN_ROOT` after actual-path sealing. `run_all.sh` restores the exact qualified compiler, Conda, cache and thread environment and records completion/failure. It makes no automatic retry. Private prompts, tokens and runtime bodies stay outside Git.

## Interpretation

Primary: TRUE versus SHUFFLED verification-call reduction, using summed paired counts and one validated trajectory per request. Also report TRUE/K, SHUFFLED/K, fresh request latency and both sweep directions. Joint paired-request bootstrap:10000 PCG64 draws, seed20261111; unadjusted95% percentile intervals. Historical timings are not pooled.

A supported TRUE/SHUFFLED gain supports informative priority placement for this one continuing seed and fixed permutation. TRUE/K is a separate incremental-benefit question. Unresolved intervals do not establish equivalence. Local repairs and weight movement do not override native serving outcomes. Stop after this event and panel.
