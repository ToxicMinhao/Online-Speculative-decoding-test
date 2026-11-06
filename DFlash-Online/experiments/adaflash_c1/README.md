# Paper-defined AdaFlash OPD at concurrency one

This experiment trains **A58** once and retimes it with retained **O58** and
**K+58**. It tests the paper-defined OPD objective in DFlash-Online, not the
complete AdaFlash system or the released trainer's different clipping order.

The only training treatment is `online_feedback.adaflash_opd`: each valid
response proposal receives 0.8 target-argmax CE plus 0.2 reverse KL, clipping
each vocabulary contribution from above at 0.01 before summing. Negative terms
remain untouched. The qualified BF16 projection/probability arithmetic,
position decay, all-position OSD masks and response normalization are retained.
There is no K/K+ prefix selection, frozen reference, adaptive length head or
additional model forward for the new objective.

`training.py` binds that installer to the persistent native full-horizon trainer.
State validation checks the A method contract first, then uses a read-only
metadata view for the old structural validator. Actual snapshots and training
continue to carry A's real contract; no tensor, RNG, counter or Adam state is
substituted. Old K/K+ guards remain unchanged.

The original 7,473 GSM8K requests retain their order. A collects its own native
feedback, trains after every 128 responses for 58 events, and serves the final
49 without updating. Two shuffled passes, microbatch 1 and accumulation 2 give
14,848 presentations and 7,424 AdamW steps. Initialization is released DFlash
with independently created seed-17 RNG and fresh native BF16 AdamW, checked
against the original initial state. No retained baseline is retrained.

Before model work, tiny CPU and GPU scalar/gradient fixtures distinguish entry
clipping from sum clipping and preserve failure details. They use zero model
forwards and zero Adam steps. Four inherited native correctness generations
(capture-off parity and updated/frozen/reloaded serving) remain separately
accounted and are not timing samples. There are no additional warmup generations.

The final original 1,024-request panel is evaluated with three repeats and three
arms: **9,216 timed completions**, plus 7,473 online generations and four
correctness generations, **16,693 complete generations total**. All six method
orders occur eight times across 64-request blocks. Every output must match the
retained canonical tokens/termination, and each arm's counts must repeat. No
historical count equality is imposed across distinct trained methods.

Primary: K+ versus A complete-request time, `100*(1-sum(K+)/sum(A))`, with
10,000 paired-request bootstrap draws, PCG64 seed 20261102. Average timing
repeats within request; use one repeat-stable trajectory for counts. Positive
95% interval favors K+, negative favors A, and a crossing interval is
inconclusive. Each arm's comparison with O is also reported. Native accepted
draft tokens per verification and committed output progress per verification
are labelled separately.

The launcher records publication/source bindings, storage admission, runtime
checks and all work. Training and transfers must end before timed serving.
Rolling retirement preserves compact evidence, the initial/final state and
current recovery boundary; exact retired paths and lost replay capability are
recorded. All costs remain separate, including nested clocks and synthetic
qualification. A valid unfavorable result continues without tuning. Correctness
or nonfinite failures stop with evidence preserved and no automatic retry.

Deliverables: `REPORT.md`, `SUMMARY.json`, `PAIRED_REQUESTS.csv`,
`ALL_REPEATS.csv`, `REPEAT_TOTALS.json`, `VALIDITY.json`, `TRAINING.json`,
`TRAINING_EXPOSURE.csv`, and `EVENT_COSTS.csv`. Private raw outputs and traces
remain outside Git. Every report identifies the execution commit.
