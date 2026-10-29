# Ordinary O/K continuation through events 5–8

This bounded experiment asks whether first-error-selected forward KL (K) keeps
its advantage over ordinary all-position forward KL (O) as both receive more
feedback. It extends orchestration only. The qualified `online_feedback`
training, native feedback capture, losses, optimizer and inference code are reused.

Each arm restores its **own** complete event-4 state at 512 Adam steps. Events
5–8 each collect 128 fresh responses with the preceding O checkpoint, then train
both arms on that immutable pool for two passes and 128 steps. Normal draws must
agree across arms; no anchors are forced and no RNG is reset. The inherited
7,424-step schedule remains unchanged. Each arm stops at 1,024 cumulative steps.

The two existing 64-request panels are evaluated at events 6 and 8. A third
64-request GSM8K test panel is selected with seed 20261029 and sealed before
training, then evaluated only at event 8. Its first fresh O execution supplies
the canonical tokens and termination; this is verifier agreement, not an
independent target-only reference. Prior unrelated-project exposure is disclosed.

| New work | Fixed budget |
|---|---:|
| Shared feedback generations | 512 |
| Ordinary response presentations across O/K | 2,048 |
| Adam steps across O/K | 1,024 |
| O/K evaluation generations | 1,920 |
| Complete-response generations | 2,432 |

All evaluations use greedy BF16, block size 16, 2,048 tokens or EOS, context
4,096, concurrency 1, disabled capture and three balanced interleaved repeats.
The first eight development requests at event 6 form the included 48-execution
correctness gate. Valid null/harmful outcomes continue. Correctness failures stop
and retain completed, invalid, unknown and unattempted work; no automatic retry.

`prepare.py` constructs the private input manifests from retained identities and
pinned question-only data. `resume.py` extends boundary admission without changing
the historical event-1..4 guards. `campaign.py` closes each trainer before timed
serving. `evaluate.py` performs genuine two-arm comparisons, and `report.py`
reaudits raw records and reports each panel separately. Call totals use one
validated trajectory per request; timing uses all three repeats. Historical
event-4 counts supply a count curve, never a latency learning curve.

`storage.py` admits actual phase peaks, including atomic checkpoint overlap and
the existing 8 GiB reserve. At idle event boundaries it waits for `courier.py` to
copy and verify explicit artifacts on Fenrir before retiring redundant A800
copies. Complete states and exports at events 6 and 8 are retained. Active latest
states and parent exports remain available until their dependent training ends.
Transfers never overlap timed serving. Checkpoint loading, publication, training,
capture/preparation, persistence and offload costs remain separate.

Run paths, commands, model/runtime/configuration identities and the actual source
commit are sealed per execution outside Git. Private prompts, outputs, model
tensors and raw runtime artifacts are not published. The run stops after event 8
and its report: no KRC arm, precision variant, tuning or extra mechanism probe.
