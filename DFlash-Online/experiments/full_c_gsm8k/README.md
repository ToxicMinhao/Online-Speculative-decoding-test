# Released-start full C58 comparison

This campaign tests the unchanged bounded prefix allocator throughout the complete
GSM8K adaptation trajectory. It does not resume the short PR53 candidate.

The PR25 controller, target capture, response loss, natural anchor sampler,
optimizer, atomic publication and complete-state checks are reused. `training.py`
adds the existing C allocator and its observations. Neither
`online_feedback/objective.py` nor `online_feedback/prefix_allocation.py` changes.
The original step-zero witness must match released weights, fresh Adam and seed-17
RNG exactly before training. C collects its own responses and teacher features.

Production has 58 events, 7,424 Adam steps, 14,848 response presentations and 7,473
served training requests. The last 49 requests have no update. Four inherited
capture/publication correctness generations are counted separately. There are no
intermediate efficacy panels, RNG forks, control retraining or new loss variants.

Qualification uses four authenticated existing feature packets only as a bounded
correctness fixture. Six disposable pair updates test bare native K+, wrapped
native, zero-cap C, active C and uninterrupted/reloaded continuation. Two startup
forwards/backwards and zero startup optimizer steps are separate. Twenty-four
serving checks use the first two requests of each old panel, both retained
controls and the disposable active export, twice each, without generation warmups.
Failed attempts remain charged; none is automatically retried.

After qualification, the disposable tensors retire with fingerprints. Production
retains its initial state and a validated latest recovery, temporarily preserving
the predecessor and publication overlap. Completed-window feature tensors retire
after durable compact evidence and publication validation. This deliberately
ends historical tensor replay; it does not delete outputs, anchors or traces.
The final C58 complete state and export remain available through review.

Final evaluation freshly compares O58, KPLUS58 and KPLUS_C58 on the unchanged
1,024-request GSM8K and 1,024-request MathQA panels. There are 18,432 scored
generations and 576 fixed warmups. MathQA measures serving transfer of GSM-trained
weights. Both panels are reused research panels. All timing occurs on physical
A after training on physical B has ended, with both cards checked for contention.

Native accepted tokens are committed progress minus verification calls. The
trace's stop-clipped accepted-proposal count is also reported, separately.
All three contrasts use the same 10,000 paired whole-request bootstrap draws per
dataset (PCG64 seeds 20261118/20261119), including all timing repeats together.
Intervals concern these fixed endpoints and panels, not independent training runs.

The run admits at most 100 GB new allocation and leaves 400 GB shared free after
projected writes. Compact evidence's 20 GB allowance is sizing, not a deletion
threshold. The original campaign clock bounds charged execution at 24 hours and
occupied GPU time at 40 card-hours. Admission requires consumed time plus 1.3
times forecast remaining work plus 30 minutes. Prelaunch inventory/cleanup is
recorded separately. A favorable or unfavorable result does not change the plan.

Entrypoints: `prepare.py` packages exact inputs; `preflight.py` verifies the
complete staged source/dependencies on CPU; `launch.py` owns bounded qualification
and the synchronous trajectory; `evaluation.py` preserves raw records before
checking them; `analysis.py` reduces valid paired records. Private data and
runtime artifacts stay outside Git. Public reports pin the actual execution
commit and distinguish code review from completed native qualification.
