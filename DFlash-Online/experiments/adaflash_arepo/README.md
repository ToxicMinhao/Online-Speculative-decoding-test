# Released AdaFlash loss supplement

This adds one arm, **A_repo: released AdaFlash loss under our matched online
protocol**. It preserves the completed O58, K+58 and paper-A58 checkpoints and
reports. The loss ports AdaFlash commit
`478bf0ff7a3a5232b13d0f14e752d5fe1da0b718`, including its operation ordering and
intermediate dtypes. It does not use the earlier common-arithmetic clipping
proxy, K+ position selection, or the adaptive length head.

The new arm starts independently from the released draft and fresh seed-17
BF16 AdamW. The original 7,473 ordered GSM8K requests produce its own feedback.
There are 58 events of 128 responses, two shuffled passes, microbatch one and
accumulation two: 14,848 presentations and 7,424 Adam steps. The last 49
responses cause no update. The learning rate, horizon, full context, native
teacher conditioning, sampler, gamma-seven weights and response averaging are
inherited unchanged from PR29.

The launcher first checks code, input, model, runtime and qualification
capacity. The separately bounded qualification compares the port and literal
source loss, gradients, clipping decisions and six disposable native updates.
Any nonfinite quantity or failed parity stops execution; it is never repaired
by substituting another loss. All temporary scientific state is discarded.
Full campaign storage admission is a separate gate after qualification.

The full-capacity budget includes native trace evidence for all 19,765 new
complete-response generations, rolling tensor overlap, compiler/workspace
allowances and the unchanged eight-GiB reserve. Retired tensor paths are recorded
after compact evidence and the replacement state/export are validated. The
old controls' tensors cannot count as capacity for the new arm.

All four frozen endpoints are measured freshly on the unchanged 1,024-request
panel, with three repeats: 12,288 timed generations. The presealed order uses
each of the 24 method permutations twice; each method occupies each position
equally within every repeat. Complete-request timing includes cache maintenance.
Loading/switching and adaptation costs remain separate. No training, capture or
artifact transfer runs during timed requests.

Primary comparison: K+58 versus A_repo58 paired request latency, using the
existing 10,000-draw paired bootstrap with seed 20261102. Every completion must
match the retained canonical tokens and termination; repeats must have consistent
verification counts. Invalid or incomplete evidence remains visible. There is
no efficacy stop, tuning, baseline retraining or automatic follow-up.

Implementation: `training.py` gives the new arm a separate contract while
reusing the native trainer and PR29 observations. `campaign.py` preserves the
qualified online sequence and first-event checks. `seal.py`, `preflight.py` and
`storage.py` bind inputs and enforce capacity. `run_campaign.sh` orders the
qualification, admission, trajectory, endpoint and reports. Run-local private
responses, features and checkpoints remain outside Git.
