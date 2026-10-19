# Five retained corrections: bounded hindsight diagnostic

This separate entry point performs no training, routing, campaign restart or
state update. It reads only the specified main-campaign after-turn states:
wildchat_00 t1/m0, wildchat_01 t2/m1, wildchat_02 t6/m3, wildchat_19 t15/m2,
personamem_persona42 t3/m2 (zero-based modules). The final saved matrix is used
unchanged even if a scheduled write step was blocked.

`retained_replay.py --campaign-root PATH --out FRESH_PATH` checks retained
inputs, snapshot/feedback identities and storage without starting a GPU worker.
Add `--execute` to run exactly one frozen and one coefficient-1 singleton
generation per case. The five fixed singleton corrections bypass routing and
stay fixed during generation. Runtime/model sources, greedy decoding, block10,
1,024-or-EOS and full histories remain pinned to the original native configuration.
Every request starts with valid cold caches and capture disabled.

Durable pre-execution claims prevent retrying a generation or feature
reconstruction. If an execution fails after its claim, investigate retained
evidence; do not restart this controller in another directory to fill the gap.
Frozen counts and both output sequences must match the original reference.
The singleton head hook must actually execute with the bound matrix and one-hot
coefficients before its comparison is valid. Routing-bypassed timings are not
deployment speedups.

Matching original precision evidence is reused only after exact matrix,
projection, token, objective and sampling identity checks. Otherwise the frozen
precision worker can reconstruct each case once using its saved anchors,
weights and tokens, without calling a sampler or constructing an optimizer.
Reconstructed compressed statistics must match retained statistics before
native/reference loss changes are reported. Missing or inconsistent precision
evidence remains unavailable; it does not trigger another generation or replay.

The storage preflight reserves 2 GiB for compact output/diagnostics, 1 GiB for
source/runtime scratch and 8 GiB free reserve. It exports no models, optimizer
states or full feature tensors. CPU tests cover forced head application,
duplicate prevention, exact-anchor reconstruction and invalid gain exclusions.

This is a same-response hindsight diagnostic: the correction has already learned
from the answer being replayed. Success demonstrates native usefulness on these
fitted examples, not transfer to subsequent requests. Failure supports deferring
the current learned corrections, but does not prove that every correction within
this space would fail. Surviving BF16 rounding does not itself establish improved
acceptance or fewer verification calls.
