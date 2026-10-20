# Bounded failed-G gradient diagnostic

Starts from PR8 commit `4a2e533`. The existing campaign and full qualification
remain paused. This entry point never generates answers or exports a trained
checkpoint. It reuses the retained G first response and released weights plus
exact initial optimizer/global/anchor RNG state.

The original ordinary two-step warm-up, restoration, and disabled-filter trial
are preserved. Every gradient-sum check records all parameter summaries and at
most32 failing coordinates before raising its original exception. BF16, groups,
full denominator, sampler, clipping/Adam/LR, tolerances and donated_buffer=False
remain unchanged. No historical per-pass anchor list survived the failure;
anchor reconstruction uses retained RNG, with hashes checked across variants.

If the gate reproduces, run only repeated full backward on one graph,
full/A/B versus the original A/B/full order, then one equivalent FlexAttention
eager-backend case if the backend explanation remains unresolved. Each variant
restores the failed pass's weights, optimizer and stochastic state. Eager keeps
the original BlockMask, GQA, BF16 tensors and exact anchors; its dense backend may
be unavailable from memory demand. No SDPA, debug flag or precision substitution.

`python -m experiments.dflash_osd_agreement_filter.gradient_diagnostic` requires
explicit retained config/state/response paths and a fresh diagnostic directory.
Use the same approved native environment and distributed-worker environment as
the original trainer. It is intentionally separate from the campaign CLI.
Private inputs, raw diagnostic records and model-sized tensors stay outside Git.
The report and compact aggregate results will be added after execution.
