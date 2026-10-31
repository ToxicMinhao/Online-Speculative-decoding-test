# K with released-prefix protection v1

This fixed coding experiment asks whether preserving useful released-DFlash
prefixes improves K beyond a single lower-learning-rate control. It uses eight
already completed seed/event feedback pools. No new answers or teacher features
are generated.

- **KP:** unchanged current-prefix-plus-first-error forward KL, plus `0.10` times
  a separately response-normalized hinge on the released F's correct prefix.
  The threshold is `0.50 * min(reference_margin, 1.0)`. Reference and current
  rivals are selected independently. The reference stays released throughout.
- **K_HALF:** unchanged K with constant LR `1.5e-4`; all other settings, including
  weight decay, remain unchanged.
- **F/K:** released and retained original event-4 serving controls. No retraining.

Both new arms begin from the original per-seed scientific step-zero state,
including empty Adam and ready-to-train RNG. Seeds17/29 each run four events,
128 responses, two passes and128 Adam steps per event. Every natural order and
anchor draw is checked against the original receipt; stored anchors never force
agreement. All own event boundaries remain retained.

The added objective lives in `online_feedback/prefix_protection.py`; historical
objectives and O/K validators are unchanged. The new experiment has explicit
contracts and migration rather than pretending KP is an old K checkpoint.

## Qualification and work

One suite uses a metadata-selected early response followed by the longest
prepared response, starting from a disposable copy of populated K17-4:
Q1 ordinary K; Q2 neutral KP with reference work; Q3 actual KP; Q4 a forced-active
plumbing fixture. Cases use fresh sequential processes and isolated cold caches.
Q2 must reproduce decoded state, real preclip gradients, ordinary loss, anchors
and RNG exactly. Q4 must change gradients and Adam moments. It need not change a
BF16 parameter entry. Production rejects its synthetic override.

| Work | Main campaign | Disposable qualification |
|---|---:|---:|
| Current response forward/backward equivalents | 4,096 | 8 |
| Frozen-reference response forwards | 2,048 | 6 |
| Adam steps | 2,048 | 4 |
| Native complete generations | 3,072 | 0 |

There are no added model warmups or automatic retries. A failed numerical gate
stops before production. A valid negative result completes the planned campaign.

## Execution

CPU preparation seals a new128-request panel using seed20261101. Original
feedback and native results are audited read-only. `seal.py` requires a clean
published commit, actual A800 paths and exact preparation/artifact identities.

On A800, after deploying the published source and sealed run:

```bash
bash experiments/k_prefix_protection_v1/run_a800.sh /absolute/approved/run/root
```

The runner admits disk/host/device capacity, executes CPU tests and the sole
qualification, then launches production in fresh processes. Four-step endpoints
are evaluated on the sealed panel with four methods and three balanced repeats.
The included eight-request gates consume192 of the3,072 generations. Every
method must match the first fresh F tokens/termination and consistent counts;
F's count trajectory is also invariant across both seeds.

An explicit `--resume` is only for a fully validated complete scientific
boundary. It preserves completed work, refuses partial/unknown work, validates
chronology and state/export identities before calculating remaining writes, and
uses exclusive new native attempt directories. It never repeats qualification.

## Interpretation and evidence

Counts use one validated trajectory per request; timing uses three repeat means.
Paired bootstraps jointly sample request identities with both seeds/all methods,
10,000 PCG64 draws, seed20261102. Gain/harm accounting is calculated within seed
before pooling. The primary KP/K comparison, KP/F benefit and lower-LR control
remain separate. Secondary intervals are descriptive and unadjusted.

A0.5% call reduction is a research-priority threshold, not deployment break-even.
A result is labeled supported useful improvement only with the handoff's
positive combined intervals and positive seed directions. Small, uncertain,
relative-only, mixed, inactive and invalid findings remain explicit.

Raw private inputs, weights and native artifacts stay outside Git. Each public
report binds the actual execution commit and distinguishes record review from
independent GPU execution.
