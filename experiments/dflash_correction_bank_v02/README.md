# Frozen DFlash correction bank v0.2

This prototype replaces v0.1's fixed correction space with mutable low-rank
residual operators. It tests whether they fit completed feedback, transfer to
subsequent requests, preserve demonstrated gains, and help serving. A smaller
parameter count is not a claim of lower adaptation cost.

**Status: implementation for review. No v0.2 GPU qualification or campaign has
run.** Historical v0.1 code and evidence remain separate. The source definition
is the user-supplied nine-page *DFlash Correction Bank v0.2 Mathematical Design*
(19 October 2026), SHA256
`189d2ceb7048622ef8a50a309bff8ef256e4fc2e76b53dcc41b71307b441de10`.
The PDF and private datasets/results are not copied into this code contribution.

## Mechanism

Four modules store `W_j = B_j A_j`, rank at most32, with both directions mutable.
The released DFlash backbone, final normalization and vocabulary head stay
fixed. The correction sits after final normalization: `E(h + W h)`. Zero
correction bypasses arithmetic exactly. Nonzero residuals use FP32, then cast
back to the existing BF16 vocabulary boundary. Two modules concatenate their
factors, retaining rank up to64. No dense matrix or SVD is used while serving.

`core.py` implements the normalized response/memory metric and spectral
proximal proposal, independent owner trials, backtracking, and winner-only
commit. `feedback.py` reuses the qualified author sampler and computes full
hidden teacher moments; `state.py` owns immutable protection and current-version
routing evidence. `native_hooks.py` provides the native residual, prompt-only
routing descriptor, and canonical frozen-feature capture. `training.py` joins
these without changing the OSD trainer. See [VALIDATION.md](VALIDATION.md) for checks and remaining GPU gates, and
[DESIGN.md](DESIGN.md) for numerical
and protocol details.

After a response, hash-select up to two nonoverlapping native blocks. Exclude
their positions from **every** overlapping bank training example. Reserve fewer
or none when necessary to leave training positions. Earlier conversation is
conditioning only. OSD retains its original exposure without this exclusion.

Only gains demonstrated by the action actually served **before the update**
can create hard protection. These records are inserted before owner trials.
KL and native-prefix protection have separate eligibility. Source fitting and
post-update counterfactual cache refresh never create protection automatically.

## Controlled execution

The unchanged baseline is **AdaFlash OSD training path under our session
protocol**: full-weight forward KL, AdamW, LR3e-4, two passes and two optimizer
steps per newest response. It is not an exact reproduction of published OSD or
full AdaFlash. All three methods use the existing pinned Llama-3.1-8B-Instruct /
DFlash-UltraChat pair and qualified SGLang runtime, greedy decoding, block10,
1024 output tokens or EOS, full histories, and valid method-specific caches.

The controller first qualifies the first three requests of `wildchat_00` and
deterministic protection/rejection fixtures. Then it restarts cleanly for six
streams:116requests,110scheduled events per learning arm,220OSD steps. A bank
event can commit zero, one or two steps; all owner/backtracking work is charged.
An unfavorable gain does not fail qualification or trigger retuning.

Each learning arm generates its own answers and feedback. Before learning from
a turn, compare frozen, OSD and bank states on the **bank's saved input** in
three fresh, rotating, capture-disabled repeats. Replays do not change history,
learning, the routing cache or cost model. The bank restores its pre-read state
and repeats descriptor extraction, selection and fusion within request timing.
Exact input/output token IDs are checked; mismatches remain in evidence and
invalidate affected gain comparisons.

## Review and launch boundary

CPU tests (with the pinned numerical environment):

```bash
python -m pytest experiments/dflash_correction_bank_v02 -q
```

Prepare a metadata plan without loading a model:

```bash
python -m experiments.dflash_correction_bank_v02.run plan \
  --dataset /absolute/path/to/six_stream_dataset \
  --out /absolute/path/outside/checkout/plan
```

After review, the explicit `run` action additionally requires a pinned tokenizer,
runtime configuration, the reviewed source inventory hash, and
`--allow-gpu-execution`. It performs storage admission before loading workers.
No launch is part of this implementation delivery. Configuration examples retain
the inherited exact model/runtime hashes; paths must refer to qualified local
artifacts. The main campaign additionally binds the qualification cost profile.

## Evidence and costs

Reports retain every timing repeat and per-turn comparison, separately for
WildChat and synthetic PersonaMem. Ratios pool summed measurements. Bank
records include allocation, singleton/mean/sum selection, fallback, predicted
versus realized block gains, source fitting, supervised exposure, every trial,
and reasons for blocked writes. Local block gains are not a percentage of
complete request time saved.

Serving includes prefill, routing/fusion, drafting and verification. Canonical
feature capture is separately measured but already inside its outer request
timer. Paired checkpoint benchmarks have no capture. Teacher preparation,
gradients, spectral work, all owner trials, protection scans, action refresh,
activation and persistence are disclosed. Nested timers explain the outer
adaptation cost and are not summed again. Benchmark-only replays and
qualification are excluded from reconstructed deployment episode cost.

Every turn keeps compact bank/memory snapshots, full-hidden selected records,
immutable certificates, anchors/RNG metadata, token IDs and native traces.
Unselected bulk capture can be retired only after selected records and their
receipt are durable. Rolling OSD restart states remain through paired scoring;
final bulk retirement requires a complete validated episode result. Retirement
manifests explicitly describe lost replay capability; deleted tensors are never
called archived. The campaign stops on an incomplete/invalid stream.
