# Continuation-value information diagnostic

This isolated, forward-only diagnostic compares the retained PR59 PAIR1 export
with the native PR60 CONTROL child. It changes no training or serving code.
Both endpoints are native C + K+PAIR; the child differs by the already-completed
128-update event. This is retained training-buffer adaptation, not held-out
transfer or a causal test of allocating additional loss weight.

`prepare` authenticates archived receipts, endpoint weights and prepared features,
then seals at most 64 outcome-independent first-pass anchors for each of all 128
responses. `native` loads exports directly, without a trainer or optimizer, and
uses the original BF16 draft/block/head path in eval/no_grad. `worker` runs exact
legacy parity and repeatability checks before `launch` permits deterministic
response sharding across the two physical cards. All parent/child pairs stay on
the same card. `metrics` retains native predictions/margins and computes the one
specified running-minimum continuation score and frozen-parent outcomes.
`statistics` implements the fixed response-weighted model, whole-response
bootstrap and influence checks. `analyze` writes private reconstructible records
and token-free public tables.

The numerical rank policy and every anchor are sealed before new endpoint
readouts. Qualification uses sixteen response forwards across two cards; the
primary cohort uses at most 256. A single identified implementation retry may add
at most eight forwards inside the unchanged 3,600-card-second and 90-minute caps.
There are zero backward calls, optimizer objects/steps, teacher forwards,
feedback/serving generations or latency benchmarks. No inherited files are
removed. New source/output/cache storage is bounded by 2 GB with a 400 GB server
reserve.

Read the attached handoff and the sealed execution contract for the full
prespecified covariates, numerical support gates and decision rule. Stop at the
report, including a negative, inconclusive or blocked result. A positive adjusted
association would support proposing a separate pilot; it does not establish that
reweighting these repairs causes better learning or serving.

CPU checks use the existing qualified Python environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:training:inference \
python -m unittest discover -s tests -p 'test_continuation_value_*.py' -v
```

Native execution additionally requires the authenticated private paths, approved
A800 environment/compiler and sealed contract. There is deliberately no default
command that silently resolves or substitutes a checkpoint.
