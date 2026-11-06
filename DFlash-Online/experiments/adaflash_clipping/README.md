# AdaFlash clipping-location diagnostic

This is a bounded sensitivity test, not another online training event. It uses
the first six event-1 presentations from the completed A58 run as three fixed
pairs. Each pair starts independently from the same retained released weights,
fresh Adam state and RNG in both branches. Exactly six disposable native AdamW
steps are permitted; no generated answers, serving exports or continuing states
are produced.

The paper branch retains the original A58 entry-clipped loss. The comparison
branch uses the official repository's whole-position cap after division by the
complete response denominator. Both use our existing BF16 projection,
log-softmax/exp, teacher-argmax CE, mixture, positional weights, response mean and
optimizer. Position-cap decisions/rescaling use FP32 before returning to the
common BF16 mixture. This tests the clipping rule under common arithmetic, not
the author's different probability/CE kernels. The source is
[AdaFlash rkl_model.py at 478bf0f](https://github.com/ZinYY/AdaFlash/blob/478bf0ff7a3a5232b13d0f14e752d5fe1da0b718/pipeline/training/rkl_model.py#L168-L207).

`prepare.py` seals the six retained presentations and reconstructs frozen target
features by replaying the original native verification inputs. Every saved
posterior and commit is checked. Original feature tensors were retired, so
bytewise historical tensor equivalence is not claimed. Anchors are explicitly
bound to retained records, never resampled.

`run.py` reuses the qualified SpecForge trainer, accumulation and BF16 AdamW.
A disposable two-forward/backward startup has zero optimizer steps. Each of
the six branches gets a fresh runtime, avoiding inherited DDP bucket state.
Pre-update predictions come from the training forwards; two post-step forwards
per branch measure source-fit prefixes under the same anchors. Full gradients
and realized displacements are compared in host memory; only compact statistics
and token-level evidence are retained outside Git.

Work: 12 training forwards/backwards, 12 post-step forwards, two separately
accounted startup forwards/backwards, six Adam steps, six native target feature
reconstructions and six terminal preparations. Projection chunks and checkpoint
recomputation are counted separately. A source manifest binds every deployed
tracked code file to the execution commit. No automatic retry is permitted.

Run in the pinned training environment after sourcing the server helper:

```bash
python -m experiments.adaflash_clipping \
  --source /path/to/original/production_001 \
  --output /path/to/new/diagnostic_001 \
  --execution-commit COMMIT \
  --source-manifest /path/to/SOURCE.json
python -m experiments.adaflash_clipping.report \
  /path/to/new/diagnostic_001 /path/to/compact-report
```

The results establish local loss/update/prefix sensitivity. They cannot determine
which clipping rule gives better held-out drafting after 58 events or explain
the old endpoint difference. Same-response prefixes are overlapping sampled
occurrences, not independent tokens or native verification calls.
