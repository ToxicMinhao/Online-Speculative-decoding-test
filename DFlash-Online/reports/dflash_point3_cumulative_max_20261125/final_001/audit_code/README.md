# Independent CPU serving audit

Actual experiment execution: `6f7f3a5d1769a630e4ec852205eb06ca55047341`. These later audit utilities are separate from the experimental runtime. They import no production experiment modules and execute no model forwards, training or generation.

To reproduce the published confidence intervals and all per-request timing/count reductions using public tables only, from the report directory run:

```bash
python audit_code/verify_public_serving_tables.py .
```

This requires NumPy and checks 3,492 scalar fields. The public audit JSON includes the exact ordinal bootstrap ordering, seed, generator and draw digest.

To run the eight focused synthetic CPU tests:

```bash
cd audit_code
python -m unittest -v test_independent_serving_audit
```

The complete private-record audit, for an authorized holder of the retained evidence, is:

```bash
python audit_code/independent_serving_audit.py \
  --evaluation PRIVATE_EVALUATION_DIRECTORY \
  --identity-receipt PRIVATE_CPU_POSTFLIGHT.json \
  --output PRIVATE_AUDIT.json
```

The identity receipt was produced by an independently implemented CPU safetensors header/raw-byte parser. It authenticates stored endpoint tensor contents and actual execution identity, without framework loading or GPU work. Full training-state, canonical-token and resource verification are covered by the companion evidence audit. Without this receipt or `--exports`, the serving auditor explicitly labels its model-identity conclusion conditional. `--exports` optionally reads local model files using CPU PyTorch/safetensors; it does not require a GPU.

`publish_serving_audit.py` creates all ordinal-only public tables from private records and a passed independent audit. Actual prompts, IDs, tokens, anchors, model tensors, recovery states and private paths are not included here.

All comparisons preserve the experiment's exact pooling and selection rules. The `1e-10` absolute / `1e-12` relative allowance applies only to independent floating-reduction agreement; eligibility uses unrounded measurements without any new outcome tolerance.
