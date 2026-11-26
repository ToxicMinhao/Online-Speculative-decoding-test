# Retained-evidence audit

Run with Python 3.10+ and the standard library only:

```bash
python audit_mechanism.py \
  --run-root /path/to/point3/final_001 \
  --historical-c-status /path/to/pr60/production_001/raw/arms/CONTROL/STATUS.json
```

The run root must contain the authenticated compact `raw/qualification/native` and `raw/arms/F_C_SUM`/`F_C_MAX` trees. The script writes sanitized tables to `public_mechanism/` under that run root and a private source-identity manifest beside it. It never imports the project models or launches a GPU, forward, optimizer or generation. Do not publish the private manifest or original raw witness files: they contain paths, original request IDs, anchors and tokens.

The audit independently reconstructs discrete metrics from retained proposals, cumulative-maximum branches from retained margins, FP64 reductions from represented FP32 terms, per-window work and nested timing summaries. Its formula check uses relative tolerance `2e-6` and absolute tolerance `2e-7` for independent Python FP64 versus saved native FP32 arithmetic; this is an offline consistency check, not a revised native qualification tolerance. Pool-mass receipt comparison uses `2e-5` relative/absolute tolerance to accommodate represented sums; original production validation remains authoritative.

It cannot independently rebuild unsaved full logits, strongest-vocabulary-rival selection, gradient vectors or model execution. `MECHANISM_AUDIT.json` states these limitations. Numeric tables are regenerated; `MECHANISM_REVIEW.md` is the separately reviewed interpretation. Native scientific execution remains commit `6f7f3a5d1769a630e4ec852205eb06ca55047341`.
