# Independent continuation metric audit

Verification status: ANALYZED. Independent CPU reconstruction of retained compact evidence; no new model execution.

Execution commit: `b6f2a2c8d487bcc2d32a5e716b870c10eb5067b3`.

Status: **passed**; 3,369,528 scalar/binding checks, 0 discrepancies.

| Cohort | Responses | Blocks | Origin errors | HIGH | Useful repairs | Lost old-prefix tokens | Signed fixed-K progress |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary_all | 128 | 8192 | 6329 | 2114 | 4861 | 1464 | 8698 |
| primary_HIGH | 128 | 2114 | 2114 | 2114 | 1741 | 290 | 6752 |
| legacy_new_buffer | 8 | 64 | 44 | 15 | 33 | 3 | 69 |
| legacy_parent_training | 8 | 64 | 31 | 8 | 9 | 32 | -11 |

Native FP32 supports were retained for the exact FP64 area, D and outcome checks. The independent rounded-libm tanh check is reported separately:

- `native_fp32_tanh`: 241,152 values; 24,207 differ from the independent rounded reference; maximum 2 ULP, absolute error 5.960464478e-08.
- `offline_cpu_fp32_tanh_of_retained_native_margins`: 2,664 values; 0 differ from the independent rounded reference; maximum 0 ULP, absolute error 0.

- The independent tanh reference uses FP64 libm rounded to FP32; its prospective allowance is four FP32 ULPs. Saved native supports, not recomputed supports, determine audited scores.
- This arithmetic audit allowance does not relax native legacy parity or change model precision.
- Full vocabulary logits and the two gathered logit values were not saved; strongest-rival optimality and native subtraction cannot be independently recomputed from compact margins alone.
- Rival exclusion, native sign/tie consistency and retained margins are checked; native source/qualification evidence covers the unsaved head operation.
- Legacy next-mismatch margins beyond origin K are unavailable. Their contribution to the area is exactly zero; missing margins are not fabricated.
- These are fixed-block training-buffer outcomes, not native acceptance, verification calls, latency or causal weight-allocation gains.

The eight-response new-buffer witness overlaps the primary cohort. The eight-response parent-training witness is a separate small retention description. Neither is pooled into additional inferential responses.
