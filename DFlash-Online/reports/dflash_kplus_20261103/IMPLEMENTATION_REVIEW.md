# K+ implementation review — November 3, 2026

This implements the fixed Track 1 handoff from the full-K scientific execution
`0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34` and report/source parent
`4d94f30c42701d0f86ac04e2829c834593c16d8e`.

The algorithm changes only K's first-error numerator weights. Membership uses
current detached argmax predictions and never trains a later matching successor.
Mixed-class statistics are reduced over the whole response; native BF16 KL,
FP32 weights, original denominator and averaging remain unchanged. Neutral
responses return the native numerator directly. The source adds a small
optional hook rather than replacing upstream training or serving components.

Separate same-team implementation and review work covers arithmetic/gradient
fixtures, the continuous native training path, retained-baseline validation,
fixed evaluation scheduling, reporting and physical-storage admission. These
are not an independent GPU reproduction. Exact CPU results and source identity
are bound in the execution seal before production. The local test harness was
made CUDA-hidden before importing Torch: an earlier CPU integration invocation
with GPUs visible initialized CUDA RNG bookkeeping and failed its zero-device
fixture expectation before model training. Its log is preserved privately;
scientific RNG behavior and tolerances were not changed.

Review found and fixed a boundary-file glob that would also have read validation
receipts, missing arguments to the existing initial-state validator, and
incomplete final-evaluation storage-growth checks. Physical allocation and
compiler growth are now recorded. Native captures and terminal rows are deleted
only after exact CPU equality with their durable prepared replacement, and
retirement applies only to new-run tensors or separately authenticated redundant
historical copies. Complete initial/final states and compact evidence remain.

Retained K58 was freshly authenticated. All3,072 old K58 executions agree with
the original1,024 canonical inputs/outputs and repeat counts:43,387calls and
314,847outputtokens. The original initial snapshot was CPU-decoded:all58BF16
model tensors equal released weights, Adam is empty, counters are zero, and
complete RNG states remain. No historical feedback tensors were reconstructed.

The prescribed native correctness checks remain pending until production.
Passing CPU checks does not establish GPU output preservation or a K+ benefit.
The fixed outcome uses one new58-event trajectory and fresh K58/K+58timings;
no performance-based selection, extra event, tuning or warm-up sweep is allowed.
