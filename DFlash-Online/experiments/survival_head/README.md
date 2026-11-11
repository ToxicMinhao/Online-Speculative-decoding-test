# K+ survival-head pilot

One GSM8K event compares ordinary KPLUS, a LOCAL token-correctness head and a
SURVIVAL prefix-correctness head. Each arm restores the original complete PR32
KPLUS4 boundary at Adam step512 and ends at640. The auxiliary coefficient is1;
all ordinary K+ settings, feedback, native anchors and serving stay fixed.

The head shares the current clean forward. Labels are detached; BCE is reduced
in FP32 once per response over all native valid proposal occurrences. Head
parameters are outside the native draft/Adam owner, use a separately seeded
identical initialization and a separately clipped BF16 optimizer. Serving exports
contain only the five-layer draft. The core lives in
`online_feedback/survival_head.py` and `survival_head_runtime.py`; the optional
hooks in `objective.py` leave the disabled path unchanged.

## Source and adaptation

Borrowed source: NAVER's [VAT](https://github.com/naver-ai/VAT/blob/124ad4d05dcf8336d91594122ed3f428bb931223/specforge/core/vat.py),
commit `124ad4d05dcf8336d91594122ed3f428bb931223`, especially `AcceptHead` and the
undecayed prefix-label BCE. The official implementation also changes CE/KD,
temperature, adaptive position weights and optional self-coverage. This pilot
adapts only the scalar-head idea; it retains K+ and adds LOCAL as the matched
head control. Our canonical-token labels, native EOS mask, FP32 response-wide
BCE, independent optimizer/clipping, zero bias and independent initialization
follow the user handoff rather than the complete VAT recipe. The user explicitly
granted a scoped author-reproduction exception. This is not a VAT reproduction.

## Exact work and scheduling

- Main: three independently restored arms,384 backbone Adam steps,
  256 head Adam steps,768 response forward/backward pairs.
- Qualification: four disposable one-step cases,4 backbone/2 head steps,
  8 case forwards/backwards plus4 diagnostic auxiliary backwards; discarded
  startup8 forwards/backwards and0 steps;6 native export checks.
- Main startup:6 separately charged forwards/backwards,0 steps.
- Native endpoint comparison:128 prespecified development requests,3arms,
  2counterbalanced sweeps,768 completions. No generation warmups.
- Reconstruct missing teacher packets from saved native verification inputs,
  never regenerated answers. First validate exact tensors against two retained
  packets. All128 target replays (including those two checks), verifier forwards
  and terminal-row preparation are separately sealed and counted.

PR36 has priority. `queued_launch` waits for its successful completion without
using a GPU; fresh capacity, source, origin and quiet-host checks then gate
`run_all.sh`. Training binds physicalGPU1, serving/teacher reconstruction binds
physicalGPU0. Only one physical card is used at a time. Every timing block checks
both cards and transfers. Separate outputs preserve PR36 and its evidence.

The report uses summed paired calls and request-mean latency, with paired
request bootstrap intervals. First-round comparisons share their complete start
prefix; later differently positioned rounds are not paired. A successful CPU
check is not a native qualification or an efficacy result. Null/harmful valid
outcomes finish unchanged; correctness/resource failures retain evidence and
stop without retry. No automatic confirmation, tuning or later event follows.
