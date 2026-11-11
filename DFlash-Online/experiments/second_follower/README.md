# Second-follower training pilot

One GSM8K KPLUS4→5 event tests a capped SHORT-to-LONG redistribution inside the
native high-priority first-error budget. The inherited K+ denominator, ordinary
BF16 KL, optimizer, two-pass schedule and serving remain unchanged. No follower
position receives new supervision. The reference is authenticated PR39/PR37
native KPLUS5; it is not retrained.

`online_feedback/second_follower.py` is the complete objective addition. It uses
the existing chunk reducer interface without another forward/head/projection.
`prepare.py` authenticates and seals the original inputs and exact256-request
panel. `qualification.py` checks the first8 scheduled parent windows and three
disposable native/disabled/active windows. `deferred.py` handles a first positive
window arising only during production, with a separately charged native twin.
`training.py` retains the native trainer's state/order/anchor checks. `evaluation.py`
is a genuine two-arm PR39-style eight-request-block evaluator: two counterbalanced
sweeps,1,024 completions. `matched_start.py` freshly evaluates both step640 children
on PR40's128 sealed GSM parent starts. `analysis.py` retains paired requests and
prespecified PCG64(20261113),10,000-draw linear-percentile95% intervals.

The controller uses physicalGPU0 only, checks both cards before timed blocks,
closes model processes between phases, and enforces a conservative4,500-second
clock from the first GPU phase. Complete endpoint measurements take priority over
local probes if the resource bound prevents finishing. Correctness failures retain
all completed/invalid/unknown work. No tuning or further event is authorized.

Private responses, teacher features, model tensors and raw runtime evidence stay
outside Git. Public reports identify the actual execution commit and sealed inputs.
