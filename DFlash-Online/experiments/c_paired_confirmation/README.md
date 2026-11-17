# Paired native / bounded-allocation continuation

This campaign asks whether the unchanged C allocation recurs as a useful addition
to native K+ under a new continuation realization. It does not introduce a new
loss, inference method or coefficient.

Both arms restore the same complete origin per dataset: GSM8K KPLUS6 or MathQA
KPLUS4. After exact restoration and disposable startup, private generators
replace only the copied Torch CPU and assigned logical-CUDA RNG fields with
seed-29 states. The historical initialization remains seed 17. The fork happens
once, immediately before natural sampling. Adam state, horizon 7424 and LR
0.000300 remain unchanged. The second event restores its own first-event state
and uses its own newly collected feedback; it does not reseed or force anchors.

The original `online_feedback/objective.py` and `prefix_allocation.py` remain
byte-identical to the qualified source. N has no allocation controller; C uses
smoothing 0.5 and relative cap 0.25. The qualification-only cap-zero arm checks
neutrality, not a third production arm.

## Execution

`prepare.py` packages the private, output-independent input seals and dependency
identities. `preflight.py` authenticates the assembled source, origins, exports,
feature pools, runtime and complete storage schedule on CPU. `launch.py` then
runs the bounded qualification followed by the declared production tasks.

```text
python -m experiments.c_paired_confirmation.launch --plan <sealed/PLAN.json>
```

`training.py` extends the existing native event trainer. `qualification.py`
checks original restoration, wrapper/cap-zero equivalence, active C, natural
RNG consumption and next-update replay. `task_runner.py` keeps each arm's
feedback and consumers separate. `evaluation.py` enforces the two-arm schedule
and exact canonical agreement. `analysis.py` pools measurements and performs
the prespecified whole-request paired bootstrap.

All eight production recoveries and four unique prepared feedback banks remain
available. `export.py` qualifies exact CPU reconstruction; `retention.py` may
then remove explicitly enumerated disposable payloads or derived exports under
the sealed schedule. Historical assets are never retirement candidates.

## Fixed work and limits

Production: eight events, 1,024 Adam applications, 2,048 response presentations,
512 own-feedback generations and teacher preparations, 516 origin canonical
generations, 3,072 scored generations and 96 warmups. The six-case qualification
per dataset adds 12 Adam applications; with ten total initializations and their
two-response startup, expected response forwards/backwards are 2,092 each.
Sixteen serving qualification generations are separate from final scores.

All attempted work stays in one ledger: at most 1,056 Adam applications, 2,144
response forwards/backwards each, 16 initializations, four elapsed hours and five
occupied card-hours. The full peak storage plan must fit 125 GB of new live
allocation while preserving 400 GB of shared-volume free space. Failed work and
infrastructure recovery do not reset these limits.

The prospective duration forecast uses observed whole-task costs, a 30% margin,
600 additional seconds, and a separately declared 300-second CPU publication /
reconstruction allowance. It is a planning estimate, not a guaranteed upper
bound. Qualification measurements replace weaker historical cost estimates before
production. The hard limits remain unchanged. The older 1.5x + 600-second forecast
is retained in preparation evidence; it was not a handoff requirement.

## Interpretation

Each dataset has 256 scored inputs plus two separate warmups. GSM8K inputs are
new to the bounded-allocation C family, with older project exposure disclosed;
MathQA is unexposed within the enumerated project inventory. Neither scope is
pretraining freshness. All final timings use one physical card after both
datasets finish training and preparation.

C/N acceptance and serving-time intervals are reported separately per dataset,
using 10,000 paired request draws, PCG64 seeds 1729 and 1730, lexicographic source
order and linear percentiles. Interval signs yield `SUPPORTED_GAIN`,
`SUPPORTED_LOSS` or `DIRECTION_UNRESOLVED`. There is no minimum-gain promotion
gate. A zero-crossing interval does not prove preservation or equivalence.
Historical PR50/52 results remain separate, and native K+ remains primary.

Private prompts, token arrays and tensor artifacts stay outside Git. The final
report records the exact executed commit and all qualification/failure evidence.
