# Recent-repair retention: one event from KPLUS6

This is the prescribed seed17 GSM8K event7 fork. It tests whether a fixed margin
floor on verified KPLUS5→KPLUS6 repairs improves subsequent native serving. The
ordinary K+ objective, sampler, optimizer and inference path remain unchanged.

Three independent arms restore the complete KPLUS6 recovery at Adam step768 and
finish896. NATIVE uses ordinary K+. RECENT adds a reachable-slot margin hinge at
coefficient0.10; its frozen floor is0.50×min(parent margin,1). CONTROL moves the
same assigned positional weight to matched established successes, retaining
unmatched recent tags. Matching is sealed before training and never uses event7
outcomes. All probability/head/parameter arithmetic retains the qualified BF16
path; margin subtraction and hinge reduction are FP32.

The private run package binds the original128 responses, teacher packets,
two-pass order and sampled anchors. The training sequence is not reconstructed
from text. Reference preparation checks frozen predictions in the actual
eight-block head layout. A three-case disposable native step compares ordinary,
disabled and zero-coefficient execution; separate active/long-response checks
add no Adam step. The first naturally firing window in each active arm has one
bounded active/off replay, followed by exact final-state restoration.

The held-out input-only panel contains at most256 GSM8K test requests; all source
IDs, exclusions and prior exposure flags are sealed. Two quiet counterbalanced
serving sweeps give at most1536 complete responses. A separate64-request natural
KPLUS6 walk and fixed-start replay measures relapse and displaced repairs.
These local outcomes are not converted to seconds saved.

Primary acceptance is **clipped accepted draft proposals / verification calls**.
Ratios pool counts, timing averages the two repeats per request, and uncertainty
uses10000 paired whole-request bootstrap draws (PCG64 seed1707). Advancing the
candidate requires the joint screen in `analysis.py`; valid negative/inactive
results still complete the experiment.

## Execution

Use an authenticated clean source and the actual-path private package generated
by `prepare_run.py`. On A800, after storage/quiet-host admission:

```bash
bash experiments/recent_repair_retention/run_all.sh /data/sources/minhao/runs/RETENTION_EXECUTION /data/sources/minhao/RETENTION_SOURCE/DFlash-Online
```

The supervisor runs reference preparation, immutable placement sealing, bounded
qualification, the three scientific arms, separate canonical preparation,
fixed-start diagnostics, timed serving and raw audit. It stops on correctness,
nonfinite or resource failures and never retunes or automatically retries.
Private prompts, tokens, optimizer states and model tensors stay outside Git.

Core scientific budget:384 Adam steps/768 response presentations,64 parent walks
and fixed-start replay,1536 timed completions at full panel size. Qualification,
startup, frozen reference forwards, missing canonical preparation and first-fire
counterfactual replays are counted separately. Runs are serial on physicalGPU0;
both cards must be quiet for timing. Operational timeout6hours; expected GPU
execution2–3hours based on retained PR39/42/43 costs, not a performance promise.

## Authorized evaluation-only recovery

The first execution completed training but exhausted a miscounted checkpoint-load
allowance before any diagnostic parent generation. The original failure and its
startup work remain preserved. The user authorized a separate evaluation-only
continuation using the same three exports, sealed inputs and completed qualification:

```bash
python -m experiments.recent_repair_retention.resume_evaluation run --plan /data/sources/minhao/runs/RETENTION_CONTINUATION/CONTINUATION_PLAN.json
```

The continuation authenticates its original evidence and performs no training or
canonical-reference generation. The plan records both execution commits and
charges the failed startup against the original time allowance. This command is
not an automatic retry policy. See the published report for actual run identities,
complete work accounting and the final result.
