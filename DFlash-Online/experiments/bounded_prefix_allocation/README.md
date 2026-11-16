# Bounded prefix allocation development screen

This is the fixed November 16 candidate C on native K+. It redistributes only
the existing correct-prefix coefficient within each block. The first-error
rule, eleven-field K+ reduction, original response denominator, BF16 KL,
optimizer, sampling and attention remain unchanged.

Canonical probabilities are detached gathers from the native log-softmax,
converted to FP32 before exponentiation. The score is the reverse cumulative
sum of products of `0.5 + 0.5*q` over all valid proposal positions. Prefix
coefficients move toward their normalized scores while conserving prefix
mass and bounding each relative change by 0.25. Only the additive correction
`sum((candidate-native)*KL)` is appended to the native response numerator.
This is a constrained allocation heuristic, not a claim of exact optimization
of native accepted length or a reproduction of D-PACE.

## Fixed execution

1. Authenticate complete K6, completed N8 and restarted/completed S8, the exact
   existing inputs, all 128 final canonical outputs and two warmup references.
2. Qualify native/no-op equivalence, an actual active intervention, serialized
   own-state continuation and native output preservation. Five disposable Adam
   applications are planned; the original ceiling is sixteen, including failed
   work. The eight native checks use no new reference generation.
3. Train C7 from complete K6; collect C7's own 128 event-eight responses/features;
   continue complete C7 to C8. Each event has 256 presentations and 128 Adam steps.
4. Evaluate N8/C8/S8 on the unchanged 128-request panel: 72 separate warmups and
   1,152 scored generations, with the inherited four-block rotating order.
   All final timing uses the same physical card with the other card idle.

The production chain is serial. Independent training qualification on A and
native serving checks on B may overlap only after RAM/VRAM admission. No control
is retrained merely to use the second card. All startup and failed work remains
charged. No retry is implicit in the launcher.

## Budgets and state

The shared ledger caps total work at 272 Adam applications, 544 response
forwards/backwards, 128 feedback generations, 128 teacher preparations, eight
qualification generations, 72 warmups and 1,152 scored generations. Disposable
startup is included. Both elapsed time from the first model load and summed
occupied-card time are capped at four hours.

The new allocation cap is 80 GB with a 400 GB shared-volume reserve after peak
writes. Keep C7/C8 complete recovery states and exports, own feedback, source,
configuration, compact traces and all failure receipts. Only verified redundant
new capture/terminal tensors may be removed after their exact rows and token
bindings are preserved in the durable prepared packet. Existing K6/N/S assets
are read-only dependencies.

## Analysis

Primary C/N and C/S comparisons use ratios of summed native accepted proposals,
verification calls and complete-request time. One count trajectory per request;
all three timing repeats stay paired in 10,000 whole-request bootstrap draws
(PCG64 seed 1708, source-ID order, linear percentiles).

The candidate qualifies for a separately designed confirmation only if C/N
time reduction is at least 0.5%, C/N timing and acceptance lower bounds exceed
zero, timing improves in every repeat, and C/S points favor C on both metrics.
Otherwise close this fixed candidate at this development screen. This reused
panel and single training origin do not establish generalization, a causal
allocation explanation, or a new comparison against OSD.

## Entry points

`prepare.py` builds the sealed compact package; `preflight.py --plan PLAN.json`
checks the actual unpacked files and read-only CPU recoveries. After publishing
and pinning the clean checkout, `launch.py --plan PLAN.json` runs qualification
and the fixed campaign. `task_runner.py` uses existing DFlash-Online training,
feedback and native serving components. No scientific parameters are command-line
sweep inputs. Raw private requests and tensors remain outside Git.
