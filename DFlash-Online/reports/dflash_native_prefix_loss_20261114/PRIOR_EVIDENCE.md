# Native prefix-loss screen: prior-evidence and scope review

ARS-Codex 3.24.2 experiment-agent. Local read-only source review, plus separately scoped analysis-code authoring. No model or remote operation. Same-family agent review with implementation-author overlap; not an independent GPU replication.

## Why this question is not already answered

| Prior evidence inspected | Actual intervention and evidence | What the new fixed-state screen adds |
|---|---|---|
| PR21 `experiments/k_prefix_protection_v1/results_20261031/REPORT.md`, “What was tested” and final table | Adds a released-prefix margin hinge to native K. KP/K call reduction was 0.3435%, below its practical threshold; K_HALF was stronger. | E removes the current prefix-gradient contribution rather than adding an external preservation term. No universal smaller-update or retention conclusion follows from PR21. |
| PR32 `reports/dflash_role_loss_pilot_20261107/REPORT.md:18,64` | ROLE keeps prefix forward KL and changes error loss; MIX changes both roles. MIX/ROLE is a conditional comparison of prefix recipes, with inconsistent dataset outcomes. Complete trajectories and their states diverge. | Native error KL is unchanged; identical full prestate, anchors, detached roles, denominator and priorities isolate the total one-step effect of withholding the current prefix term through Adam. We must not claim prefix learning was never changed before. |
| PR33 `reports/dflash_functional_step_geometry_20261108/PILOT_REPORT.md:5` and diagnostic table | All 512 FSG updates applied, protected KL decreased, serving regressed. Role-specific scores describe geometry interventions. | The proposed branches do not apply geometry or infer role-gradient causes from observed loss values. |
| PR35 `reports/dflash_kplus_learning_audit_v1/REPORT.md:22,53–55` | Establishes exact selected-weight denominator, K+ aggregate error correction, and limits of scalar-mass arguments. Algebraically equivalent graphs can change BF16 backward. | Gate row gradients before both native selected numerator and K+ group numerators. Native N remains the original full backward, not a component-gradient sum. |
| PR46 `reports/dflash_threatened_repair_replay_20261113/REPORT.md:10,31,42,81` | Strong selected-memory rescue (48/54) but no demonstrated independent benefit. T−N = −51/4096, with interval containing zero. | Investigates an existing native term instead of adding another replay/protection module. It does not reopen PR46 or substitute a posthoc acceptance gate. |

This question was feasible earlier. The oversight was interpreting “correct-prefix loss” as preservation without isolating its contribution. Recent evidence changes priority, not past feasibility. The reviewer-supplied split-request gate analysis is background and has not been independently recomputed in this task.

## What the screen can decide

Primary E−N uses accepted proposal counts on the unchanged 64-request/four-start/16-window development panel, with 10,000 paired whole-request bootstrap resamples. Every fixed window remains included. A positive mean and positive lower 95% bound, with complete validity, supports a next **design** step only. C and H remain explanatory controls. H is an Adam update with zero current gradients, not P.

Source acquisition and loss are measured separately from independent acquisition and loss. Blocker repairs count blocks; acceptance gains count proposal positions. Correctly matched EOS and censoring follow the inherited native scorer. Source improvement cannot substitute for transfer. An interval containing zero does not establish equivalence. All contrasts except E−N are descriptive and have no multiplicity-adjusted or alternative-winner interpretation.

Withholding a current loss role changes norm, clipping and interaction with stored moments/decay. This screen intentionally does not establish a pure directional cause or a scale-controlled upgrade. No norm-search machinery, extra LR arm, DKD training, replay module, long rollout or full-response generation is added. DKD is a reserved analytical identity, not a transplanted mechanism in this implementation.

## Analysis implementation

`experiments/native_prefix_loss/analysis.py` retains raw scalar rows before checking validity; invalid, unknown and unattempted counts remain explicit. A complete estimate requires all five states, all sealed windows/starts and the fitting comparison. Reductions average starts within request/window, windows within request, then requests equally; only requests are bootstrap units. Main counts are not repeated observations and local one-step effects are not summed into an alternative trajectory.

Eighteen focused CPU tests currently pass, covering exact denominator hierarchy, cluster bootstrap, H/P distinction, acquisition/loss decomposition, blocker denominators, zero-effect windows, invalid/missing/duplicate rows, aliases, source occurrences, decision gates and evidence preservation. This is CPU analysis correctness only; native implementation and GPU controls remain the executor's separate gates.
