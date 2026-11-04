# K+ objective CPU qualification

## Material Passport

ARS experiment-agent, run→validate. Status: CPU arithmetic qualified. Scientific base: `4d94f30c42701d0f86ac04e2829c834593c16d8e`; the new execution commit awaits parent publication. Exact implementation hashes and command are in `OBJECTIVE_QUALIFICATION.json`. This review establishes arithmetic/integration fixtures, not GPU qualification or efficacy.

The first focused run passed **21 tests: seven new K+ tests and 14 inherited first-error objective/mask tests**, exit 0, unittest time 0.192 seconds. Only line wrapping in the test file followed that run. No model optimizer steps, GPU work, remote operation, scientific retry or production result was used. Tolerances were saved in `OBJECTIVE_TOLERANCES_PREDECLARED.json` before executing fixtures.

`online_feedback.kplus.install_kplus(model)` attaches the fixed reducer to the existing BF16 `osd_first_error` model and returns the serializable `kplus_contract()`. It changes no model parameters, sampler, optimizer, teacher path or old objective registration. `objective.py` has 23 additive optional-hook lines; ordinary K has no installed reducer and retains its numerical operations. The experiment must install K+ exactly once after native runtime construction and retain the returned contract in its state receipts. Microbatch sizes other than one are rejected.

Each existing projection chunk contributes detached W1/W2/counts and differentiable N1/N2 using the already materialized current proposals and ordinary teacher KL rows. All chunks are pooled before the response correction. The original native N_K and selected-weight sum S are retained; only N_K is changed for a mixed response. No-error, priority1-only, priority2-only and empty responses return the same original N_K tensor, preserving exact native neutral gradients. The original response denominator and its single epsilon remain unchanged.

The independent explicit reference traverses each block, builds K's coefficients and applies `w*(0.5+0.5*v/vbar)` at first errors. Tests verify:

- Equal first-error weights produce priority2 coefficient 7/6, priority1 coefficient 5/6 and moved-weight share 1/12.
- Unequal error masses in separate homogeneous eight-block chunks still produce the correct global mixed response. FP64/FP32 loss and row gradients agree; N1/N2 remain differentiable and W1/W2 detached.
- Native BF16 probability/KL with FP32 weighted reductions agrees with direct response weighting for loss and logit gradients under the declared tolerances. No post-error successor acquires KL gradient.
- Canonical EOS, an EOS seed, the last valid proposal, a validity gap and the final block slot prevent invalid successor access; a correctly predicted EOS immediately after an earlier error remains a valid successor.
- Prefix coefficients and all-correct blocks remain unchanged. Neutral K+ loss and BF16 gradients exactly equal native K, including empty supervision.
- The full wrapper pools once, retains the native denominator, draws no additional RNG, calls the backbone once and uses only the current/teacher vocabulary projections already required for each chunk. Checkpoint replay leaves per-response diagnostics unchanged.

Per-response detached `sum_metrics` retain `kplus_N_K`, `kplus_S`, `kplus_W1/W2`, `kplus_N1/N2`, `kplus_N_new`, both error counts and coefficients, total error weight after redistribution, signed conservation error, moved mass/share, mixed-response flag and response count. Mixed frequency is the sum of mixed flags divided by response count. End-event `kplus_chunk_s` includes actual checkpoint recomputation work; `kplus_response_s` measures scalar pooling/assertion/diagnostic work. Neither timer is another observation.

The production reduction asserts first-error mass conservation at `rtol=2e-6, atol=2e-7`. FP64 fixture tolerance is 1e-12; FP32 loss/gradient and native BF16 loss tolerate `rtol=2e-6, atol=2e-7`. BF16 logit gradients tolerate `rtol=0.02, atol=2e-4`, with relative L2 error at most 0.01. Neutral equality requires zero tolerance. These are fixed numerical tolerances, not efficacy thresholds or evidence about a future trained model.
