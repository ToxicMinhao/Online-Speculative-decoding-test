# K/K+ learning audit and next-upgrade decision

**Decision — NO SUPPORTED CANDIDATE FROM AVAILABLE EVIDENCE.**

This completed CPU audit explains the implemented learning changes and their measured scale. It identifies no supported harmful limitation that selects a new algorithm. K+ remains the working method; K remains its essential incremental reference. O denotes this project's ordinary full-position forward-KL baseline on DFlash, not every OSD implementation. No follow-up experiment is authorized by this report.

**Material Passport — ANALYZED.** ARS-Codex 3.24.2, experiment-agent validation; retained evidence and source analysis, not a native reproduction.

**Scope and provenance.** CPU analysis execution commit: `7c458eb3279096d4db0c5d7fba76854a1d3bd2ca`. Reference snapshot: `d47ab1ce644c0702ade657e857f7f00af4185fbb`. Historical execution identities and configurations were checked separately; PR23 K, PR25 K+ and PR32 KPLUS used `0698eeb2…`, `e0a9d3c…` and `cdb6405…`, respectively. Their activated learning paths match the account below. Later optional extensions were not assumed active. The original retained O58 training commit is not identified by the consumed PR23 provenance; its configuration/manifest identities are retained, and PR23's execution commit is not assigned to O58. Full identities, commands, missing fields and elapsed time are in [PROVENANCE.json](PROVENANCE.json).

## 1. EXACT CODE/ALGEBRA: what actually learns

These runs use Qwen3-8B/DFlash-b16, a known seed plus 15 proposals, forward KL at temperature one, two shuffled passes per 128-response event, microbatch one and accumulation two. Training uses up to 512 anchors/response, decay parameter seven, Flex attention, BF16 parameters/probabilities, constant learning rate `3e-4`, AdamW weight decay `.01`, betas `(.9,.999)`, optimizer epsilon `1e-8`, and global clipping at one. Optimizer state persists between events. K_HALF is separately labeled at `1.5e-4`. Serving remains greedy, concurrency one, 2,048 tokens or EOS and context 4,096; this audit executes none of it.

| Symbol/operation | Actual source and behavior |
|---|---|
| Response r; occurrence i=(anchor,position) | [training.py](../../online_feedback/training.py), [dflash_family_model.py](../../training/specforge/algorithms/common/dflash_family_model.py): prompt loss masked; anchor and next token must be supervised. Random ranks select anchors without replacement, then sort them. Each training block has a clean known seed and 15 mask embeddings. Overlapping blocks can supervise the same canonical token. |
| Label y, weight w | [objective.py](../../online_feedback/objective.py), `_block_labels_and_weights`: in-bounds response positions, seed excluded, weight exp(−(position−1)/7). Teacher row immediately preceding the label supplies its conditioned prediction. |
| Current selection S | [first_error.py](../../online_feedback/first_error.py#L12): detached current argmax; retain consecutive correct proposals and the first mismatch, including that error. Canonical EOS is included; later positions, padding and validity gaps end eligibility. |
| Full block computation | [dflash_mask.py](../../training/specforge/modeling/draft/dflash_mask.py): context before the anchor and same-block attention remain unchanged. Selection removes direct loss, not attention positions. |
| KL and response reduction | [objective.py](../../online_feedback/objective.py), [chunking.py](../../training/specforge/core/chunking.py): teacher and draft logits use the same frozen target vocabulary head; eight-block head chunks, full-block attention, sum all response terms before normalization. |
| First-error priorities | [kplus.py](../../online_feedback/kplus.py#L36): priority two only if the immediately following valid proposal already matches its canonical label; otherwise priority one. Response-wide redistribution follows chunk summation. |
| Two-response update | [ReductionAdapter](../../online_feedback/training.py), [TrainerCore](../../training/specforge/training/controller.py), [backend](../../training/specforge/training/backend.py), [optimizer](../../online_feedback/optimizer.py#L16): average separately normalized response losses, accumulate gradients, clip once, apply native BF16 AdamW, retain moments/counters. No FP32 master parameters. |

Let A contain positive original weights, S the selected occurrences, p the frozen teacher distribution, q the current draft distribution, and ell=KL(p||q). At fixed weights, anchors and detached selection, define D=Σ_A w+ε and Z=Σ_S w+ε, with ε=10⁻⁶ once per response:

\[
L_{O,r}=\frac{\sum_{A_r}w_i\ell_i}{D_r},\qquad
L_{support,r}=\frac{\sum_{S_r}w_i\ell_i}{D_r},\qquad
L_{K,r}=\frac{\sum_{S_r}w_i\ell_i}{Z_r}
       =\frac{D_r}{Z_r}L_{support,r}.
\]

**K makes two changes:** removes direct supervision after the first error, then amplifies retained terms relative to the original denominator. `L_support` is an analytical counterfactual, not a new arm. Its ideal gradient has the same scaling identity. Averaging two responses with different D/Z factors can change their combined gradient direction; this is not necessarily one global learning-rate multiplier. Variation alone does not show harm.

K+ separately redistributes first-error weight. Let W1/W2 be original error weights and N1/N2 their weighted KL numerators. For mixed groups:

\[
N_+=N_K+\frac{0.5(W_1N_2-W_2N_1)}{W_1+2W_2},\qquad
c_1=1-\frac{0.5W_2}{W_1+2W_2},\qquad
c_2=1+\frac{0.5W_1}{W_1+2W_2}.
\]

Prefix weights and Z are unchanged; c1W1+c2W2=W1+W2. If either group is absent, native code returns the original numerator directly, rather than adding an algebraic zero. Priority two measures one immediate successor, not a later run length or a simulated successful repair.

With normalized p, current logits z_i, ideal effective weights w̃ and model Jacobians J_i=∂z_i/∂θ:

\[
\frac{\partial L_+}{\partial z_i}=\frac{\widetilde w_i}{Z}(q_i-p_i),\qquad
\nabla_\theta L_+=\sum_i\frac{\widetilde w_i}{Z}J_i^\top(q_i-p_i).
\]

Conserving scalar weight therefore guarantees neither gradient-direction preservation, gradient-norm preservation nor unchanged Adam displacement. For an arbitrary fixed rounded teacher vector, replace q−p by qΣ_v p_v−p. Parallel proposals share parameters and attention dependencies. Zero direct suffix loss does not freeze suffix predictions. Current training selection is not necessarily the historical serving rejection index; neither K nor K+ guarantees preservation of old correct predictions.

**Numerical boundary:** projection, probabilities, KL entries/vocabulary sums and parts of backward are BF16; scalar weights and chunk/response reduction are FP32. The single FP32 epsilon addition can round away at larger sums. K+ corrects aggregate numerators; an algebraically equivalent per-position graph need not reproduce its BF16 backward. The equations and tiny [algebra checks](check_algebra.py) are not native replay. Clipping and persistent Adam do not generally cancel response-dependent normalization.

## 2. RECORDED MEASUREMENT: exposure and effective weights

The manifest-led local inventory was performed once. PR25 supplies **14,848 original response presentations / 7,424 windows**; PR32 supplies **1,024 / 512 per KPLUS dataset**, kept separate. All 132 consumed original JSONLs match retained identities. All 5,524 comparable public scalar totals and every adjacent response/window pairing reconcile. Receipt copies and checkpoint recomputation are not extra observations.

PR23 originals are unavailable at the 348 manifest-led local candidate paths across K/K_HALF; its public event aggregates remain usable. This does not mean those records never existed or were deleted. No recovery search occurred. PR25's 116 public rows are pass aggregates; PR23's 116 rows are event aggregates across two arms. Repeated presentations are not independent experimental samples.

| Population | Selected / available weight, pooled | Available / selected weight, pooled | Mean ideal response D/Z | Actual equal-response loss |
|---|---:|---:|---:|---:|
| PR23 K | 74.2623% | 1.346578 | Unavailable | Unavailable |
| PR23 K_HALF | 72.5363% | 1.378620 | Unavailable | Unavailable |
| PR25 K+ | 74.3769% | 1.344503 | 1.351367 | 0.182106 |
| PR32 KPLUS GSM8K | 71.3765% | 1.401021 | 1.414142 | 0.228174 |
| PR32 KPLUS MathQA | 74.3561% | 1.344880 | 1.352779 | 0.176025 |

PR25 selected 33,411,824 of 54,368,391 available **occurrences**, including overlaps. PR23 histograms yield ideal decay-weighted first-error shares of 7.4165% (K) and 8.0515% (K_HALF); these use recorded position counts and gamma seven, not native FP32 error-mass logs. D/Z p05/median/p95/max are 1.153277/1.329082/1.624732/2.168725. Within-window max/min factor ratios average 1.109900, p95 1.291535. These are descriptive ideal ratios reconstructed in FP64 from logged FP32 sums, not bitwise native gradient multipliers. PR32 separately records native response-ratio means; the CSV distinguishes them from pooled ratios and ideal reconstructions.

| Population | Original error / selected mass | Moved / original error mass | Moved / selected mass | Mean logged c1 / c2 |
|---|---:|---:|---:|---:|
| PR25 K+ | 7.3758% | 7.9211% | 0.5842% | 0.881878 / 1.263756 |
| PR32 GSM8K | 8.5007% | 8.0288% | 0.6825% | 0.878414 / 1.256828 |
| PR32 MathQA | 7.3370% | 8.1749% | 0.5998% | 0.873067 / 1.246135 |

Moved mass is one-way transfer, half the total absolute weight change. Every observed response is mixed priority; empty available/selected/error mass and single-priority response counts are zero in these observed populations. Mean logged response moved-shares are 7.8831%, 8.0266% and 8.1762%, respectively—not the pooled fractions above. Coefficient means use actual response counts, not pooled W1/W2 substituted into a nonlinear formula.

PR25's pooled numerator/denominator is **0.176840**, different from its actual response-mean loss **0.182106**. Pre-clip norms exceed one in **3/7,424 windows**; PR32 has **0/512** in each dataset. This describes clipping, not gradient conflict or optimizer invariance. PR25 event-one→58 selected-weight fractions are 70.5595%→73.9638%; response losses 0.242494→0.171975. All pass/event trends remain in [AUDIT_METRICS.csv](AUDIT_METRICS.csv); changing feedback/model states prevent causal normalization inference.

## 3. RECORDED MEASUREMENT → INTERPRETATION: prior controls

Source intervals below are historical 95% request-level intervals, not independent-seed guarantees. They are not pooled across studies.

| Source observation | Supported inference | Competing explanation / design implication |
|---|---|---|
| [PR23](../dflash_full_k_gsm8k_20261031/REPORT.md): K/O call reduction 3.0025% [2.6667,3.3457]; request-time reduction 2.1886%. K_HALF is worse than K. | Full K recipe has stronger evidence than K+'s increment. | Support, normalization and own-feedback trajectories change together; smaller updates are not universally preferable. |
| [PR25](../dflash_kplus_20261103/REPORT.md): K+/K saves 120 calls, 0.2766% [0.0045,0.5457]; request time −0.8230%. | Small positive increment for this trained seed and reused evaluation panel. | Gross saved/added 792/672; own-feedback differences and cache costs remain. Native-generation interval includes zero. Small scalar mass is not a speedup ceiling or reason to amplify it. |
| [PR26](../dflash_mathqa_transfer_20261104/REPORT.md): K+/K adds 15 calls; time reduction −0.1003% [−0.4399,0.2427]. | Increment did not replicate on frozen MathQA transfer. | Near tie does not prove equivalence or identify a faulty coefficient. |
| [PR17](../dflash_krc_mechanism_diagnostic_20261028/REPORT.md): hinge falls in 16/20 windows; six have no prefix gain. | Lower fitted violation is insufficient. | Sparse repair opportunity, reference choice and BF16 realization remain; local oracle gains are not feasible serving improvements. |
| [PR21](../../experiments/k_prefix_protection_v1/results_20261031/REPORT.md): KP/K saves 0.3435% calls, below 0.5% target; timing unresolved. K_HALF stronger. | Retaining more gains did not establish the better tradeoff. | Exposure, LR/weight decay and seed differences matter; does not select universal smaller LR. |
| [KW](../../experiments/kw_v1/production_20261029/results/REPORT.md): +0.2154% calls saved [−0.6897,1.0527] on primary panel; training +41.082%. | Tested allocator supplies no demonstrated upgrade. | Weak realized allocation and signal quality remain unresolved; not evidence against all weighting. |
| [PR27](../dflash_vfd_second_pass_20261104/REPORT.md): PROGRESS64 adds 73 calls versus FULL; saves one versus RANDOM64. | Tested prioritization signal has no established added value. | Exposure differs versus FULL; equal-budget random/static controls leave the score’s added value unestablished. |
| [PR32](../dflash_role_loss_pilot_20261107/REPORT.md): ROLE call changes unfavorable/unresolved on both datasets; MIX directions differ. | Role-specific loss replacement is not a consistent upgrade. | Scale, clipping and role assignment are unisolated; no particular normalization repair follows. |
| [PR33](../dflash_functional_step_geometry_20261108/PILOT_REPORT.md): FSG calls +6.11%/+5.27%; all 512 corrections applied; protected KL lower. | Tested lower-drift mechanism worsens serving. | Lower enforced KL is not success. Sparse diagnostics do not uniquely explain endpoint harm; this version remains closed. |
| [PR34](../dflash_verification_transition_replay_20261110/REPORT.md): both restart effects positive in 0/31 and 1/40 advances. | Recurrent harmful farther restart unsupported at audited starts. | Advances and verified repairs remain distinct; all 71 happen to pass both. Conditional replay is not gradient-interference evidence or unique latency attribution. |

## 4. INTERPRETATION and decision

**NO SUPPORTED CANDIDATE FROM AVAILABLE EVIDENCE.**

The narrow unresolved question is: **how much of K's useful learning comes from removing suffix supervision, versus selected-weight response normalization and its interaction with shared-parameter optimization?** Existing comparisons change these together, often along different own-feedback trajectories. Retained scalar observations cannot recover the missing parameter-gradient directions or isolate their causal contribution. K+'s immediate-successor redistribution is active, but its small GSM8K increment and unresolved MathQA transfer do not identify a particular harmful limitation to repair.

**UNTESTED HYPOTHESIS boundary:** normalization could influence outcomes through relative response contributions; that mathematical possibility is not evidence that it is harmful. Neither variable D/Z, small redistributed mass, nor lost old predictions justifies an upgrade by itself. No candidate, borrowing survey or pilot is selected. This does not imply optimality or impossibility of improvement.

ARS supervision comprised separate same-team source, metric and historical reviews plus integration checks; it is not an external reproduction. All 11 statistical-fallacy categories were checked, particularly aggregation, reused panels, selected strata, survivorship, multiple comparisons and proxy-to-causal claims. **Zero model loads, forwards/backwards, teacher/serving generations, optimizer applications or checkpoint replays occurred.** Production code/state is unchanged. The audit ends here with K+ retained and the project stage unchanged.
