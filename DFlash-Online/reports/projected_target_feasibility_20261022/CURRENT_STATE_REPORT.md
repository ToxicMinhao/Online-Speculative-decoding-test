# DFlash-Online: implementation, training procedure and projected-target results

**October 22 subsequent evidence:** the proposed checkpoint-only comparison is now complete. P's final fixed-target KL is worse than released F on all16source responses (all32layouts); combined F/C/P KL0.1664/1.5199/1.1799. This replaces the earlier missing-fit evidence with an unfavorable measured result. The historical experiment and its qualification limits remain unchanged. See [the completed diagnostic](fit_check_001/REPORT.md). No new training has been launched.

## Material Passport

- Origin: ARS experiment-agent, validate mode; consolidated technical report.
- Date: October 22, 2026.
- Verification status: **ANALYZED** from retained source and measurements. Preparing this report does not rerun model training or generation.
- Version: current-state report v1.
- Scope: the current Qwen3 / DFlash-Online foundation and the first projected-target feasibility screen. Earlier Llama/OSD, correction-bank and agreement-filter experiments are background, not measurements pooled into this report.
- Decision status: the campaign remains stopped. No new experiment, setting change or project-direction selection follows from this report.

## 1. Current assessment

**We have a working, inspected feedback-to-training path. We do not yet have evidence that the projected-target component improves native drafting over released DFlash.**

The two completed dataset groups distinguish three findings:

1. Ordinary hard-label training improves drafting on its source requests in aggregate, but worsens held-out drafting.
2. Projected-target training reduces that held-out regression relative to ordinary training, but remains worse than frozen DFlash and loses the ordinary control's source-fitting benefit.
3. The incomplete ShareGPT group stopped on exact output equality. A later bounded reconstruction reproduces the disagreement with one unchanged target and observes a BF16 logit tie. No verification/commit defect was identified. This is a numerical validity issue distinct from the two completed groups' unfavorable learning results.

| Qualified workload | Frozen F calls | Ordinary C calls | Projected P calls | Main observation |
|---|---:|---:|---:|---|
| 16 source requests A | 812 | 775 | 890 | C improves source drafting; P worsens it. |
| 16 held-out requests B | 901 | 1,023 | 987 | P is better than C, but both are worse than F. |

These are sums of verification calls for identical output tokens, not training losses or measured deployment speedups. The source and held-out rows answer different questions and should stay separate.

The component is therefore **implemented and partly evaluated, with negative evidence under the declared recipe**. The infrastructure can support further work, but a successful mechanism has not been established.

## 2. What this experiment was designed to test

The question in the [approved handoff](../projected_target_component_20261022/HANDOFF.md) is whether a target distribution that preserves already-correct reference predictions and repairs incorrect ones can retain useful source learning and improve subsequent drafting compared with ordinary adaptation.

This is a bounded component test: collect one eight-response feedback pool, train once over that pool for two passes, then evaluate. It is **not** a repeated per-turn adaptation experiment or a long conversation. Each source response retains its own prompt; requests use separate prompts and fresh request-local caches. This execution separation is not a claim of statistical independence.

| Arm | Initialization | Training signal | Role |
|---|---|---|---|
| F | Released DFlash | None | Frozen native drafting baseline. |
| C | Released DFlash and fresh optimizer | Weighted hard-label cross-entropy on target-verified tokens | Current ordinary SpecForge control. |
| P | Released DFlash and fresh optimizer | Weighted soft cross-entropy against a projected, fixed-reference distribution | Proposed preservation-and-repair component. |

C is **not** the historical AdaFlash OSD forward-KL control. Importing that name would obscure a real difference in the experiment: the current ordinary loss uses the verified token, not the target model's full probability distribution.

All groups and arms are initialized independently. A previous experiment's learned draft or Adam state is not inherited.

## 3. Actual foundation and pinned settings

### Code and environment

The maintained implementation is on A800 at:

```text
/data/sources/minhao/DFlash-Online/
    inference/                     native DFlash inference
    training/                      SpecForge model/training/optimizer code
    online_feedback/               feedback, objective, update and publication
    experiments/projected_target/  this screen's small orchestration package
```

Fenrir `/home/minhao2/DFlash-Research` remains the clean debugging foundation and was not modified by this screen or its consolidation. The local `EAGLE_SPEC/research/projected_target_feasibility_20261022/` directory contains evidence and review material. Its old `runner.py` is historical executed source, not a second maintained runner.

| Item | Pinned value |
|---|---|
| Target | Qwen3-8B, fixed weights |
| Draft | Released Qwen3-8B-DFlash-b16 |
| Target/tokenizer revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Draft revision | `9b41424b7109f9c5413454f481b09a82b85333f4` |
| DFlash upstream revision | `07ebd93db9f472af339b644bb70221ad8428328a` |
| SpecForge upstream revision | `1839aae30003e3573d00979eb1edd57347b3578c` |
| Python / Torch / Transformers | 3.12.15 / 2.13.0+cu130 / 5.12.1 |
| A800 environment | `/data/conda_envs/minhao/dflash_specforge_20261022/` |
| Model directory | `/data/sources/minhao/models/qwen3_dflash_foundation_20261022/{target,draft}` |

These upstream revisions are accompanied by exact executed-source snapshots and compatibility changes; this is not a claim that the checkout is unmodified upstream code. The inference additions include an optional verification-trace callback and cache compatibility handling. The earlier training preparation patch honors the configured cache directory. The online training objective and orchestration live in `online_feedback/` and reuse SpecForge's underlying training machinery.

### Inference and learning

| Setting | Actual value |
|---|---|
| Native serving | Official DFlash/Transformers path; target SDPA attention |
| Draft training attention | Eager, BF16 model tensors |
| Optimizer | AdamW, FP32 CPU master weights/moments; betas 0.9/0.999, epsilon 1e-8, weight decay 0 |
| Trainable parameters | Draft decoder layers, target-feature projection (`fc`) and draft normalization parameters; target model, token embedding and vocabulary head fixed |
| Decoding | Greedy, reasoning off, concurrency 1 |
| Block size | 16 slots, including the known anchor |
| Output limit | 512 tokens or EOS |
| Prompt / context limits | At most 1,024 rendered prompt tokens; context limit 2,048 |
| Cache policy | Fresh target/draft request caches; published weights used by subsequent requests |
| Training seed | 17, reset for each independently initialized arm |
| Event | Eight responses, two passes, response microbatch 1, accumulation 2, eight optimizer steps |
| Sampling | Up to 64 eligible anchors per response presentation |
| Objective chunks | Eight blocks per loss chunk; these are memory chunks, not optimizer batches |
| Position weight | Prediction offset k=1…15: exp(−(k−1)/7) |
| Gradient clipping | Norm limit 1.0 after the accumulated response gradients |
| Projected target | c=1.1; temperature 1; FP32 softmax, projection and loss arithmetic |

The YAML retains `target_backend: sglang` from the upstream configuration, but the online backend explicitly supplies saved native target features. **That field does not mean this screen used the old SGLang serving runner.** Likewise, the online event driver explicitly performs two passes; interpreting `num_epochs: 1` in the generic YAML as the event's pass count would be wrong. See [configuration](execution_config.yaml), [executed-source audit](CONSOLIDATION_PROTOCOL_AUDIT.md), and [consolidated source snapshot](consolidation_001/final_source.zip).

## 4. Dataset construction and evaluation order

Each dataset has eight A requests for feedback and eight B requests for held-out evaluation: 48 distinct requests across three groups. Selection seed is 20261022, and the manifest was sealed before generation.

| Source | Revision | Partition used |
|---|---|---|
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | Main/train for A; main/test for B. |
| `sahil2801/CodeAlpaca-20k` | `3bd2de8099ee8a140653ad9b9df783554d5efabb` | Last 1,024 rows held for B; preceding rows for A. |
| `Aeala/ShareGPT_Vicuna_unfiltered` | `8b0048ad6ae8c22f46a78c15559dec98feef5539` | Declared source file, rows 0–29,999 for A and 30,000–31,023 for B; eligible initial requests. |

The [manifest](data/approved/manifest.json) contains exact source IDs, prompts, ordering, language/length/continuation exclusions and tokenizer bindings. ShareGPT uses the first eligible user request from a verified initial conversation chunk; it does not turn an arbitrary continuation into a standalone prompt. No dataset assistant answer is used as a target or conversation history.

The user explicitly allowed historical reuse of the eight GSM8K A questions, with **fresh generated responses and fresh learners**. B remains outside training and unused according to the retained project-history inventory. This is not a claim of absence from model pretraining, semantic duplicates or inaccessible history.

For each group, the actual order is:

```text
Released F serves A1…A8 and B1…B8 → save canonical tokens and traces
                         ↓
Prepare teacher features for A1…A8 only
                         ↓
Fresh C → train on A twice → publish → generate A and B naturally
Fresh P → train on the same A twice → publish → generate A and B naturally
                         ↓
Compare exact outputs/termination, verification work and recorded costs
```

B's F answers exist before adaptation because they are saved references, but their answers and features never enter C or P training. Evaluating A is a source-fit probe; evaluating B asks about transfer. Adapted evaluation generates answers through ordinary speculative decoding, not teacher forcing.

## 5. What is actually stored in feedback

The reusable online controller has a durable response buffer. This experiment calls the same backend and trainer directly to give C and P **one identical immutable pool**. It does not need the controller's changing serving queue.

There are two stages of retained information:

| Stage | Saved information | Purpose |
|---|---|---|
| Native serving record | Exact prompt and output IDs, termination, proposals, verifier argmax IDs, accepted/committed tokens, block positions and metrics | Canonical answer, greedy feedback and reproducible native trajectory. |
| Prepared training payload | `input_ids`, `loss_mask`, `hidden_states`, `target_last_hidden_states`, `record_id`, `prompt_length`, `label_binding` | Target conditioning, correctly aligned labels, and reproducible response-only training. |

Thus “the buffer saves the response” is incomplete: the training path also needs exact tokens, masks and target conditioning. Conversely, it does **not** buffer every vocabulary-sized probability vector.

| Symbol | Meaning | Stored or recomputed? |
|---|---|---|
| y | Target-verified token at the aligned canonical training position | Bound to the saved answer and checked native target replay. |
| q0 | Immutable pre-event draft's probability distribution for a sampled block position | Recomputed during P training from the fixed reference. |
| r | Projected soft target calculated from q0 and y | Recomputed and detached; not a saved teacher distribution. |
| qθ | Current learner's distribution | Recomputed with the changing trainable draft. |

Native generation records compact traces; it does not capture large training features inside the request timer. A separate teacher-preparation stage replays the exact saved target prefill and verification schedule, validates every saved target argmax, and assembles features for the committed canonical sequence. Rejected proposals can occur in the reconstruction, but they are not substituted for committed answer tokens in the training payload.

The historical receipt field `full_posterior_match` means **all checked argmax token IDs match**. It does not mean full logits/probability vectors were stored or compared bitwise.

## 6. Exact training procedure: eight responses → eight steps

### Sampling blocks

Responses are visited in fixed manifest order, A1 through A8. Within each presentation, SpecForge randomly selects up to 64 eligible anchor positions without replacement, then orders the selected anchors by position. An eligible anchor and its successor must both be in the response mask. There must be at least two eligible consecutive response tokens.

Each selected block has this structure:

```text
slot:        0          1          2             …        15
input:    known token  [MASK]     [MASK]          …       [MASK]
loss:         no       predict    predict        …       predict
```

Block size 16 therefore means **one known anchor plus at most 15 predicted positions**. The first response token can be an anchor, but the author sampler does not supervise it as a prediction label in this setup. Prompt positions, anchor slots and positions beyond the saved answer are masked out. End blocks can have fewer than 15 supervised positions.

Sampling is over possible training anchors, not over the exact block boundaries used during generation. Distinct sampled blocks may overlap and supervise a response token more than once. Two passes may use different anchors. C and P must use identical draws for the corresponding presentation. The repeated exposure counts below are not counts of unique answer tokens.

Each block has target conditioning from the canonical history strictly before its anchor, plus its known anchor token. It cannot read other sampled blocks or later canonical answer features. Within the block, masked slots may attend to each other under the author's full-attention configuration; these slots contain masks, not future answer tokens. This preserves causal access to the answer without requiring triangular attention among the masked positions.

Consequently, q0 at offset k is the frozen draft's **parallel-block prediction from the anchor's available context**. It is not an ordinary next-token prediction supplied with all canonical tokens through offset k−1. The verified target label y, in contrast, corresponds to its own canonical prefix. This distinction is part of DFlash's training setup, and prevents describing the training forward as autoregressive regeneration of the saved response.

### Loss and optimizer accumulation

For response i, let M mark eligible predicted positions and w be the decay weight. Both arms use the same denominator:

\[
L_i = \frac{\sum_{b,k}M_{i,b,k}w_k\,\ell_{i,b,k}}
              {\sum_{b,k}M_{i,b,k}w_k}.
\]

For C, \(\ell=-\log q_\theta(y)\). For P, \(\ell=-\sum_v r(v)\log q_\theta(v)\). Compute a weighted mean **within each response**, then accumulate half of each of two response losses. This gives equal response weighting; it is not one pooled position mean that lets the longer response dominate.

Loss work is chunked into eight-block pieces to limit memory. Chunking and backward recomputation do not create extra optimizer steps. The optimizer changes weights only after the second response in an accumulation pair.

| Pass | Response presentations | Resulting optimizer steps |
|---|---|---|
| 1 | A1+A2, A3+A4, A5+A6, A7+A8 | 1, 2, 3, 4 |
| 2 | A1+A2, A3+A4, A5+A6, A7+A8, with newly sampled blocks | 5, 6, 7, 8 |

The learner and optimizer retain their changes throughout both passes. The P reference stays fixed throughout all eight steps. Its extra forward preserves RNG state, so reference evaluation does not perturb C/P anchor matching.

Conceptually:

```python
learner = released_draft_with_fresh_optimizer(seed=17)
reference = frozen_copy(learner) if arm == "P" else None

for training_pass in range(2):
    for response in A_in_manifest_order:
        anchors = sample_response_anchors(response, max_anchors=64)
        labels = verified_labels(response, anchors)
        current_logits = learner.masked_blocks(response, anchors)
        if arm == "P":
            with preserve_rng(), no_grad():
                reference_logits = reference.masked_blocks(response, anchors)
                r = projected_target(reference_logits, labels)
            loss = weighted_response_mean(soft_cross_entropy(current_logits, r))
        else:
            loss = weighted_response_mean(hard_cross_entropy(current_logits, labels))
        backward(loss / 2)
        if two_responses_have_accumulated:
            clip_gradients_and_step_optimizer()
            advance_scheduler_and_clear_gradients()

publish_trained_draft()  # native evaluation starts after this
```

This is explanatory pseudocode, not a replacement implementation. The constructor takes original reference logits, determines native argmax/ties, then computes FP32 q0 and r internally; checking argmax only after softmax rounding could change tie classification. Real vocabulary work is chunked; the source also records anchors, loss denominators, RNG, publication and validation receipts.

### Actual learning rates and exposure

The nominal LR is 6e-4, with cosine scheduling and warmup ratio 0.04. The eight-step event has zero effective warmup steps. The preserved upstream scheduler starts slightly above the nominal rate; it was measured, disclosed and matched, rather than silently replaced by a constant rate.

| Step | Actual LR used |
|---:|---:|
| 1 | 0.0006237396779379481 |
| 2 | 0.0006000000000000001 |
| 3 | 0.0005323951169514922 |
| 4 | 0.00043121725939676895 |
| 5 | 0.0003118698389689741 |
| 6 | 0.00019252241854117926 |
| 7 | 0.00009134456098645594 |
| 8 | 0.000023739677937948036 |

The scheduler reaches zero after step 8. These are not the historical per-turn OSD learning rates.

| Dataset | Presentations per arm | Steps per arm | Supervised-position exposures per arm | Sum of position weights |
|---|---:|---:|---:|---:|
| GSM8K | 16 | 8 | 15,014 | 6,690.117535 |
| CodeAlpaca | 16 | 8 | 14,914 | 6,661.177241 |
| ShareGPT | 16 | 8 | 14,948 | 6,665.177930 |

C/P anchors, masks, exposure, actual LR sequence and initial/final RNG receipts match in every group, including the stopped ShareGPT group. Although the configured cap is “up to 64,” all recorded presentations in this screen selected 64 anchors. Source: [protocol audit](CONSOLIDATION_PROTOCOL_AUDIT.md) and the six raw `TRAINING.json` receipts under [final recovery](retrieved/final_recovery/campaign_002/).

## 7. What the projected objective changes—and does not guarantee

P obtains q0 from a detached copy of the pre-event draft. The reference is outside the trainable module/optimizer state, remains in evaluation mode, and is discarded after the complete event. For the same sampled example, q0 and r remain fixed while qθ changes. A different sampled anchor or context has its own q0 and r; r is not a global distribution attached to a vocabulary token.

- If the reference's native argmax already selects y, use r=q0.
- Otherwise, reduce the highest competitor probabilities to a common threshold τ and raise y to cτ, where c=1.1. Smaller unaffected probabilities remain unchanged.

For the active clipped competitors S:

\[
\tau=\frac{q_0(y)+\sum_{j\in S}q_0(j)}{c+|S|},\qquad
r(y)=c\tau,\qquad r(j)=\min(q_0(j),\tau),\ j\ne y.
\]

The implementation finds the active set by sorting competitors. Ties follow native lowest-ID argmax. Correctly selected tied rows are preserved; a wrong tied selection is repaired. Native BF16 head logits are converted for FP32 softmax/projection/loss. Underflow zeros are retained; there is no probability floor or silent renormalization.

Example, with the third token as y:

```text
q0 = (0.400000, 0.350000, 0.050000, 0.200000)
r  = (0.258065, 0.258065, 0.283871, 0.200000)
```

The learner is trained toward r. **Learning changes qθ, not r.** The earlier request to show “r changing” is better understood as showing qθ move toward a correctly constructed fixed r.

The unweighted soft-CE logit gradient is qθ−r. Soft CE equals KL(r‖qθ) plus the fixed entropy of r, so those two have the same mathematical gradient with respect to the learner. This does not make P a full-target-distribution distillation method: r comes from the draft reference and the verified token.

This objective encourages preservation; it does not impose a hard constraint on the model's other predictions. At initialization an already-correct sampled row has approximately zero direct logit gradient, but updates from other rows can still change it through shared backbone parameters. Later samples need not cover that same row. Similarly, assigning a better target does not guarantee the real optimizer will fit it within eight steps or that greedy accepted prefixes will improve.

The prior [token walkthrough](../projected_target_token_trace_20261022/REPORT.md) verified six cases and 606 observations through the actual target constructor/loss. Maximum discrepancy from q−r was 7.98e-8. Its free-logit example first selected the desired token at step 12. That used FP32 free logits and a direct step of 1.0; it is not a prediction of DFlash/AdamW learning speed or a demonstration of transfer.

## 8. What completed and what stopped

| Work | Planned | Executed | Qualification status |
|---|---:|---:|---|
| GSM8K F/C/P generations | 48 | 48 | Complete and qualified. |
| CodeAlpaca F/C/P generations | 48 | 48 | Complete and qualified. |
| ShareGPT F/C/P generations | 48 | 40 | Exact-output mismatch at P A07; eight P-B generations missing. |
| All response presentations | 96 | 96 | All six learning events completed. |
| All optimizer steps | 48 | 48 | Eight steps for each C/P event in all three groups. |

There are **136 distinct generated-answer executions**, comprising 16 F answers reused from the initial attempt and 120 new recovery executions. The original attempt failed during teacher preparation before any training. Its existing F answers were reused, not regenerated or counted twice.

The initial preparation method used a full-sequence target forward and disagreed with the native cached target path at 2/240 positions in the first source response. The repair reconstructed the exact recorded native target blocks and cache schedule. Across all 24 source responses, it performed 2,300 target forwards and validated 35,626 verifier argmax rows. These are teacher-preparation operations, not extra generated answers.

Independent audit reconstructed all 11,730 recorded verification blocks. Each trace was internally consistent with its own committed answer, accepted prefix and termination. Of the 136 records, 135 pass the corresponding F input/output/termination equality checks. Only the **96 records in the two completed groups** enter qualified gains; all 40 ShareGPT records remain visible outside that denominator.

All six publication checks matched 58 draft tensors. Initialization, frozen-reference and optimizer-exclusion receipts passed. The target guard checks parameter identities/versions, gradient flags and optimizer exclusion; it is not a repeated bitwise scan of every target weight. ShareGPT's final group-wide target check was not reached after the mismatch.

The maintained experiment package was subsequently consolidated without changing its scientific recipe. All 133 CPU tests pass in the pinned A800 environment, and the moved reporter regenerates the five historical outputs byte-for-byte. This is evidence of the code consolidation's consistency, **not a new GPU run of the consolidated driver**.

## 9. Native drafting results

“Verification calls” counts speculative target-verification blocks. It excludes target prefill and later teacher-preparation forwards. A call may accept several draft tokens and commit a target correction/bonus token; it is not one call per output token.

Here, percentage change is:

\[
100\left(\frac{\sum\text{arm calls}}{\sum\text{baseline calls}}-1\right).
\]

**Positive means more verification work; negative means less.** This sign convention differs from an earlier slide's “reduction” convention. Ratios use summed matched measurements, not an average of per-request percentages.

| Dataset | Set | Requests | F | C | P | C vs F | P vs F | P vs C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | A: source | 8 | 356 | 351 | 394 | −1.40% | +10.67% | +12.25% |
| GSM8K | B: held out | 8 | 303 | 367 | 350 | +21.12% | +15.51% | −4.63% |
| CodeAlpaca | A: source | 8 | 456 | 424 | 496 | −7.02% | +8.77% | +16.98% |
| CodeAlpaca | B: held out | 8 | 598 | 656 | 637 | +9.70% | +6.52% | −2.90% |
| Two qualified groups | A | 16 | 812 | 775 | 890 | −4.56% | +9.61% | +14.84% |
| Two qualified groups | B | 16 | 901 | 1,023 | 987 | +13.54% | +9.54% | −3.52% |

| Paired comparison | Source A: fewer / more / tied calls | Held-out B: fewer / more / tied calls |
|---|---|---|
| C vs F | 10 / 6 / 0 | 1 / 13 / 2 |
| P vs F | 4 / 11 / 1 | 2 / 12 / 2 |
| P vs C | 1 / 15 / 0 | 11 / 2 / 3 |

The observed pattern is consistent across both completed datasets: C has aggregate source gains and held-out losses; P has less held-out loss than C but greater source loss. Individual exceptions remain in the table rather than being removed.

| Qualified set | Output tokens per arm | Accepted draft tokens F / C / P | EOS requests | Cap hits |
|---|---:|---|---:|---:|
| A | 4,251 | 3,436 / 3,474 / 3,357 | 14/16 | 2/16 |
| B | 5,131 | 4,229 / 4,103 / 4,138 | 13/16 | 3/16 |

Accepted draft tokens exclude target correction/bonus tokens. Verification-committed progress includes them, but excludes the initial token produced by target prefill. Across these 16-request subsets, verification-committed totals are therefore 4,235 for A and 5,115 for B, identical across arms. Dividing those totals by verification calls gives committed progress per verification. Dividing accepted draft tokens by calls gives a different quantity; neither should be mislabeled as the other. Dividing all output tokens by calls also differs slightly because it includes the initial prefill tokens.

These data demonstrate **reduced held-out harm compared with C**, not adaptation benefit over F, successful source preservation, or general superiority of the projection. Different raw hard/soft losses are not comparable fit scores. No common held-out loss study, margin sweep, learning-rate sweep or alternative soft-target control was added to this screen.

Sources: [all per-request records](retrieved/final_recovery/campaign_002/PER_REQUEST.csv), [paired pooled records](retrieved/final_recovery/campaign_002/POOLED.csv), [original results](RESULTS.md).

## 10. ShareGPT numerical mismatch: updated diagnosis

The last token of `sharegpt_A_07` differs: F/C generate ID 9492, ` tags`; P generates ID 3847, ` users`. The prompt and first 511 answer tokens match. All arms reach the 512-token cap.

F/C's last block starts at output position 508 with width four. P's starts at 510 with width two. Both predict the disputed token from the same canonical token prefix, but their numerical block/cache execution histories differ.

The later diagnostic loaded **one fixed target**, without a draft, training or checkpoint publication, and reconstructed both saved paths. It reproduced all 3,602 F and 3,292 P verifier argmax rows. At the disputed position:

| Reconstructed target path | Logit for ` tags` | Logit for ` users` | Target choice |
|---|---:|---:|---|
| F schedule | 25.125 | 25.000 | ` tags` |
| P schedule | 25.125 | 25.125 | ` users`, lower ID in the exact tie |

At this magnitude the observed difference is one BF16 representable step, 0.125. Repeating each final call from an exact copy of its own pre-block cache gives a bitwise-identical logit vector. All logits are finite. Target identities/versions remain unchanged.

F/C reject proposed `users` and commit their target's `tags`. P accepts `users` because its own target prediction agrees. Thus the retained traces do not show a rejected proposal being improperly appended, a wrong verification offset or a cap-handling error.

**The supported explanation is reproducible execution-path sensitivity with a measured BF16 tie.** Target-weight mutation is unnecessary to reproduce the difference. The diagnostic does not isolate a specific kernel, accumulated cache rounding versus the last block's shape, or what higher-precision execution would predict. It does not certify every possible input.

This supersedes the original reports' “BF16 tie not established” statement. Those reports remain unchanged as historical evidence. The exact-output gate still failed under its declared definition; this report does not silently relax it, discard A07 or promote ShareGPT into qualified gains.

The numerical check used 443 target forwards, zero new answers and zero optimizer steps; wall time was 27.464 s including loading, peak allocated GPU memory 16,846,258,176 bytes. These are separate diagnostic costs. See [numerical report](numerical_check_001/REPORT.md) and [raw result](numerical_check_001/RESULT.json).

## 11. Cost and timing boundaries

The screen records single, sequential, instrumented request latencies. Request timing includes native generation, prefill/drafting/verification, compact trace recording and synchronization. Tokenization, validation, JSON persistence, feedback preparation, training and activation are outside that boundary. Capture of the large teacher features is disabled during generation.

There are no three-repeat timing estimates here. These timings do not establish robust serving acceleration, and a verification-call improvement is not automatically an elapsed-time improvement.

| Qualified set | F request time sum | C request time sum | P request time sum |
|---|---:|---:|---:|
| A, 16 requests per arm | 33.623 s | 31.317 s | 35.860 s |
| B, 16 requests per arm | 34.846 s | 39.075 s | 37.873 s |

These are descriptive single-run measurements. Method order is sequential, with the initial cold execution included; no inference-only percentage here should be promoted into a deployment speedup. All raw request times remain available for inspection.

| Group | Shared teacher preparation | C / P step training | C / P checkpoint save | C / P export | C / P activation |
|---|---:|---:|---:|---:|---:|
| GSM8K | 13.87 s | 31.53 / 31.69 s | 38.07 / 37.73 s | 18.11 / 18.20 s | 0.37 / 0.41 s |
| CodeAlpaca | 15.61 s | 30.28 / 32.58 s | 38.07 / 38.12 s | 18.22 / 17.64 s | 0.43 / 0.41 s |
| ShareGPT, incomplete | 48.92 s | 27.77 / 31.01 s | 37.78 / 37.87 s | 17.50 / 17.32 s | 0.40 / 0.39 s |

Across all six events, step training totals 184.856 s, checkpoint/save 227.633 s and export 106.995 s. These are **inside** 524.037 s event wall. Trainer loading is separately measured at 104.311 s. Shared teacher preparation is 78.405 s, plus 1.366 s feature persistence. Failure and incomplete-group costs remain included.

P's reference/head/projection timings are nested within training. Reference setup is inside event wall but outside step training. Do not add nested subcosts a second time. `activation_s` and `native_activation_s` in this runner describe the same interval; count once. Publication validation is a separate diagnostic boundary. The component tables are not a complete partition of total runtime.

| Overall boundary | Measured value | Interpretation |
|---|---:|---|
| Original stopped attempt | 37.984 s | Includes the original 16 F generations and failure before training. |
| Recovery runner | 1,175.149 s | Includes 120 new generations and all learning events. |
| Consumed runner time | **1,213.133 s, about 20.22 min** | Sum of the two intervals; reused F time counted once. |
| All 136 request timers | 455.379 s | Component of consumed work, not additive to runner time. |
| Separate numerical diagnostic | 27.464 s | Extra analysis work; not part of the original campaign timer. |

The recovery wrapper independently records 1,177.474 s around that same recovery process; it is not another additive campaign. Development, repair pauses, external preflight and retrieval are outside the consumed-runner total. It is not a wall-clock time-to-completion claim for the full planned screen, which did not finish.

Observed training-through-export GPU allocation peaks range from 28.00 to 36.90 GiB. The first group's C/P peaks are 28.00/36.90 GiB. These intervals include resident native models; they exclude later serving, host Adam state and disk. The difference is not an isolated causal measurement of projection overhead.

Detailed overlapping timer definitions, per-event receipts and GPU scope are preserved in [COST_AUDIT.md](COST_AUDIT.md) and [COST_BOUNDARIES.csv](COST_BOUNDARIES.csv).

## 12. Evidence strength and remaining questions

| Question | Supported answer | Boundary |
|---|---|---|
| Does the local target constructor/loss implement the intended mapping? | The numerical cases, detachment and gradient checks pass. | Free-logit correctness does not guarantee shared-model fitting. |
| Can the online infrastructure collect feedback, update, export and reload? | Recorded native qualifications and completed groups support this. | Not a proof of all-input invariance or asynchronous deployment feasibility. |
| Did C and P receive matched data and optimization exposure? | Yes, within audited records. | Different objectives legitimately yield different gradients. |
| Is q0 drifting during the event? | Fixed-reference creation/end checks pass. | Different sampled blocks have different q0 values. |
| Did P improve drafting over F? | No in the two completed groups. | Small, fixed-recipe screen; not a universal impossibility claim. |
| Did P improve over C on B? | Yes in aggregate and on 11/16 paired requests. | Both remain worse than F; source fitting is worse under P. |
| Did the ShareGPT mismatch prove verification was broken? | No; both target paths reproduce, with the measured BF16 tie. | Exact output equality still fails under the original gate. |
| Is the complete online research problem solved? | No. | No useful complete mechanism or deployment speedup is established. |

The source-target construction is a soft preservation objective, not a guarantee that useful predictions stay fixed. Several explanations for its outcome remain possible: insufficient real-model fitting, a trade-off between preservation and repair, optimizer interaction, or a gap between the supervised positions and greedy block acceptance. **This screen does not distinguish those causes.** In particular, fewer calls than C on B alone does not show that P learned which changes transfer.

There are 16 qualified B requests, two datasets and one training seed. No confidence interval or significance claim is supplied. Historical freshness filters, length eligibility and the excluded incomplete third group limit generalization. Accepted tokens/blocks are dependent measurements within requests, not thousands of independent test examples.

## 13. What to discuss next

My recommendation is to keep the working infrastructure and discuss the objective's evidence before adding another mechanism. Three decisions are now concrete:

1. **What counts as acceptable native output validity in future screens?** The current mismatch is consistent with numerical path sensitivity. A future protocol could distinguish algorithmic trace validity from cross-path token identity, but must explicitly state how differing outputs affect comparisons. The existing record should remain a failed exact-equality pair.
2. **What evidence would justify more work on this objective?** The target-construction checks pass, yet P does not improve source drafting. Any further bounded check should distinguish a stated explanation—for example, failure to fit the intended targets versus fitting without useful native progress—rather than search settings until a positive number appears. This is a discussion question, not authorization for another diagnostic.
3. **What useful outcome must a revised component achieve?** Retain competent source learning while improving held-out drafting over frozen F, and then account for preparation/training/activation. Beating a control whose held-out drafting regresses while remaining worse than F is insufficient by itself.

There is no evidence here that we need a larger controller, a new router or a more complicated buffer. The known numerical issue and the unfavorable learning result should be handled as separate questions. No continuation, new arm, tuning or redesign has been started while preparing this report.

## 14. Where to inspect the code and evidence

Read the A800 maintained code in this order:

| File under `/data/sources/minhao/DFlash-Online/` | What to inspect |
|---|---|
| `experiments/projected_target/run.py` | `run()` orchestrates F collection, shared A preparation, C/P learning and evaluation. |
| `online_feedback/native.py` | Native generation, `prepare_feedback`, and checkpoint publication/timing. |
| `online_feedback/native_replay.py` | Exact saved-block target reconstruction and canonical feature/label binding. |
| `online_feedback/training.py` | `train_event`, pass/response loop, SpecForge calls and checkpoint export. |
| `online_feedback/objective.py` | `FrozenReference`, event lifecycle, matched block forwards and chunked loss. |
| `online_feedback/projected_target.py` | The small q0→r constructor. |
| `training/specforge/` | Author block sampler, block/context attention masks, trainer/optimizer and LR scheduling. |
| `online_feedback/buffer.py`, `controller.py` | Reusable persistent online sequence, not the fixed-pool screen's orchestration. |

The [final source archive](consolidation_001/final_source.zip) is available locally for review. The [exact executed archive](repair_001/executed_source.zip) preserves the pre-consolidation run. The consolidation changes reading structure, LR logging and optional reference qualification hooks; it does not replace the objectives or optimizer.

Important retained artifacts:

- [Sealed manifest and exact request IDs](data/approved/manifest.json), [GSM8K reuse authorization](data/approved/AUTHORIZED_EXCEPTION.json), [configuration](execution_config.yaml).
- [Per-request raw table](retrieved/final_recovery/campaign_002/PER_REQUEST.csv), [pooled table](retrieved/final_recovery/campaign_002/POOLED.csv), [structured summary](retrieved/final_recovery/campaign_002/SUMMARY.json).
- [Recomputed report tables and audit totals](CURRENT_STATE_TABLES.json), including accepted versus committed progress and all observed request-time sums.
- [Protocol audit](CONSOLIDATION_PROTOCOL_AUDIT.md), [final stopped-run audit](FINAL_AUDIT.md), [new numerical diagnosis](numerical_check_001/REPORT.md).
- [Consolidation verification](consolidation_001/REPORT.md), [CPU test log](consolidation_001/TESTS_FINAL.log), [historical report reproduction](consolidation_001/REPORT_REPRODUCTION.json).
- [This report's review](CURRENT_STATE_REVIEW.md), which separates checks performed now from prior GPU evidence.

On A800, original campaign artifacts remain under `/data/sources/minhao/runs/projected_target_feasibility_20261022/recovery_001/campaign_002/`. All six learned model exports, 24 source feature payloads, exact token records, compact traces, configurations and anchor/RNG evidence were retained at the last verified inventory. Five completed-arm optimizer payloads were intentionally retired after export validation: their exact historical Adam/master-state continuation is lost, **not archived**. ShareGPT P's full stopped-run restart state remains retained.

This report reads local retained evidence; it does not perform a new remote storage/availability audit. The formal research stage and paused campaigns are unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
