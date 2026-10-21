# Online speculative decoding — research tools

Focused code for controlled speculative-decoding research. Generated datasets, model weights, raw experiment results and private operational records are kept outside this repository.

- [DFlash matched-prefix diagnostic](experiments/dflash_matched_prefix_diagnostic/README.md): retained F/S8/L8 checkpoints at fixed canonical block starts, with EOS-inclusive prefix scoring and censored cap boundaries. Implementation for review; GPU execution remains paused.
- [DFlash lower-LR control](experiments/dflash_lower_lr_control/README.md): six fresh L8 checkpoints at `3e-5`, matched to retained F/S8 training and diagnostic records. All six groups completed and audited; source fitting and held-out transfer remain separate.
- [DFlash KL-to-native diagnostic](experiments/dflash_kl_native_diagnostic/README.md): retained F/S8 checkpoints, 24 natural source replays and 48 read-only held-out KL evaluations, with zero optimizer steps. Code submitted for review; GPU execution has not started.

- [Bounded OSD gain attribution](experiments/dflash_osd_attribution/README.md): existing five-stream analysis, at most 160 matched-prefix pairs, and a fixed-space mathematical capacity screen. Implementation for review; no new analysis or GPU diagnostic has run.

- [STABLE/SHORT dataset construction](datasets/stable_short/README.md): pinned-source prompt selection, deterministic stream assembly and validation. No inference or training.
- [Two-update dataset extension](datasets/two_update/README.md): preserves the original selection and adds balanced RANDOM_MIX streams with chronological evaluation-then-support phases.
- [DFlash two-update experiment](experiments/dflash_two_update/README.md): response-specific masking, persistent full-weight forward-KL updating, matched native evaluation and cost accounting. Code review stage; no new GPU experiment result is claimed.
- [Two-update follow-up comparisons](experiments/dflash_two_update_followup/README.md): retained U1/U2 on identical final inputs and the pinned AdaFlash OSD training path under the existing session protocol. Capture-free checkpoint timing, per-example response masks and explicit trajectory comparisons; awaiting review before GPU execution.
- [Per-turn adaptation pilot](experiments/dflash_per_turn/README.md): six approved conversation streams, one two-step update per completed response, identical-input checkpoint replays and isolated lower-rate diagnostics. Exact user whitespace is preserved. Code review only; model execution waits for approval.
- [Frozen-DFlash correction bank](experiments/dflash_correction_bank/README.md): four fixed-space correction modules, causal selection and constrained writes with cumulative protection; fresh comparisons with frozen DFlash and the qualified OSD path. Implementation submitted for review; native qualification and the six-stream campaign have not run.

- [Agreement-filtered OSD pilot](experiments/dflash_osd_agreement_filter/README.md): bounded five-arm, three-request comparison of selective versus uniform update control; implementation for review, GPU execution on hold.

- [Fixed-feedback DFlash learning curve](experiments/dflash_baseline_learning_curve/README.md): six independent four-feedback/four-evaluation groups, continuous checkpoints at steps 0/2/8, and training-free held-out timing. Implementation submitted for review; GPU execution remains paused.
