# Online speculative decoding — research tools

Focused code for controlled speculative-decoding research. Generated datasets, model weights, raw experiment results and private operational records are kept outside this repository.

- [STABLE/SHORT dataset construction](datasets/stable_short/README.md): pinned-source prompt selection, deterministic stream assembly and validation. No inference or training.
- [Two-update dataset extension](datasets/two_update/README.md): preserves the original selection and adds balanced RANDOM_MIX streams with chronological evaluation-then-support phases.
- [DFlash two-update experiment](experiments/dflash_two_update/README.md): response-specific masking, persistent full-weight forward-KL updating, matched native evaluation and cost accounting. Code review stage; no new GPU experiment result is claimed.
- [Two-update follow-up comparisons](experiments/dflash_two_update_followup/README.md): retained U1/U2 on identical final inputs and the pinned AdaFlash OSD training path under the existing session protocol. Capture-free checkpoint timing, per-example response masks and explicit trajectory comparisons; awaiting review before GPU execution.
- [Per-turn adaptation pilot](experiments/dflash_per_turn/README.md): six approved conversation streams, one two-step update per completed response, identical-input checkpoint replays and isolated lower-rate diagnostics. Exact user whitespace is preserved. Code review only; model execution waits for approval.
