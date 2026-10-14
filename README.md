# Online speculative decoding — research tools

Focused code for controlled speculative-decoding research. Generated datasets, model weights, raw experiment results and private operational records are kept outside this repository.

- [STABLE/SHORT dataset construction](datasets/stable_short/README.md): pinned-source prompt selection, deterministic stream assembly and validation. No inference or training.
- [Two-update dataset extension](datasets/two_update/README.md): preserves the original selection and adds balanced RANDOM_MIX streams with chronological evaluation-then-support phases.
- [DFlash two-update experiment](experiments/dflash_two_update/README.md): response-specific masking, persistent full-weight forward-KL updating, matched native evaluation and cost accounting. Code review stage; no new GPU experiment result is claimed.
