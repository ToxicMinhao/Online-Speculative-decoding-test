# Online speculative decoding — research tools

Focused code for controlled speculative-decoding research. Generated datasets, model weights, raw experiment results and private operational records are kept outside this repository.

- [STABLE/SHORT dataset construction](datasets/stable_short/README.md): pinned-source prompt selection, deterministic stream assembly and validation. No inference or training.

- [DFlash-Online implementation and experiments](DFlash-Online/README.md): the actual online feedback/training infrastructure, pinned upstream source, bounded comparison runners, and [experiment reports](DFlash-Online/reports/README.md). Historical source versions are preserved with explicit report-to-code mappings; private feedback and model states remain outside Git.
