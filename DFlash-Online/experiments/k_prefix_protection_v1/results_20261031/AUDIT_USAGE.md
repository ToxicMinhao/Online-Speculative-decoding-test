# Reproducing the record audit

Scientific execution commit: `d10521a1aa7d256bb75b37a6aae07586ae7cc8b3`.

The included `INDEPENDENT_RECORD_AUDIT_SCRIPT.py` is the exact post-run CPU analysis script (SHA-256 in `PROVENANCE.json` and `INDEPENDENT_RECORD_AUDIT.json`). It was not part of the GPU execution. It requires NumPy and the full private compact evidence tree, including native traces and the eight original schedules, rather than this smaller public package.

```bash
CUDA_VISIBLE_DEVICES='' python INDEPENDENT_RECORD_AUDIT_SCRIPT.py /path/to/compact_evidence_001 --output /path/to/new_audit_directory
```

Use a new output directory. This reads retained measurements and metadata, runs no models, and performs no training or generation. Large model/optimizer tensors remain on A800 and are outside this audit's scope.
