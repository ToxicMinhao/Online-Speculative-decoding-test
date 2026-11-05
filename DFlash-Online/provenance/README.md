# Retained experimental source

This is a post-execution source archive. It does not claim that historical experiments originally launched from this Git commit. Each report links to its run manifest, which preserves source paths and original source-hash contracts. Content-addressed objects store exact source bytes once; the same file may have several historical revisions. Missing historical bytes remain explicit.

The run manifests include implementation, launch, diagnostic, audit and analysis code. Readable original run-root Python and shell files are in `run_tools/<run-id>/`; fixed model/runtime/training pins and `config.yaml` copies are in `config_pins/<run-id>/` with their original relative paths. Every readable copy is bound to an exact object hash. These scripts retain their original qualified paths and are not newly validated turnkey launches.

Raw feedback, weights, optimizer states, outputs and request panels remain outside this source archive. Immutable original reports are identified by their original hashes and receive provenance sidecars rather than being rewritten.

Use `python archive.py list RUN_ID` to inspect available source contracts, `python archive.py verify` to check the archive, and `python archive.py extract RUN_ID --contract ORIGINAL_MANIFEST_PATH --destination NEW_DIRECTORY` to reconstruct one exact DFlash-Online source contract. Use `--prefix` for an external runtime or original run-driver directory. Never combine conflicting versions from different manifests into a claimed historical run.
