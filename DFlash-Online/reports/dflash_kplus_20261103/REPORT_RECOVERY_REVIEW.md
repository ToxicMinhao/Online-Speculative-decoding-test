## Material Passport

- Mode: validate; scoped CPU reporting repair review.
- Verification status: source reviewed and 20 focused CPU tests passed; complete production endpoint reconstruction remains pending.
- Scientific execution: `e0a9d3c235e1efd15404c549282d21c99243a21c`. The separately published reporting repair does not replace that scientific execution identity.
- Scope: tests and local source review only; no GPU, remote operation or production evidence edit.

The minimal caller contract is `audit_stage(root / 'campaign_001/evaluation/final', ..., evidence_root=root)` in both campaign and report. The existing reader uses the recorded production-root prefix to rebase the saved native path. Passing the run root restores access to the native sidecar without altering validation, counts or tokens.

The new fixtures place actual saved sidecars under `campaign_001/final_native_attempt_001/native`, with paths recorded under a different original host root. They reproduce the omitted-root failure, then pass using the run root. Removing a sidecar still fails; modifying a sidecar still fails even when the embedded raw capture remains valid. The caller tests verify the run-root argument at both entry points.

The explicit report-only recovery gate requires the preserved missing-path failure, exact original exit states, passed postflight, complete training and a recomputed full 6,144-row audit. It verifies the original resource/timing boundaries and rejects cleanup errors or retries. Original statuses and reports remain unchanged, with preserved-file hashes attached to the recovery disposition. Recovery requires a new separate output directory. Two review findings were resolved before qualification: recovery validation now precedes endpoint calculation/writes, and opt-in always validates the original failure rather than silently accepting a different campaign status.

The tests reject absent opt-in, incomplete/invalid raw evidence, different original failures/statuses, partial training, failed timing/resource/postflight, wrong exits, altered original audit, cleanup errors and retry flags. They also verify original failure bytes and hashes survive successful recovery. Statistical computations are mocked only in recovery-control fixtures; saved-sidecar reconstruction tests use the real raw validator and path reader.

Exact command from the publication checkout:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=DFlash-Online:DFlash-Online/tests /home/minhao2/.venvs/dflash-specforge/bin/python -m unittest -v test_kplus_full_evaluation test_kplus_full_report test_kplus_full_campaign > /home/minhao2/EAGLE_SPEC/research/dflash_kplus_20261103/reviews/REPORT_RECOVERY_CPU_001.log 2>&1
```

All **20 tests passed in 1.204 s**, exit 0. [Retained log](REPORT_RECOVERY_CPU_001.log). The original operational failure remains in [failure_001](EXECUTION_AND_PROVENANCE.md). No endpoint efficacy conclusion follows from this CPU repair qualification.
