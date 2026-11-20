# Archive verifier CPU review

The portable verifier passes **23 focused CPU tests**. Actual archive delivery remains **pending** until the completed transfer passes the verifier against its authenticated member manifest and remote transport identity.

A separate local agent reviewed the implementation and wrote 21 corruption, coverage and keeper-chain fixtures. Review found and resolved retirement-companion counting, missing tar-footer checks, and exact predecessor-path binding. The implementation author then added two tests for supplemental external keeper evidence. This is independent local code review, with the stated implementation overlap; it is not independent GPU replication.

The verifier checks every regular archive member without extraction, rejects missing/extra/duplicate or unsafe members, validates compression-frame completion and tar end blocks, and reconciles all 58 event boundaries and the 7,424 preparation-retirement records. Retired tensors remain explicitly unavailable. Four current C tensor paths represent three physical retained files. The exact old KPLUS58, original K0 and seven runtime overlays are checked through a separate current authentication receipt bound to the archived production plan.

Publication changes only make paths portable: the authenticated private manifest supplies the host prefix, and tests use a synthetic prefix. Source commits, checkpoint/configuration hashes, counts, exact-equality gates, retirement logic and accepted failure conditions are unchanged. The verifier has no local-path defaults. Raw receipts and model/tensor data remain private.

The [machine-readable CPU receipt](ARCHIVE_VERIFIER_CPU_REVIEW.json) pins the portable source and tests. Run the test command from the DFlash-Online directory:

```bash
PYTHONPATH=. python -B -m unittest experiments.fresh_mathqa_endpoints.archive_audit.test_verify_archive -v
```

A successful synthetic test run does not permit marking archive closeout complete. The real invocation writes a new `CLOSEOUT.json` only after checking the received archive and writes its scalar evidence reconciliation separately.
