# Evidence access and replay limits

Execution commit: `8a4cbcc8b78adfee845a0a1409db4a0d29c54058`.

The public package contains all 2,304 final scalar rows, all 3,072 pre-update scalar rows, 256 endpoint pairs, exact selection/insertion/pass-binding ledgers, per-pass training exposure, cost/retirement records and separate CPU review outputs/code. Complete original input/output token arrays, native JSON sidecars, descriptors, anchors and RNG receipts are in the private hash-verified compact archive identified by PROVENANCE.json. Raw response text and tensors are excluded from Git. Original relative links inside copied reviews resolve in the private task workspace; see reviews/README.md.

The original generated numerical report is preserved unchanged as GENERATED_REPORT_ORIGINAL.md. REPORT.md supersedes its point-sign-only C recommendation with the complete evidence review; numerical outputs are unchanged.

Final tensor states, serving exports and the128-record final reservoir remain on A800. Intermediate tensors explicitly listed in RETIREMENT_PATHS.csv were deleted after validation, not archived. The compact archive is sufficient for the included CPU record audits, but cannot restore retired native features or earlier full optimizer states.

From the directory containing audit_tools, using the verified private run root:

```sh
/home/minhao2/.venvs/dflash-specforge/bin/python audit_tools/audit_track2.py /home/minhao2/EAGLE_SPEC/research/dflash_k_workload_replay_v1_20261102/retrieved_001/production_001 --output NEW_REVIEW_DIRECTORY
python audit_tools/summarize_cost_retention.py --run-root /home/minhao2/EAGLE_SPEC/research/dflash_k_workload_replay_v1_20261102/retrieved_001/production_001 --output NEW_COST_DIRECTORY
```

These are local CPU evidence-analysis commands, not authorization to rerun GPU training/generation. Reports record ANALYZED status and disclosed same-team authorship overlap, not independent GPU replication.
