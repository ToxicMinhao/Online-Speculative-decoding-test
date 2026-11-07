# Native qualification evidence audit

Verdict: **passed, with the evidence limits below**. No unresolved qualification or admission blocker was found. This is a local review of retained receipts and published execution source `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`; no GPU work, remote operation, numerical replay, or experiment was performed.

90 consistency checks passed across 46 retrieved files. The audit independently reconstructed all 60 `WORK.jsonl` records: exactly 12 response/draft forwards, 12 completed backwards and six Adam updates, with no unmatched attempt or retry. Every response backward completes before the next response begins. The two fixed pairs are reused independently in all three arms; the other two authenticated inventory packets are not presented.

Each case has matching initial, reset-before-decode and restored-initial receipts: zero counters, empty accumulation, fresh Adam, cleared gradients and exact state equality. Every case records a positive BF16 parameter change, all 58 Adam entries at step one, finite retained scalar metrics, and frozen target-derived parameter isolation. MIX/ROLE report zero nonfinite probability entries. Serialized state was compared, the trainer reset, decoded state loaded and compared again, and then the original initial state restored. These numerical facts are receipt-based, not independently repeated tensor comparisons.

| Case | Changed BF16 coordinates | Wall seconds | Peak allocated GiB | Head calls incl. recomputation |
|---|---:|---:|---:|---:|
| KPLUS_pair_01 | 717,960,027 | 47.102 | 15.359 | 392 |
| KPLUS_pair_02 | 720,361,705 | 40.919 | 19.048 | 232 |
| MIX_pair_01 | 721,489,086 | 43.780 | 21.701 | 392 |
| MIX_pair_02 | 721,745,538 | 41.826 | 18.329 | 232 |
| ROLE_pair_01 | 719,456,360 | 42.412 | 21.585 | 392 |
| ROLE_pair_02 | 720,722,589 | 41.511 | 19.048 | 232 |

Total qualification wall time is **377.465 s**. The six nested case timers sum to 257.549 s; they omit portions of construction and global bookkeeping. Maximum recorded allocated memory is **23,300,741,120 bytes** (23.300741120 GB, 21.701 GiB). There are 60 layer forwards and 1,872 head calls including recomputation. Per-pair physical counts match KPLUS exactly in MIX and ROLE: 10 layer calls and 392/232 head calls per case. The published code constructs a fresh trainer for every case, bounds attempted work before execution, performs no startup response passes or complete generations, deletes disposable states after checks, and cannot pass qualification after a cleanup exception.

All nine saved input identities match the sealed dependency pins and the final summary. The final receipt says the source inputs were rehashed unchanged. The retained compatibility receipt records `donated_buffer=False` before all six fresh constructions. Reviewed local qualifier, preflight, campaign and launcher source match the published execution commit. The combined CPU log contains both passing invocations: 41 tests in 0.677 s and 77 tests in 43.212 s. Independently recounting their named tests confirms 20 overlap, leaving 21 retained regression tests beyond the 77 new tests (98 distinct tests overall).

Both qualification and main preflight receipts passed with zero model work, 3,847 verified source/input files, clean pinned source, idle GPUs, no foreign workers or transfers, and released-state equality. Main admission has **149,583,163,392 bytes free** against **149,061,682,562 required**, leaving 521,480,830 bytes above the conservative budget; the required amount already includes the unchanged 8 GiB reserve. The standalone storage receipt matches the main preflight exactly. These receipts establish admission at that time, not current storage or later campaign completion.

All six retirement identities match their decoded-state results; cumulative retired disposable size is 37,751,086,338 bytes. The exact paths and hashes are retained in the companion JSON. Production reuse is false in the terminal receipt and excluded by the reviewed production startup path. No disposable numerical state is retained in the retrieved package.

Evidence limits:

- Evidence and source audit only: no GPU, remote access, model work, or replay performed by this reviewer.
- Exact BF16 weights, Adam moments, gradients, decoded payload and RNG comparisons rely on the execution receipts and reviewed code. Disposable numerical tensors were retired, not archived; they cannot be independently recomputed from this retrieved package.
- Remote original inputs were not rehashed by this audit. All nine saved identities independently match sealed dependency pins; unchanged-after-execution is an execution receipt.
- Maximum recorded GPU memory is torch.cuda.max_memory_allocated after resetting peak statistics following each fresh trainer construction. Construction transient peaks and reserved memory are not separately measured.
- Retirement free-before/free-after values are equal in all six receipts, so the receipts support unlink completion and lost replay capability, not a measured immediate free-space gain. The cumulative retired byte count is not a simultaneous peak-storage reduction.
- PROGRESS.json intentionally remains running and PLAN.json is the pre-execution plan; QUALIFICATION.json is the terminal status, and all six progress case receipts match it.
- No efficacy, full-trajectory stability, endpoint accuracy, or production success is established by these fixed qualification cases. This reviewer authored evaluation/diagnostics/report modules, but not the reviewed qualifier or admission implementation.

Evidence: `/home/minhao2/EAGLE_SPEC/research/dflash_role_loss_pilot_20261107/qualification_review_001`. Detailed checks, file hashes, per-case work, retirement identities and sealed input bindings: [QUALIFICATION_AUDIT.json](QUALIFICATION_AUDIT.json). The public qualification report was checked for consistency with this evidence; no factual correction was required.
