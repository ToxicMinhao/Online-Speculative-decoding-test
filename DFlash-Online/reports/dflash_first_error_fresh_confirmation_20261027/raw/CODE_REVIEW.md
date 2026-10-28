## Material Passport

- Workflow: ARS experiment-agent, bounded implementation and reproducibility review within the authorized run. Verification status: **CPU VERIFIED / native execution pending**.
- Reviewer: independent agent `/root/component_checkpoint_impl`; implementation authored separately.
- Scope: new serving-only O/K evaluator and run-local launch/seal/preflight orchestration. No implementation edits, remote operations, GPU calls, training, or scientific generations by this reviewer.
- Decision: **No unresolved code blocker found in the reviewed versions.** Actual native output/weight correctness remains gated during the authorized execution; CPU fixtures do not establish a performance result.

## Contracts checked

The campaign requires64 requests, seed20261027, O/K only, and384 planned generations. Selection completion, current-study-only freshness, ordered source IDs and the exact NEW_PANEL bytes are bound to the inventory audit before sealing. The amended scope is inputs new to the current O/K study, with older project execution explicitly disclosed; this review does not claim globally unused questions.

The evaluator executes each request at three repeats, alternating O/K with `(ordinal + repeat) % 2`. First O supplies canonical tokens and termination. K-first measurements remain durable and pending until that O arrives. No historical O verification-count gate remains. Native records, loaded checkpoint equality, publication/cache state, exact outputs and termination, scalar/count accounting, and each arm's repeat consistency are checked. First eight requests form a48-generation correctness gate inside384, and harmful but valid results continue.

Returned invalid measurements, canonical-pending measurements, unknown interrupted attempts and unattempted executions are distinct. Raw returned evidence precedes validation. A request enters gain calculations only with all six unique valid O/K/repeat slots; both arms share that cohort. Calls use one repeat-0 trajectory/request. Latency uses the sum of per-request three-repeat means; ratios are pooled totals, not averages of request ratios. Three timings are not independent examples. Loading, switching and weight-validation costs are reported separately. No trainer, optimizer, feedback, reference-generation or extra generation-warmup path is introduced.

All117 prior source hashes independently match the local worktree. Target/released draft are verified against historical exact manifests; O export has nonempty exact retained membership and matching historical digests. K is explicitly a newly captured digest at the fixed completed export path, without a claim of a prior export hash. Exact checkpoint/model file sets and bytes are rechecked by pre/postflight. Only native loaded-tensor equality can establish that the serving worker loaded those exported tensors; it remains an execution gate.

## Findings and disposition

1. Malformed count dictionaries/lists could previously crash the legacy repeat checker and then failure accounting. Fixed by integer validation before count tuples and initializing every appended row with an explicit invalid status. Regression fixture passes for list/dict counts and nonfinite latency; returned evidence and reports survive.
2. Report freshness wording could imply global freshness. Corrected to current-study-only freshness with older exposure disclosed.
3. Initial seal checked only matching O-prefix entries and enumerated files. Fixed with required exact O export membership plus pre/postflight exact checkpoint/model file sets. K digest provenance is explicitly limited to its new seal.
4. The panel could have been sealed alongside a stale selection receipt. Fixed by completion/count/scope checks, exact panel SHA256 and ordered selected-source binding.
5. Backend cleanup exceptions could skip final campaign receipts or mask the original error. Fixed by separately recording cleanup failure, always writing STATUS/COSTS and preserving a prior evaluation exception. Both cleanup-only and evaluation-plus-cleanup fixtures pass.

## Independent CPU evidence

Final command used `/home/minhao2/.venvs/dflash-specforge/bin/python`, `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, and the run-root/worktree/tests/local-training/inference Python paths:

```text
python -m unittest experiments.fresh_confirmation.test_evaluate test_campaign -v
Ran 14 tests in 2.710s — OK
```

This includes a full384-return fake run with balanced96/96 O/K positions, the48-generation embedded gate, delayed K canonical transitions, mismatches and transport interruption, malformed native records, report revalidation and heterogeneous pooled arithmetic. An independent import subprocess verifies evaluator import does not load Torch, SGLang or the trainer. Four additional in-memory wrapper checks passed: unchanged seal, added generation_config rejection, cleanup-only failure receipts, and original-error preservation during cleanup failure. The local selected panel has64 consecutive ordinals and matches its selection SHA256.

No native execution result is reviewed here. This fixed-checkpoint comparison cannot establish adaptation-inclusive speedup, seed robustness, global data novelty, or target-only reference equivalence.

## Reviewed source identities

| Relative to this run directory | SHA256 |
|---|---|
| `campaign.py` | `5e4a74e4fee49fe54adc12c13dd6832580fa125bbff71932c991961f67c92a8c` |
| `preflight.py` | `241670d064bdae40e3c4f147f85931fbfe1c0aa94fa09cf89ce9b7d006e1e5e2` |
| `seal_remote.py` | `af3a40abc40038ce6b054a92909a41ad251bae0aab89af3756a702f003a26b49` |
| `run_campaign.sh` | `c07482377ea45e9d74e9d5dff959dcf0ed1346bb4aa5985c9ee6b46ca256d677` |
| `supervise.sh` | `6393fcb983d095bcf12f3e63a3369fa37e7581c5737b60424693b70df5796092` |
| `test_campaign.py` | `848add0ba423eb307912ba5b897895a0c81cc4454ea38c4e8870e111a2390fc8` |
| `worktree/experiments/fresh_confirmation/__init__.py` | `6f6beae041a927b2fd7e16f48832d5177926afe5ab8ef036f80b50a14cfd8a7b` |
| `worktree/experiments/fresh_confirmation/protocol.py` | `2caa2d7e7e5a9347330e1e3101783a245cf3dd63bf7ba65ac5750e072f8048db` |
| `worktree/experiments/fresh_confirmation/evaluate.py` | `58a8aac52ab3a82193f893cd2e9bb49f35d320d6bfcf6b7b5cc4322b5193eba6` |
| `worktree/experiments/fresh_confirmation/report.py` | `8a5c85acc14370aaf325286a90ab42497caa99d9f3e2a17a5f4ac714ace0b93a` |
| `worktree/experiments/fresh_confirmation/test_evaluate.py` | `77323ce42200d5f03ea6de0be503607eea48b129156103debd953637f0a2928b` |


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_fresh_confirmation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
