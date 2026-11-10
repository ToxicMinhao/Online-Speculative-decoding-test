# Qualification checker repair and explicitly authorized rerun

The first native attempt at `8862bc3161c281d11eb21d619bd272486558a80f` stopped in the first ordinary-KPLUS case. It completed one disposable Adam step, then passed a dictionary-valued endpoint to a tensor-only finite checker. This raised `AttributeError: 'dict' object has no attribute 'detach'`. The ABT two-view path was not installed in that case. No main training, qualification serving or augmented comparison occurred; no performance conclusion follows.

The old endpoint was not saved: persistence followed the failing validation. Its logs, work ledger, exact-resume/startup receipts and four prepared feedback packets remain. The endpoint is explicitly unavailable, not archived. The user authorized one bounded rerun after repairing this error, with the failed work counted separately.

## Narrow changes

- Walk nested endpoint containers and validate their tensor leaves using the unchanged tensor finite checker.
- Save the endpoint, identity and captured observations before post-step accounting, finite checks and comparisons. Record finite status as unchecked until validation completes, and retain the endpoint on a subsequent failure.
- Use a new sealed run/source destination. Preserve the failed execution unchanged and authenticate the original requests, model/runtime pins, recovery origins and unmodified production code.
- Reuse the four valid original feedback/teacher packets; this attempt collects none during qualification. The main campaign still generates only the remaining 508 responses, giving 512 unique feedback responses overall.
- Keep the new qualification bounded at 12 disposable Adam steps, 68 draft forwards/backwards and eight serving checks. There is no automatic retry loop.

No objective, mask, radius, precision, tolerance, optimizer, update schedule, panel, serving configuration or primary comparison changes. The historical tensor-only helper and native K+ training remain unchanged.

## Accounting across attempts

| Work | Failed first attempt | Authorized rerun | Main campaign |
|---|---:|---:|---:|
| Disposable Adam steps | 1 | 12 | 0 |
| Committed Adam steps | 0 | 0 | 2,048 |
| Draft forwards/backwards, including qualification startup/resource | 4 / 4 | 68 / 68 | 7,168 / 7,168 |
| Newly generated shared feedback | 4 | 0 | 508 |
| Qualification serving checks | 0 | 8 | 0 |
| Endpoint serving generations | 0 | 0 | 4,096 |

Main startup adds 32 forwards/backwards and fixed diagnostics add 128 forwards, as originally planned. Thus the prospective combined draft total is 7,400 forwards / 7,272 backwards and 2,061 Adam steps, including 13 disposable steps across attempts. Extra teacher/head chunk and recomputation work remains separately measured. Nested preparation/startup timers are disclosed as overlapping, not summed as independent cost.

## Verification and status

CPU coverage includes real nested BF16 endpoint tensors, nonfinite leaves, unchanged state/RNG, and persistence on finite, accounting and comparison failures. Reuse tests reject changed packet bytes, requests, settings and unrelated production changes, and verify no serving backend is constructed by the reuse path. All 109 CPU regression tests pass on the stable source snapshot identified in `CPU_REPAIR_RECEIPT.json`. The first full regression run exposed an old mock admission receipt missing the new finite-status field; that fixture was updated and a rejection test added. No GPU work was used to repair the tests. A separate same-team source review found no additional blocker; this is not external review.

Native rerun results are pending at this implementation publication. Main training remains conditional on all native qualification checks and fresh storage admission; valid negative or mixed scientific outcomes must complete the unchanged campaign. The rerun's sealed plan records its exact execution commit, while the earlier failure retains its original identity.
