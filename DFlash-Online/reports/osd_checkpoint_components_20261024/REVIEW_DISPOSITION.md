# Independent internal review disposition

ARS experiment-agent implementation review, October 24, 2026. Reviews were
internal agents, not external peer review. All changes are experiment controls;
no inference algorithm or training recipe changed.

| Finding | Resolution / evidence |
|---|---|
| Prepared checkpoint can change before execution | Seal ARTIFACT_IDENTITIES plus construction receipts; hash actual loaded files once before engine startup; mutation and changed-path fixtures |
| Added target/checkpoint files bypass listed-file hashes | Require exact top-level model file membership; added-generation-config regression |
| Initial F skips publication | Force publication on first use; failure fixture proves no generation occurs |
| Native equality identity optional | Require actual checkpoint_path returned by installed hook; missing/wrong fixtures |
| Structural failures could continue after qualification | Persist row then stop; five post-qualification structural fixtures; coherent isolated output mismatch remains partial |
| Unstable ratio could trigger descriptive 80% screen | None for unstable/undefined denominator; preserve raw ratios and denominator states |
| Analysis trusts impossible token counts | Independently check accepted<=progress and progress=output_tokens-1 |
| Between-operation deadline can overrun | Shell timeout 7200s, SIGTERM cleanup, 120s kill grace; native 600s operation bound unchanged |

322 CPU tests passed in actual A800 environment; 46 are new component tests.
Native output preservation and publication behavior on hybrids are still untested
on GPU. Final O lacks a historical file digest; current identity is bound to the
retained event-58 receipt and must reproduce historical output/count evidence.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_checkpoint_components_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
