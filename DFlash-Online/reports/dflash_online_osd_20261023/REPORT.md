# OSD baseline implementation in DFlash-Online

**Subsequent execution completed, October 23:** the authorized matched follow-up
passed native qualification in all three groups (96 generations, 24 optimizer
steps). Held-out calls increased 9.28%; source gains were concentrated in ShareGPT.
See the [experiment report](execution_001/RESULTS.md). The implementation-stage
report below is retained as history; its native-qualification-pending status is
superseded by this completed run.

The coding survey and minimal implementation are complete. The reviewed changes
are installed at **A800 `/data/sources/minhao/DFlash-Online`**. All **147 CPU tests
pass on both Fenrir and A800**, including 14 new OSD tests. No full-model GPU
qualification, new answer generation or scientific training campaign ran.

## Survey result

AdaFlash and our training code share SpecForge ancestry. Its OSD recipe supplies
the needed baseline: pure teacher-to-draft forward KL, temperature 1 and no
hard-label mixture. Our previous current default is hard-token CE; it should not
be relabeled OSD. The buffer, replay, training core and publication already exist.
The [survey](CODING_SURVEY.md) pins the author code and lists adaptations.

The only important strategy difference is that newer SpecForge normally treats
final teacher features as optional diagnostics. OSD requires them as training
inputs even when diagnostic metrics are off. A short external strategy handles
that without changing SpecForge itself.

## How to select and inspect it

```python
trainer = SpecForgeEventTrainer(
    config_path, output_root, objective="osd", reduction="response_mean",
)
```

The existing controller continues to serve, queue completed responses, prepare
target features, train and publish. For prediction position j, teacher hidden
state j-1 is projected through the frozen vocabulary head. The current draft
predicts j under the existing masked-block conditioning. Only response positions
receive weighted forward KL. Full teacher distributions are reconstructed in
bounded chunks; there is no pre-event reference draft or q0 for this objective.

| A800 file beneath DFlash-Online | Change |
| --- | --- |
| `online_feedback/objective.py` | OSD loss, teacher alignment and always-on teacher input strategy |
| `online_feedback/training.py` | Objective registration, teacher feature validation and resume identity |
| `online_feedback/qualify.py` | `--objective osd`; restart keeps objective and reduction |
| `tests/test_osd_objective.py`, `test_osd_strategy.py`, `test_osd_trainer.py` | 14 focused checks |
| `README.md` | Usage, loss definition, differences from author code and cost interpretation |

The three production files grew by 82 lines combined. Neither upstream
`inference/` nor `training/` was edited; Fenrir DFlash-Research was not modified.
The [local review copy](worktree/online_feedback/objective.py) and
[complete patch](IMPLEMENTATION.patch) mirror the installed files. Local imported
inference source is a read-only copy of A800's existing runtime, not a new patch.

## Correctness evidence

The new tests cover independent per-position loss/gradient calculation, target
predecessor alignment, prompt/anchor/tail masks, chunk invariance, RNG stability,
teacher isolation, metrics-disabled forwarding, both response reductions, actual
trainer wiring, malformed teacher rejection and checkpoint identity. Tiny real
DFlash blocks, BF16 parameters, SpecForge TrainerCore and its real optimizer are
used. Full-model loading/distributed wrapping/export are replaced only in the
small CPU integration fixture.

The restart test reproduces the next event's weights, Adam state, RNG, counters
and sampled anchors exactly. An OSD checkpoint is refused when resuming CE.
Two delegated source reviews found no blocking math or conditioning defect.

- [Fenrir test log](CPU_TESTS.log): 147 tests, 5.966 s, pass.
- [A800 test log](A800_CPU_TESTS.log): 147 tests, 5.590 s, pass; CUDA hidden from test process.

One initial local whole-suite invocation accidentally imported Fenrir's unchanged
inference checkout rather than A800's already-qualified cache/trace implementation.
It failed 6 assertions and 12 error cases in old cache/replay tests. The retained
[log](CPU_TESTS_WRONG_INFERENCE_PATH.log) records this harness mistake. Correcting
the import to a copy of actual A800 inference resolved it without source changes
or relaxed tests. The OSD tests passed in both invocations.

## Interpretation and cost boundary

Name this **OSD-style teacher forward-KL under DFlash-Online**. We preserve our
newer sampler, configured anchors, FP32 objective arithmetic, FP32-master optimizer
and explicit response reduction. This is not bitwise AdaFlash BF16 reproduction.
No learning rate or buffer size is silently selected by the new objective.

Teacher replay/preparation remains separately measured. Teacher vocabulary-head
work, including checkpoint recomputation, is a component of existing training
time; it is not added a second time to totals. OSD does not make feedback free.

These tests establish implementation behavior on small CPU cases. They do not
establish full Qwen GPU memory fit, updated native output preservation, learning
effectiveness or serving acceleration. The next execution step would be the
existing native qualification with `--objective osd`, after full checkpoint-overlap
storage admission. The earlier hard-CE qualification is not an OSD qualification.
No existing paused campaign was resumed and no mechanism candidate was implemented.

## Storage and provenance

Before installation A800 had 13,455,855,616 free bytes; immediately afterward
13,455,429,632 bytes. The admitted source/test-write budget was 20 MiB plus the
existing 8 GiB reserve. Both GPUs were idle before work. Fenrir had about 80 GiB
free; the user's existing debugger process was left untouched.

Only seven selected files were installed, verified against the surveyed originals
and transfer manifest. Original source is retained locally in `before/source.tar`
and remotely in `/data/sources/minhao/runs/dflash_online_osd_20261023/before_install.tar`.
Installation receipt: [INSTALL.json](INSTALL.json). No research artifacts, caches,
checkpoints or recovery state were removed. One malformed local temporary source
transfer was discarded after a valid replacement was saved; no evidence was lost.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
