# U/P implementation: reading and review guide

**Status: implementation and CPU review only.** The API example below documents a possible later run. It has not been executed and does not authorize GPU qualification, training or generation. Completed O artifacts remain unchanged.

## What changed

U and P retain ordinary OSD forward KL at every supervised position and add a strongest-current-rival ranking term. P applies it to the event reference's consecutive matching prefix; U spreads the same total weighted coefficient over every supervised position in that response presentation. Both default to lambda 1. Lambda 0 uses the exact ordinary O path without creating a reference.

The ordinary sampler, full-block attention, teacher conditioning, denominator, response averaging, BF16 optimizer and update schedule remain the foundation. The added reference runs first with the same anchors and released weights fixed for the event. It supplies a detached mask, not training labels or a serving checkpoint.

## Read the code in this order

Source paths below are relative to the A800 production root `/data/sources/minhao/DFlash-Online`. The same review files are under this directory's `worktree` snapshot.

| File / entry | What to inspect |
|---|---|
| `online_feedback/ranking.py`: `matching_prefix_mask`, `allocation_coefficients`, `rival_softplus` | The entire new mathematical component: prefix boundaries, response-wide rho, exact rival and differentiable logits |
| `online_feedback/objective.py`: `TargetObjectiveModel.begin_event`, `_reference_prefix`, `_forward_ranking`, `_chunk_loss`, `end_event` | Independent event reference; reference-first execution; chunked head projection; unchanged KL plus auxiliary numerator; shared denominator and reference cleanup |
| `online_feedback/presentation_schedule.py`: `PresentationSchedule` | The retained O receipt is checked against normal shuffle/sampler draws; it does not replace the sampler or silently resample |
| `online_feedback/training.py`: `SpecForgeEventTrainer.__init__`, `_train_passes`, `train_event` | Objective selection, verified EOS metadata, original optimizer lifecycle, optional retained schedule check, independent output paths |
| `tests/test_ranking.py`, `test_ranking_objective.py`, `test_ranking_training.py`, `test_ranking_schedule.py` | Boundary/gradient fixtures, wrapper integration, lambda-zero state equivalence, receipt and schedule mismatch detection |

Useful breakpoints are `_forward_ranking` immediately after the single anchor draw, `_reference_prefix` after constructing each prefix mask, and `_chunk_loss` where `combined = loss_sum + ranking_lambda * ranking_sum`. The first shows shared block identity; the second shows which previously matching proposals are selected; the third shows the ordinary and additional losses separately.

## Future one-event trainer API

**Illustrative future API only; no auto-launch script is provided.** After review and separate GPU qualification, each arm would construct a new trainer, train exactly one event and close. Use the retained completed O receipt at:

`/data/sources/minhao/runs/dflash_osd_one_event_20261025/campaign_001/trainer/events/event_001/RESULT.json`

Its `features` list contains the exact original prepared tensor paths, for example:

`/data/sources/minhao/runs/dflash_osd_one_event_20261025/campaign_001/event_001/loop/features/train_00000.pt`

Do not relocate those paths silently: schedule validation checks feature identity, metadata and paths as well as order and anchors. Missing feedback is a blocker to report, not permission to regenerate answers. The qualified source configuration is the unchanged 2272-token sealed configuration from the completed O run; the source template's historical 2048-token training value is not a substitute.

```python
# Documentation only. No GPU execution is authorized by this implementation review.
from pathlib import Path
import json
from online_feedback.training import SpecForgeEventTrainer

baseline_root = Path("/data/sources/minhao/runs/dflash_osd_one_event_20261025")
ordinary_receipt = baseline_root / "campaign_001/trainer/events/event_001/RESULT.json"
ordinary = json.loads(ordinary_receipt.read_text())
feedback_paths = [Path(feature["path"]) for feature in ordinary["features"]]

# Use the verified sealed configuration, unchanged, including max_length=2272.
qualified_config = baseline_root / "sealed/config.yaml"
# A new destination; never use baseline_root or O's trainer state as an output.
candidate_root = Path("/data/sources/minhao/runs/dflash_prefix_ranking_one_event_REVIEWED")

# Choose one arm per independent runtime: U = osd_uniform; P = osd_prefix.
arm, objective = "U", "osd_uniform"
trainer = SpecForgeEventTrainer(
    qualified_config,
    candidate_root / arm / "trainer",
    resume_state=None,                 # released draft and fresh optimizer
    reduction="response_mean",
    objective=objective,
    ranking_lambda=1.0,
)
try:
    result = trainer.train_event(
        "event_001",
        feedback_paths,
        passes=2,
        presentation_schedule=ordinary_receipt,
    )
    assert result["response_presentations"] == 256
    assert result["optimizer_steps"] == 128
    assert result["start_step"] == 0 and result["end_step"] == 128
finally:
    trainer.close()
```

P would use `arm="P"`, `objective="osd_prefix"` and a separate fresh runtime/output directory. Do not initialize P from U or O. Both retain seed 17, the released model path, constant LR 3e-4 and the complete qualified configuration. Its historical 7424-step horizon and 58-event maximum do not cause execution by themselves: this API calls one 128-step event only and stops.

The schedule checker preserves normal RNG consumption: the trainer still draws each pass order and each presentation's anchors once, then requires an exact match to O's receipt before the draft forward. Reference evaluation receives those anchors and cannot consume additional sampler draws. Pass 2 keeps its own retained anchor draws rather than reusing pass 1's anchors.

## Review boundaries and next proposed gate

CPU tests establish the covered formulas, wrapper behavior and small-fixture state contracts. They do not establish native GPU memory fit, compiled Flex behavior, output preservation or an algorithmic benefit.

The proposed GPU qualification is five discarded trials O/U(0)/P(0)/U(1)/P(1), two retained presentations each: **5 temporary optimizer steps, 10 current forward/backward presentations, 4 reference forwards, and 4 additional detached matched-mode forwards; zero generations and zero teacher reconstruction**. This is proposed work, not executed work. Full details and the later one-event comparison budget are in [HANDOFF_AUDIT.md](HANDOFF_AUDIT.md).

Preserve O's completed results and checkpoint throughout. Any future F/O/U/P latency comparison must measure all four freshly together; historical O timing must not be mixed with new candidate timing. No performance claim follows from this code review alone.

Qualification scheduling clarification: the two presentations are the first two rows of the full retained 128-response pass, including its original 128-element shuffle and subsequent anchor RNG draws. Do not replace this with a newly shuffled two-response pool. Stop the disposable trial after the first complete accumulation window and restore state between trials.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_implementation_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
