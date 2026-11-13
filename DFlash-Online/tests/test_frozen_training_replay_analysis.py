from copy import deepcopy

import numpy as np
import unittest

from experiments.frozen_training_repair_replay.analysis import (
    bootstrap_requests, outcome, pair_occurrences, summarize,
)
from experiments.frozen_training_repair_replay.report import narrative


def row(rid="a", start=0, slot=2, presentation=0, prefix=2, correct=True, margin=.5):
    return dict(request_id=rid, canonical_start=start, proposal_slot=slot,
        presentation_id=presentation, pass_index=presentation // 128,
        anchor_ordinal=0, parent_margin=.75, floor=.375, valid=True,
        correct_prefix=prefix, native_correct=correct, margin=margin,
        canonical_token=123, rival_token=456, private_path="never publish")


def test_outcome_boundaries():
    for prefix, slot, label in [(0,2,"earlier_break"),(1,2,"first_mismatch_at_tag"),(2,2,"accepts_through"),(0,1,"first_mismatch_at_tag"),(15,15,"accepts_through")]:
        assert outcome(prefix, slot) == label


def test_pairing_keeps_failure_and_private_tokens_out():
    native = [row(prefix=0, correct=True, margin=.5)]
    recent = [row(prefix=1, correct=False, margin=0)]
    result = pair_occurrences(native, recent)[0]
    assert result["native_outcome"] == "earlier_break"
    assert result["native_direct_correct"] == 1
    assert result["recent_wrong_at_tie"] == 1
    assert result["net"] == 0
    assert not {"canonical_token", "rival_token", "private_path"} & result.keys()


def test_singleton_gets_full_slot_weight():
    # Slot a is lost in both layouts, slot b is rescued in its only layout.
    native = [row(presentation=0), row(presentation=128), row("b", presentation=129, prefix=1, correct=False, margin=-1)]
    recent = [row(presentation=0, prefix=1, correct=False, margin=-1),
              row(presentation=128, prefix=1, correct=False, margin=-1), row("b", presentation=129)]
    summary, paired, slots, requests, table = summarize(native, recent, require_full_population=False)
    assert summary["primary"]["net_difference_pp"] == 0
    assert summary["primary"]["rescues"] == summary["primary"]["losses"] == 1
    assert abs(summary["occurrence_summary"]["net_difference_pp"] + 100 / 3) < 1e-12
    assert sum(r["slot_equivalents"] for r in table) == 2
    assert sum(r["paired_occurrences"] for r in table) == 3
    assert sum(r["slots"] for r in requests) == len(slots) == 2
    assert len(paired) == 3


def test_exact_fraction_and_one_pass_reversal_not_selected_away():
    native = [row(presentation=0, prefix=1, correct=False, margin=-1), row(presentation=128)]
    recent = [row(presentation=0), row(presentation=128, prefix=1, correct=False, margin=-1)]
    summary, _, slots, _, _ = summarize(native, recent, require_full_population=False)
    assert slots[0]["rescue"] == slots[0]["loss"] == .5
    assert summary["primary"]["exact_net_difference_pp"] == {"numerator": 0, "denominator": 1}
    assert summary["pass_summaries"]["0"]["net_difference_pp"] == 100
    assert summary["pass_summaries"]["1"]["net_difference_pp"] == -100
    assert not summary["all_retention_tied"]


def test_bootstrap_uses_resampled_slot_denominator_not_request_mean():
    requests = [dict(request_id="a", slots=1, net_equivalents=1),
                dict(request_id="b", slots=3, net_equivalents=-1)]
    result = bootstrap_requests(requests, draws=1000, seed=1707)
    rng = np.random.Generator(np.random.PCG64(1707))
    index = rng.integers(0, 2, size=(1000, 2))
    exact = [100 * sum((1, -1)[i] for i in draw) / sum((1, 3)[i] for i in draw) for draw in index]
    assert result["lower_pp"] == np.quantile(exact, .025, method="linear")
    assert result["upper_pp"] == np.quantile(exact, .975, method="linear")
    assert result["undefined_draws"] == 0


def test_full_population_gate_is_not_replaced_by_small_fixture():
    with unittest.TestCase().assertRaisesRegex(ValueError, "population"):
        summarize([row()], [row()])


def test_invalid_pairs_do_not_enter_analysis():
    cases = [
        (lambda rows: rows.append(deepcopy(rows[0])), "Duplicate"),
        (lambda rows: rows[0].update(presentation_id=1), "identities"),
        (lambda rows: rows[0].update(anchor_ordinal=1), "anchor_ordinal"),
        (lambda rows: rows[0].update(valid=False), "no longer valid"),
        (lambda rows: rows[0].update(margin=float("nan")), "Nonfinite"),
        (lambda rows: rows[0].update(correct_prefix=2, native_correct=False, margin=-1), "directly incorrect"),
        (lambda rows: rows[0].update(correct_prefix=1, native_correct=True), "directly correct"),
    ]
    for mutation, error in cases:
        native, recent = [row()], [row()]
        mutation(recent)
        with unittest.TestCase().assertRaisesRegex(ValueError, error):
            pair_occurrences(native, recent)


def test_all_ties_not_equivalence_and_floor_is_strict():
    native = [row(margin=.375)]
    recent = [row(margin=.375)]
    summary, _, _, _, _ = summarize(native, recent, require_full_population=False)
    assert summary["all_retention_tied"]
    assert summary["primary_interval"]["lower_pp"] == summary["primary_interval"]["upper_pp"] == 0
    assert summary["primary"]["native_below_floor_pct"] == 0
    body = narrative(summary, execution_commit="test", reproduction_command="test-only")
    assert "does not establish equivalence" in body
    assert "+0.50% serving target is not a threshold" in body


def test_impossible_prefix_and_argmax_margin_rejected():
    with unittest.TestCase().assertRaisesRegex(ValueError, "Invalid prefix"):
        outcome(16, 2)
    with unittest.TestCase().assertRaisesRegex(ValueError, "contradicts"):
        pair_occurrences([row()], [row(prefix=0, correct=False, margin=1)])


def test_3x3_table_exposes_earlier_blocker_repair_separately():
    native = [row(prefix=0, correct=True), row("b", prefix=1, correct=False, margin=-.5)]
    recent = [row(prefix=2, correct=True), row("b", prefix=2, correct=True)]
    summary, _, _, _, table = summarize(native, recent, require_full_population=False)
    cells = {(r["native_outcome"], r["recent_outcome"]): r["paired_occurrences"] for r in table}
    assert cells["earlier_break", "accepts_through"] == 1
    assert cells["first_mismatch_at_tag", "accepts_through"] == 1
    assert summary["primary"]["net_difference_pp"] == 100
    assert summary["primary"]["direct_correctness_difference_pp"] == 50


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
