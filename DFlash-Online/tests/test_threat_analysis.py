"""Hand-checkable controls for the prespecified local transfer reduction."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.threatened_repair_replay.analysis import (
    analyze, bootstrap_requests, decide, reduce_probes, summarize_memory,
    summarize_selection, tradeoff, write_report,
)


def probe(rid, start, window, state, prefix, *, valid=15, reused=None):
    return dict(request_id=rid, canonical_start=start, round_ordinal=start,
                window=window, state=state, accepted_prefix=prefix,
                valid_proposals=valid, first_error=prefix + 1 if prefix < valid else None,
                valid=True, reused_from=reused)


def panel(patterns, windows=(8, 16)):
    """patterns maps request -> per-start lists of P/N/U/T by window."""
    sealed, rows = [], []
    for rid, starts in patterns.items():
        for ordinal, entries in enumerate(starts):
            sealed.append(dict(request_id=rid, canonical_start=ordinal, round_ordinal=ordinal))
            for window, values in zip(windows, entries):
                rows.extend(probe(rid, ordinal, window, state, value)
                            for state, value in zip(("P", "N", "U", "T"), values))
    return sealed, rows


def selection(rid="m", start=0, slot=2, window=8, p=2, n=1, *, u=True, t=True, excluded=False):
    eligible = not excluded and p >= slot
    return dict(request_id=rid, canonical_start=start, proposal_slot=slot, window=window,
                a_P=p, a_N=n, excluded_current=excluded, currently_accepted=eligible,
                threat=eligible and n == slot - 1, selected_U=u, selected_T=t,
                P_direct_correct=p >= slot, N_direct_correct=n >= slot,
                P_margin=1 if p >= slot else -1, N_margin=1 if n >= slot else -1,
                P_wrong_tie=False, N_wrong_tie=False)


def memory_rows(selections, values=None):
    result = []
    values = values or {}
    for selected in selections:
        if not (selected["selected_U"] or selected["selected_T"]):
            continue
        prefixes = values.get((selected["window"], selected["request_id"], selected["canonical_start"]),
                              (selected["a_P"], selected["a_N"], selected["a_N"], selected["proposal_slot"]))
        for state, a in zip(("P", "N", "U", "T"), prefixes):
            row = probe(selected["request_id"], selected["canonical_start"], selected["window"], state, a)
            row["proposal_slot"] = selected["proposal_slot"]
            row.update(direct_correct=a >= row["proposal_slot"], margin=1 if a >= row["proposal_slot"] else -1)
            result.append(row)
    return result


def test_equal_request_and_fixed_window_weight_not_pooled_blocks():
    # A has one start and gains 4 in one window, zero in the other: mean 2.
    # B has three starts, all gaining zero: mean 0. Equal-request result is 1.
    sealed, rows = panel({"a": [[(0, 1, 1, 5), (0, 1, 1, 1)]],
                          "b": [[(0, 1, 1, 1), (0, 1, 1, 1)]] * 3})
    summary, requests, windows, blocks = reduce_probes(rows, sealed, windows=(8, 16))
    assert summary["contrasts"]["T_N"]["estimate"] == 1
    assert summary["contrasts"]["T_N"]["exact"] == {"numerator": 1, "denominator": 1}
    assert requests[0]["T_N"] == 2 and requests[1]["T_N"] == 0
    assert [row["T_N"] for row in windows] == [2, 0]
    assert len(blocks) == 8
    assert sum(row["T_N_net"] for row in blocks) / len(blocks) == .5


def test_integer_acquisition_loss_and_displaced_blocker():
    row = tradeoff({"P": 4, "N": 7, "U": 3, "T": 4}, 15)
    assert row["N_acquired"] == 3
    assert row["U_lost"] == 1
    assert row["T_N_net"] == -3
    assert row["T_N_acquisition_difference"] == -3
    assert row["T_N_loss_difference"] == 0
    assert row["T_displaced_native_blocker_repair"] == 1
    assert row["U_displaced_native_blocker_repair"] == 1
    terminal = tradeoff({"P": 3, "N": 3, "U": 2, "T": 3}, 3)
    assert terminal["pre_has_blocker"] == terminal["T_displaced_native_blocker_repair"] == 0
    assert terminal["U_earlier_error"] == 1


def test_shared_bootstrap_draws_sorted_and_paired():
    rows = [dict(request_id="b", T_N=-2, T_U=4), dict(request_id="a", T_N=1, T_U=-2)]
    result = bootstrap_requests(rows, draws=1000, seed=1707)
    indices = np.random.Generator(np.random.PCG64(1707)).integers(0, 2, (1000, 2))
    native = np.asarray([1., -2.])[indices].mean(axis=1)
    uniform = -2 * native
    assert result["request_order"] == ["a", "b"]
    assert result["intervals"]["T_N"]["lower"] == np.quantile(native, .025, method="linear")
    assert result["intervals"]["T_U"]["upper"] == np.quantile(uniform, .975, method="linear")
    assert result == bootstrap_requests(list(reversed(rows)), draws=1000, seed=1707)


def test_probe_grid_rejects_missing_duplicate_invalid_or_boundary_change():
    sealed, rows = panel({"a": [[(1, 2, 2, 3), (1, 2, 2, 3)]]})
    mutations = [lambda r: r.pop(), lambda r: r.append(deepcopy(r[0])),
                 lambda r: r[0].update(valid=False), lambda r: r[0].update(first_error=7),
                 lambda r: r[0].update(valid_proposals=14), lambda r: r[0].update(accepted_prefix=16)]
    for mutate in mutations:
        changed = deepcopy(rows)
        mutate(changed)
        with unittest.TestCase().assertRaises(ValueError):
            reduce_probes(changed, sealed, windows=(8, 16))


def test_explicit_reuse_cannot_invent_different_outcome():
    sealed, rows = panel({"a": [[(1, 2, 2, 3), (1, 2, 2, 3)]]})
    rows[3]["reused_from"] = "N"
    with unittest.TestCase().assertRaisesRegex(ValueError, "Reused"):
        reduce_probes(rows, sealed, windows=(8, 16))


def test_selection_population_and_funnel_distinguish_earlier_break():
    rows = [selection(), selection("b", p=3, n=0, u=False, t=False),
            selection("c", p=0, n=0, u=False, t=False),
            selection("d", p=3, n=1, u=False, t=False, excluded=True)]
    result = summarize_selection(rows, windows=(8,))
    window = result["per_window"][0]
    assert window["all_tags"] == 4
    assert window["current_response_exclusions"] == 1
    assert window["currently_accepted"] == 2
    assert window["already_incorrect"] == 1
    assert window["threats"] == window["earlier_breaks"] == 1
    assert window["earlier_break_j_wrong"] == 1 and window["earlier_break_j_correct"] == 0
    assert window["overlap"] == 1
    assert not result["opportunity_placement_pass"]


def test_position_matched_control_and_budget_are_required():
    rows = [selection(u=False), selection("b", slot=3, p=3, n=3, u=True, t=False)]
    with unittest.TestCase().assertRaisesRegex(ValueError, "positional"):
        summarize_selection(rows, windows=(8,))
    rows = [selection(u=False, t=False)]
    with unittest.TestCase().assertRaisesRegex(ValueError, "budget"):
        summarize_selection(rows, windows=(8,))


def test_distinct_support_not_repeated_occurrences():
    rows = [selection(window=w) for w in (8, 16, 24, 32)]
    support = summarize_selection(rows, windows=(8, 16, 24, 32))
    assert support["selected_T_occurrences"] == 4
    assert support["distinct_selected_T_tags"] == 1
    assert support["active_windows"] == 4
    assert support["nonidentical_selection_windows"] == 0
    assert not support["opportunity_placement_pass"]


def test_selected_memory_rescue_ties_and_earlier_break_are_separate():
    selected = [selection()]
    rows = memory_rows(selected)
    # Native rejects at slot 2 via a native wrong-at-tie decision.
    rows[1]["margin"] = 0
    # Uniform has an earlier blocker, but predicts the tag correctly.
    rows[2].update(accepted_prefix=0, first_error=1, direct_correct=True, margin=1)
    result = summarize_memory(rows, selected)
    assert result["manipulation_pass"]
    assert result["selected_T"]["N_wrong_tie"] == 1
    assert result["selected_T"]["U_earlier_break"] == 1
    assert result["selected_T"]["U_direct_correct"] == 1
    assert result["selected_T"]["T_N_rescue"] == result["selected_T"]["T_U_rescue"] == 1


def test_memory_p_n_must_reproduce_selection_and_no_nan():
    selected = [selection()]
    rows = memory_rows(selected)
    rows[0].update(accepted_prefix=3, first_error=4)
    with unittest.TestCase().assertRaisesRegex(ValueError, "selection"):
        summarize_memory(rows, selected)
    rows = memory_rows(selected)
    rows[2]["margin"] = float("nan")
    with unittest.TestCase().assertRaisesRegex(ValueError, "margin"):
        summarize_memory(rows, selected)


def test_zero_threat_window_stays_in_primary_and_requires_fallback():
    sealed, rows = panel({"eval": [[(1, 2, 2, 3), (1, 2, 2, 2)]]})
    selected = [selection(window=8), selection(window=16, n=2, u=False, t=False)]
    summary, *_ = analyze(rows, selected, memory_rows(selected), sealed_probes=sealed,
        windows=(8, 16), validity=dict(complete=True, all_required_checks_passed=True), work={}, require_full_population=False)
    assert summary["primary"]["contrasts"]["T_N"]["estimate"] == .5
    assert summary["active_window_secondary"]["contrasts"]["T_N"] == 1
    assert summary["decision"] == "INSUFFICIENT_OPPORTUNITY_OR_PLACEMENT"
    rows[-1].update(accepted_prefix=3, first_error=4)
    with unittest.TestCase().assertRaisesRegex(ValueError, "no-threat"):
        analyze(rows, selected, memory_rows(selected), sealed_probes=sealed, windows=(8, 16),
                validity=dict(complete=True, all_required_checks_passed=True), work={}, require_full_population=False)


def test_decision_is_joint_strict_and_validity_first():
    primary = {"contrasts": {"T_N": dict(lower=.1, upper=.5), "T_U": dict(lower=.1, upper=.5)}}
    support, memory = dict(opportunity_placement_pass=True), dict(manipulation_pass=True)
    valid = dict(complete=True, all_required_checks_passed=True)
    assert decide(primary, support, memory, valid) == "LOCAL_ADVANCEMENT_SCREEN_PASSED"
    primary["contrasts"]["T_U"]["lower"] = 0
    assert decide(primary, support, memory, valid) == "REPLAY_BENEFIT_WITHOUT_DEMONSTRATED_TARGETING_ADVANTAGE"
    primary["contrasts"]["T_N"]["lower"] = -.1
    assert decide(primary, support, memory, valid) == "UNRESOLVED_LOCAL_TRANSFER"
    primary["contrasts"]["T_U"].update(lower=-.5, upper=-.1)
    assert decide(primary, support, memory, valid) == "THREAT_SELECTION_WORSE_THAN_UNIFORM"
    assert decide(primary, support, memory, dict(complete=False)) == "INCOMPLETE"
    assert decide(primary, support, memory, dict(complete=True, all_required_checks_passed=False)) == "INVALID"


def test_support_pass_requires_distinct_tags_requests_and_placement():
    # Four active windows select disjoint groups of four threatened tags. Every
    # position-matched uniform set uses accepted nonthreats instead.
    windows = (8, 16, 24, 32)
    rows = []
    for window in windows:
        for group in range(4):
            for ordinal in range(4):
                rid = f"m{group * 4 + ordinal:02d}"
                is_active = window == windows[group]
                rows.append(selection(rid, window=window, p=2, n=1 if is_active else 2,
                                      u=False, t=is_active))
        for ordinal in range(4):
            rows.append(selection(f"u{ordinal}", window=window, p=2, n=2, u=True, t=False))
    result = summarize_selection(rows, windows=windows)
    assert result["opportunity_placement_pass"]
    assert result["distinct_selected_T_tags"] == result["distinct_selected_T_requests"] == 16
    assert result["active_windows"] == result["nonidentical_selection_windows"] == 4


def test_incomplete_report_preserves_rows_without_partial_primary_or_private_fields():
    sealed, rows = panel({"eval": [[(1, 2, 2, 3), (1, 2, 2, 3)]]})
    rows = rows[:3]
    rows[0].update(private_path="secret", tokens=[100, 101])
    with tempfile.TemporaryDirectory() as directory:
        result = write_report(directory, probe_rows=rows, selection_rows=[], memory_rows=[], sealed_probes=sealed,
            windows=(8, 16), validity=dict(complete=False, errors=["budget exhausted"]), work={"forwards": 3}, execution_commit="test")
        assert result["decision"] == "INCOMPLETE" and "primary" not in result
        data = Path(directory, "PROBE_OUTCOMES.csv").read_text()
        assert len(data.splitlines()) == 4
        assert "secret" not in data and "tokens" not in data
        assert not Path(directory, "PER_REQUEST.csv").exists()
        assert "No completed-panel efficacy estimate" in Path(directory, "REPORT.md").read_text()


def test_invalid_completed_grid_is_published_as_invalid():
    sealed, rows = panel({"eval": [[(1, 2, 2, 3), (1, 2, 2, 3)]]})
    with tempfile.TemporaryDirectory() as directory:
        result = write_report(directory, probe_rows=rows[:-1], selection_rows=[], memory_rows=[], sealed_probes=sealed,
            windows=(8, 16), validity=dict(complete=True, all_required_checks_passed=True), work={}, execution_commit="test",
            require_full_population=False)
        assert result["decision"] == "INVALID" and "primary" not in result
        assert "probe grid" in result["analysis_error"]
        assert json.loads(Path(directory, "PRIMARY_RESULTS.json").read_text())["status"] == "invalid"


def test_production_report_cannot_pass_on_a_shrunken_plan():
    sealed, rows = panel({"eval": [[(1, 2, 2, 3), (1, 2, 2, 3)]]})
    with tempfile.TemporaryDirectory() as directory:
        result = write_report(directory, probe_rows=rows, selection_rows=[], memory_rows=[], sealed_probes=sealed,
            windows=(8, 16), validity=dict(complete=True, all_required_checks_passed=True), work={}, execution_commit="test")
        assert result["decision"] == "INVALID" and "primary" not in result
        assert "64 requests" in result["analysis_error"]


def test_funnel_reports_reachable_earlier_break_and_native_ties():
    rows = [selection(), selection("b", p=3, n=0, u=False, t=False),
            selection("c", p=1, n=1, u=False, t=False)]
    rows[0].update(N_margin=0, N_wrong_tie=True)
    rows[1].update(N_direct_correct=True, N_margin=0)
    summary = summarize_selection(rows, windows=(8,))["per_window"][0]
    assert summary["earlier_break_j_correct"] == 1
    assert summary["eligible_denominator"] == 2
    assert summary["eligible_N_ties"] == 2
    assert summary["eligible_N_wrong_ties"] == 1
    assert summary["already_wrong_denominator"] == 1
    rows[0]["N_wrong_tie"] = False
    with unittest.TestCase().assertRaisesRegex(ValueError, "tie metadata"):
        summarize_selection(rows, windows=(8,))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
