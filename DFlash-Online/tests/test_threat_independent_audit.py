"""No model work: independent scalar arithmetic and fail-closed reporting."""
import csv
import hashlib
import json

import numpy as np
import unittest
from pathlib import Path
import tempfile

from experiments.threatened_repair_replay.independent_audit import (
    audit_report, compare, decomposition, paired_intervals, read_csv, recompute,
)


def fixture():
    windows = range(8, 129, 8)
    tags = [dict(request_id=f"m{i % 128:03}", canonical_start=i // 128, proposal_slot=2)
            for i in range(815)]
    probes = [dict(request_id=f"p{i:02}", canonical_start=s, round_ordinal=s)
              for i in range(64) for s in range(4)]
    selections, observations, memory = [], [], []
    for windex, window in enumerate(windows):
        threatened = set(range(windex * 4, windex * 4 + 4)) if windex < 4 else set()
        # All 815 tags are eligible; the uniform set is a fixed hash ordering.
        def uniform_key(i):
            r = tags[i]
            digest = hashlib.sha256(f"trr-v1-U|{window}|2|{r['request_id']}|{r['canonical_start']}".encode()).digest()
            return digest, (r["request_id"], r["canonical_start"], 2)
        uniform = set(sorted(range(815), key=uniform_key)[:len(threatened)])
        for i, tag in enumerate(tags):
            n = 1 if i in threatened else 2
            selected = dict(tag, window=window, a_P=2, a_N=n, excluded_current=False,
                currently_accepted=True, threat=i in threatened, selected_T=i in threatened,
                selected_U=i in uniform, P_direct_correct=True, N_direct_correct=n == 2,
                P_margin=1., N_margin=1. if n == 2 else -1., P_wrong_tie=False, N_wrong_tie=False)
            selections.append(selected)
            if i in threatened | uniform:
                for state, a in zip(("P", "N", "U", "T"), (2, n, n, 2)):
                    memory.append(dict(tag, window=window, state=state, accepted_prefix=a,
                        valid_proposals=15, first_error=a+1, direct_correct=a >= 2,
                        margin=1. if a >= 2 else -1., valid=True, reused_from=""))
        for probe in probes:
            gain = (1 + int(probe["request_id"][1:]) % 2) if windex < 4 else 0
            for state, a in zip(("P", "N", "U", "T"), (2, 3, 3, 3+gain)):
                observations.append(dict(probe, window=window, state=state, accepted_prefix=a,
                    valid_proposals=15, first_error=a+1, valid=True, reused_from=""))
    return observations, selections, memory, probes, tags


def write_csv(path, rows, empty_fields=("window",)):
    fields = tuple(rows[0]) if rows else empty_fields
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_fixture(path, data, published):
    for name, rows in zip(("PROBE_OUTCOMES.csv", "WINDOW_SELECTION.csv", "MEMORY_OUTCOMES.csv",
                           "PROBE_KEYS.csv", "MEMORY_REFERENCE.csv"), data):
        write_csv(path / name, rows)
    (path / "PRIMARY_RESULTS.json").write_text(json.dumps(published))


def test_complete_reduction_has_hand_computed_means_and_support():
    summary = recompute(*fixture())
    assert summary["primary"]["levels"] == {"P": 2., "N": 3., "U": 3., "T": 3.375}
    for contrast in ("T_N", "T_U"):
        assert summary["primary"]["contrasts"][contrast]["estimate"] == .375
        assert summary["primary"]["contrasts"][contrast]["exact"] == {"numerator": 3, "denominator": 8}
    assert summary["primary"]["gross_counts"]["T_N_net"] == 1536
    assert summary["primary"]["decomposition"]["T_N_acquisition_difference"] == .375
    assert summary["active_window_secondary"]["contrasts"]["T_N"] == 1.5
    assert summary["support"]["distinct_selected_T_tags"] == 16
    assert summary["support"]["active_windows"] == 4
    assert summary["support"]["opportunity_placement_pass"]
    assert summary["memory"]["selected_T"]["T_N_rescue"] == 16
    assert summary["decision"] == "LOCAL_ADVANCEMENT_SCREEN_PASSED"


def test_bootstrap_uses_same_whole_request_draws_for_both_contrasts():
    values = {"b": [-2., 4.], "a": [1., -2.]}
    result = paired_intervals(values)
    draws = np.random.Generator(np.random.PCG64(1707)).integers(0, 2, (10000, 2))
    samples = np.array([1., -2.])[draws].mean(axis=1)
    assert result["T_N"]["lower"] == np.quantile(samples, .025, method="linear")
    assert result["T_U"]["upper"] == np.quantile(-2 * samples, .975, method="linear")
    assert result == paired_intervals(dict(reversed(list(values.items()))))


def test_exact_acquisition_loss_identity_and_terminal_blocker():
    row = decomposition(4, 7, 3, 4, 15)
    assert row["T_N_net"] == -3
    assert row["T_N_acquisition_difference"] - row["T_N_loss_difference"] == -3
    assert row["U_lost"] == 1
    assert row["T_displaced_native_blocker_repair"] == 1
    terminal = decomposition(3, 3, 2, 3, 3)
    assert terminal["U_displaced_native_blocker_repair"] == 0
    assert terminal["U_earlier_error"] == 1


MUTATIONS = [
    (lambda d: d[0].pop(), "Probe outcomes"),
    (lambda d: d[1].pop(), "Selection must cover"),
    (lambda d: d[3].pop(), "64 sealed requests"),
    (lambda d: d[4].pop(), "815 sealed tags"),
    (lambda d: d[1][0].update(selected_T=False), "Deterministic replay"),
    (lambda d: d[0][-1].update(accepted_prefix=4, first_error=5), "No-threat window"),
]

def test_incomplete_or_changed_population_cannot_claim_completion():
    for mutation, match in MUTATIONS:
        data = fixture()
        mutation(data)
        with unittest.TestCase().assertRaisesRegex(ValueError, match):
            recompute(*data)


def test_report_csv_roundtrip_and_corruption_is_detected(tmp_path):
    data = fixture()
    summary = dict(recompute(*data), status="completed", validity={"complete": True, "all_required_checks_passed": True})
    write_fixture(tmp_path, data, summary)
    result = audit_report(tmp_path)
    assert result["status"] == "passed"
    assert result["checks"] > 100
    assert read_csv(tmp_path / "WINDOW_SELECTION.csv")[0]["excluded_current"] is False
    summary["primary"]["contrasts"]["T_N"]["estimate"] = 99
    (tmp_path / "PRIMARY_RESULTS.json").write_text(json.dumps(summary))
    result = audit_report(tmp_path)
    assert result["status"] == "failed"
    assert any("contrasts.T_N.estimate" in err for err in result["errors"])


def test_incomplete_report_keeps_raw_rows_without_efficacy(tmp_path):
    data = fixture()
    summary = dict(status="incomplete", decision="INCOMPLETE", validity={"complete": False})
    write_fixture(tmp_path, ([data[0][0]], [], [], data[3], data[4]), summary)
    result = audit_report(tmp_path)
    assert result["status"] == "passed" and result["efficacy_audited"] is False
    assert result["observed_rows"]["PROBE_OUTCOMES.csv"] == 1
    summary["primary"] = {"contrasts": {"T_N": {"estimate": 1}}}
    (tmp_path / "PRIMARY_RESULTS.json").write_text(json.dumps(summary))
    assert audit_report(tmp_path)["status"] == "failed"


def test_missing_or_nonfinite_published_scalar_fails():
    expected = {"primary": {"effect": .25}}
    assert compare(expected, {"primary": {}})[1] == ["Missing primary.effect"]
    assert compare(expected, {"primary": {"effect": float("nan")}})[1]


def load_tests(loader, tests, pattern):
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            if function.__code__.co_argcount:
                def run(fn=function):
                    with tempfile.TemporaryDirectory() as directory:
                        fn(Path(directory))
                tests.addTest(unittest.FunctionTestCase(run, description=name))
            else:
                tests.addTest(unittest.FunctionTestCase(function))
    return tests


if __name__ == "__main__":
    unittest.main()
