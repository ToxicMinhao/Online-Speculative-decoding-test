"""Synthetic CPU fixtures for the private, separate arithmetic audit."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_qualification import starts, summarize_routes, inspect_run, ROUTES, MODES, RESTORATION, GRAPH_LAYERS


def write(path, value):
    path.write_text(json.dumps(value))


def pricing_rows(request_id="r0", ordinal=0, rounds=8):
    result = []
    times = dict(FIVE=5, THREE=3, ADAPTIVE3=3.5, ADAPTIVE5=5.5)
    for anchor, index in enumerate(starts(rounds)):
        for repeat in range(4):
            rotate = (ordinal + anchor + repeat) % 4
            for position, route in enumerate(ROUTES[rotate:] + ROUTES[:rotate]):
                result.append(dict(request_id=request_id, round_index=index, anchor_ordinal=anchor,
                                   prefix_length=100 + index, repeat=repeat, route=route,
                                   route_position=position, elapsed_ms=times[route], valid=True,
                                   depth=5 if route in ("FIVE", "ADAPTIVE5") else 3,
                                   proposal_ids=[10, 11] + [0] * 13))
    return result


def completed_fixture(root):
    requests, generations, flattened = [], [], []
    for ordinal in range(16):
        request = dict(request_id=f"r{ordinal}", prompt_ids=[ordinal + 1, 88],
                       stop_token_ids=[151645, 151643], max_new_tokens=2048)
        requests.append(request)
        outputs = [42] + list(range(100, 115)) + [151645]
        full = None
        for phase, mode in (("READOUT", "FIVE"), ("qualification", MODES[ordinal])):
            stride = 1 if mode in ("THREE", "ALTERNATE") else 2
            trace = []
            for index in range(0, 16, stride):
                committed = outputs[index + 1:index + 1 + stride]
                trace.append(dict(output_start=index, anchor_token=outputs[index],
                                  positions=list(range(2 + index, 18 + index)),
                                  committed_progress=len(committed), committed_tokens=committed,
                                  accepted_draft_tokens=len(committed), draft_tokens=committed + [0] * (15 - len(committed))))
            label = f"{ordinal:02d}_{phase}_{mode}"
            record = dict(request, response_ids=outputs, finish_reason="eos", metadata=dict(
                trace=trace, verification_calls=len(trace), committed_progress=16,
                accepted_draft_tokens=16, request_latency_s=0.1, native_generate_s=0.09,
                feedback_capture_in_timed_request=False))
            write(root / (label + ".json"), record)
            generations.append(dict(label=label, valid=True, verification_calls=len(trace)))
            if phase == "READOUT":
                full = record
            else:
                n = len(trace)
                routes = [5] * n if mode == "FORCED5" else [3] * n if mode == "THREE" else [(3, 5)[i % 2] for i in range(n)]
                audit = [dict(all_15_equal=True, original_five_proposals=[0] * 15,
                              forced_five_proposals=[0] * 15, canonical_start=t["positions"][0]) for t in trace]
                graphs = [dict(graph=name, captured_layer_calls=calls, warmup_executions=2,
                               capture_executions=1) for name, calls in GRAPH_LAYERS.items()]
                write(root / f"{ordinal:02d}_ROUTE_STATUS.json", dict(status="passed", routes=routes, head_invocations=3 * n,
                    live_rounds=n, shadow_proposal_forwards=2 * n, shadow_vocab_projections=2 * n,
                    audit_records=audit, graph_setup=graphs))
        rows = pricing_rows(request["request_id"], ordinal)
        for row in rows:
            block = full["metadata"]["trace"][row["round_index"]]
            row["prefix_length"] = block["positions"][0]
            row["proposal_ids"] = list(block["draft_tokens"])
        write(root / f"{ordinal:02d}_PROFILE.json", dict(
            status="passed", request_id=request["request_id"], sampled_round_indices=starts(8),
            measurements=rows, parity=[], restoration_status="passed", restoration_checks=dict.fromkeys(RESTORATION, True),
            accounting=dict(planned_timed_measurements=128, returned_timed_measurements=128,
                            unknown_timed_measurements=0, unattempted_timed_measurements=0)))
        flattened.extend(rows)
    manifest = root / "manifest.json"
    write(manifest, dict(manifests=dict(READOUT=dict(requests=requests))))
    raw = manifest.read_bytes()
    write(root / "PLAN.json", dict(code_commit="fixture", manifest_file="remote/manifest.json",
                                    input_identities={"remote/manifest.json": dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())}))
    write(root / "PREFLIGHT.json", dict(status="passed", code_commit="fixture"))
    write(root / "GENERATIONS.json", generations)
    write(root / "PRICING_MEASUREMENTS.json", flattened)
    write(root / "STATUS.json", dict(status="completed", returned_generations=32, valid_generations=32,
                                      invalid_generations=0, unknown_generations=0, unattempted_generations=0,
                                      optimizer_steps=0, fitting_presentations=0, final_generations=0, active_profile=None))
    for name in ("WEIGHTS_BEFORE.json", "WEIGHTS_AFTER.json"):
        write(root / name, dict(status="passed", all_equal=True))
    return manifest


class AuditTests(unittest.TestCase):
    def test_independent_weighted_prompt_bootstrap(self):
        native = {f"r{i}": dict(rounds=8 * (i + 1), request_s=0.05 * (i + 1)) for i in range(16)}
        rows = [row for i in range(16) for row in pricing_rows(f"r{i}", i, native[f"r{i}"]["rounds"])]
        stats = summarize_routes(rows, native, economic_eligible=True)
        self.assertEqual(stats["economic"]["point_pct"], 24.0)
        self.assertEqual(stats["economic"]["ci95_pct"], [24.0, 24.0])
        self.assertEqual(stats["economic"]["continuation_delta_ms"], 2.0)
        self.assertEqual(stats["economic"]["decision"], "continue_to_readout_fit")

    def test_incomplete_grid_withholds_economic_result_and_preserves_complete_starts(self):
        native = {"r0": dict(rounds=8, request_s=0.1)}
        stats = summarize_routes(pricing_rows()[:-1], native, economic_eligible=False)
        self.assertIsNone(stats["economic"])
        self.assertEqual(stats["complete_start_costs"], 7)
        self.assertEqual(stats["complete_prompt_costs"], 0)

    def test_zero_savings_clipped_when_routing_is_more_expensive(self):
        rows, native = [], {}
        for i in range(16):
            native[f"r{i}"] = dict(rounds=8, request_s=0.1)
            for row in pricing_rows(f"r{i}", i):
                if row["route"].startswith("ADAPTIVE"):
                    row["elapsed_ms"] = 8
                rows.append(row)
        result = summarize_routes(rows, native, economic_eligible=True)
        self.assertEqual(result["economic"]["point_pct"], 0)
        self.assertEqual(result["economic"]["decision"], "stop_pricing_only")

    def test_complete_raw_fixture_passes_without_production_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = completed_fixture(root)
            result = inspect_run(root, manifest, "fixture")
            self.assertEqual(result["audit_errors"], [])
            self.assertTrue(result["qualified_complete"])
            self.assertEqual(result["route_accounting"]["returned"], 2048)
            self.assertEqual(result["statistics"]["economic"]["point_pct"], 12.0)
            self.assertEqual(result["generation_accounting"]["returned"], 32)

    def test_tampered_proposal_and_false_pass_receipt_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = completed_fixture(root)
            path = root / "00_PROFILE.json"
            value = json.loads(path.read_text())
            value["measurements"][0]["proposal_ids"][0] += 1
            write(path, value)
            result = inspect_run(root, manifest, "fixture")
            self.assertFalse(result["qualified_complete"])
            self.assertIn("profile_0:row_0_native_parity", result["audit_errors"])
            self.assertIsNone(result["statistics"]["economic"])

    def test_wrong_layer_execution_head_count_or_mixed_schedule_fails(self):
        mutations = (
            (lambda r: r["graph_setup"][0].update(captured_layer_calls=[1, 1, 1, 1, 1]), "captured_layers_static_three"),
            (lambda r: r.update(head_invocations=r["head_invocations"] + 1), "one_live_head_plus_two_audit_heads"),
            (lambda r: r["routes"].__setitem__(2, 5), "whole_block_depths"),
        )
        for mutate, expected in mutations:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = completed_fixture(root)
                path = root / "08_ROUTE_STATUS.json"
                value = json.loads(path.read_text())
                mutate(value)
                write(path, value)
                result = inspect_run(root, manifest, "fixture")
                self.assertFalse(result["qualified_complete"])
                self.assertIn("08_qualification_ALTERNATE:" + expected, result["audit_errors"])

    def test_interrupted_profile_keeps_unknown_work_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = completed_fixture(root)
            for path in root.glob("*.json"):
                if path.name[:2].isdigit() and not path.name.startswith("00_READOUT"):
                    path.unlink()
            first = json.loads((root / "PRICING_MEASUREMENTS.json").read_text())[0]
            profile = dict(status="failed", request_id="r0", sampled_round_indices=starts(8), measurements=[first],
                           active_operation=dict(kind="timed_route"), accounting=dict(planned_timed_measurements=128,
                           returned_timed_measurements=1, unknown_timed_measurements=1, unattempted_timed_measurements=126))
            write(root / "00_PROFILE.json", profile)
            write(root / "PRICING_MEASUREMENTS.json", [first])
            write(root / "GENERATIONS.json", [dict(label="00_READOUT_FIVE", valid=True, verification_calls=8)])
            write(root / "STATUS.json", dict(status="blocked", returned_generations=1, valid_generations=1,
                invalid_generations=0, unknown_generations=0, unattempted_generations=31,
                optimizer_steps=0, fitting_presentations=0, final_generations=0, active_profile=dict(ordinal=0)))
            result = inspect_run(root, manifest, "fixture")
            self.assertEqual(result["audit_errors"], [])
            self.assertFalse(result["qualified_complete"])
            self.assertEqual(result["route_accounting"]["incomplete_profiles"][0]["accounting"]["unknown_timed_measurements"], 1)
            self.assertIsNone(result["statistics"]["economic"])


if __name__ == "__main__":
    unittest.main()
