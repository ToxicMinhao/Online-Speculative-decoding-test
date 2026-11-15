"""CPU adapter tests; native numerical and cache parity still require the gate."""
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.native_update_size import probe_client as client
from online_feedback.storage import write_json
from test_threat_native_probe import record_fixture, keys_fixture, observed


def exports_fixture():
    return [dict(window=window, state=state, checkpoint=f"/retained/{window}_{state}",
        weights_identity=dict(bytes=128, sha256=f"{window}_{state}"), reused_from=None)
        for window in client.WINDOWS for state in client.STATES]


def receipt_fixture():
    return dict(new_logical_control_rows=0, physical_control_blocks=0, aliased_control_rows=0,
        invalid_control_rows=0, unknown_control_rows=0, control_attempts=0,
        neutral_attempts=0, neutral_blocks=0, sentinel_attempts=0, sentinel_blocks=0,
        sentinel_checks=[], reuse_sentinel_status="unattempted", active_probe=None,
        active_qualification=None)


def history_fixture(root, keys):
    exports = [dict(window=window, state=state, checkpoint=f"/deleted/{window}_{state}",
        weights_identity=dict(bytes=128, sha256=f"{window}_{state}"))
        for window in client.WINDOWS for state in client.HISTORICAL_STATES]
    rows = [dict(key, window=export["window"], state=export["state"], checkpoint=export["checkpoint"],
        valid=True, accepted_prefix=0, valid_proposals=4, first_error=1,
        matched_eos=False, censored_at_cap=False)
        for export in exports for key in keys]
    write_json(root / "exports.json", exports)
    write_json(root / "identity.json", dict(status="passed", historical_execution_commit=client.HISTORICAL_EXECUTION))
    (root / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    plan = dict(historical_probe_outcomes=str(root / "rows.jsonl"),
        historical_probe_exports=str(root / "exports.json"), reuse_identity_receipt=str(root / "identity.json"))
    return plan, rows, exports


class BindingTests(unittest.TestCase):
    def test_only_exact_two_control_grid_is_accepted(self):
        exports = exports_fixture()
        self.assertEqual(len(client.validate_exports(exports)), 32)
        for changed in (exports[:-1], exports + [exports[0]], exports + [dict(exports[0], state="E")]):
            with self.assertRaises(ValueError):
                client.validate_exports(changed)

    def test_alias_requires_same_physical_export_and_identity(self):
        exports = exports_fixture()
        exports[1].update(checkpoint=exports[0]["checkpoint"],
            weights_identity=exports[0]["weights_identity"], reused_from="S_low")
        client.validate_exports(exports)
        for field, value in (("reused_from", "E"), ("weights_identity", dict(bytes=128, sha256="different")),
                             ("checkpoint", "/another/file")):
            changed = deepcopy(exports)
            changed[1][field] = value
            with self.assertRaises(ValueError):
                client.validate_exports(changed)
        exports[1]["reused_from"] = None
        with self.assertRaises(ValueError):
            client.validate_exports(exports)

    def test_complete_seal_precedes_any_backend_work(self):
        with tempfile.TemporaryDirectory() as folder:
            seal = Path(folder) / "seal.json"
            exports = exports_fixture()
            write_json(seal, dict(status="sealed", exports=exports[:-1]))
            with patch.object(client, "ProbeBackend") as backend:
                with self.assertRaisesRegex(ValueError, "sealed"):
                    client.run(dict(control_seal=str(seal)), exports, Path(folder) / "run", {})
                backend.assert_not_called()

    def test_historical_grid_reuses_missing_checkpoint_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            plan, rows, exports = history_fixture(Path(folder), keys_fixture())
            result, receipt = client.historical_measurements(plan, keys_fixture())
            self.assertEqual(len(result), 192)
            self.assertEqual(result[0]["measurement_origin"], "historical_PR47")
            self.assertEqual(result[0]["historical_weights_identity"], exports[0]["weights_identity"])
            self.assertEqual(receipt["status"], "passed")
            self.assertFalse(Path(exports[0]["checkpoint"]).exists())

    def test_historical_duplicates_invalid_or_unsealed_source_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, rows, exports = history_fixture(root, keys_fixture())
            write_json(root / "exports.json", exports + [exports[0]])
            with self.assertRaises(ValueError):
                client.historical_measurements(plan, keys_fixture())
            write_json(root / "exports.json", exports)
            for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], valid=False)] + rows[1:],
                            [dict(rows[0], checkpoint="/wrong")] + rows[1:]):
                (root / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in changed))
                with self.assertRaises(ValueError):
                    client.historical_measurements(plan, keys_fixture())
            write_json(root / "identity.json", dict(status="passed", historical_execution_commit="wrong"))
            with self.assertRaises(ValueError):
                client.historical_measurements(plan, keys_fixture())

    def test_fixed_eight_keys_are_sorted_without_outcome_selection(self):
        keys = [dict(key, request_id=f"r{request}", improvement=100 - request)
                for request in (2, 0, 1) for key in reversed(keys_fixture())]
        selected = client.sentinel_keys(keys)
        self.assertEqual([(r["request_id"], r["canonical_start"]) for r in selected],
                         [(f"r{r}", start) for r in (0, 1) for start in range(4)])


class ScoringTests(unittest.TestCase):
    def test_scoring_phase_reconstructs_each_of_64_parents_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            keys, paths, snapshots = [], {}, {}
            for request in range(64):
                rid = f"r{request:02d}"
                record = dict(record_fixture(), record_id=rid)
                path = root / "parents" / (rid + ".json")
                write_json(path, record)
                paths[rid] = str(path)
                keys.extend(dict(key, request_id=rid) for key in keys_fixture())
            plan, _, _ = history_fixture(root, keys)
            exports = exports_fixture()
            write_json(root / "keys.json", keys)
            write_json(root / "seal.json", dict(status="sealed", exports=exports))
            plan.update(parent_records=paths, probe_keys=str(root / "keys.json"),
                control_seal=str(root / "seal.json"), target_path="/target", worker_python="/python",
                checkpoints=dict(K6=dict(export="/K6")))
            backend = SimpleNamespace(loading_s=.1, close=Mock(), reconstruct=Mock(side_effect=lambda record, subset, checkpoint:
                dict(status="passed", snapshots={str(key["round_ordinal"]): dict(handle=key["request_id"])
                                                  for key in subset})))
            budget = Mock()
            budget.summary.return_value = {kind: {} for kind in ("attempted", "completed", "failed", "unknown")}
            calls = []
            def qualification(*args):
                calls.append("qualification")
                self.assertEqual(backend.reconstruct.call_count, 64)
                self.assertEqual(len(args[5]), 12288)
                args[-1]["reuse_sentinel_status"] = "passed"
            def scoring(*args):
                calls.append("scoring")
                self.assertEqual(len(args[1]), 32)
                self.assertEqual(len(args[2]), 256)
                self.assertEqual(len(args[4]), 256)
                self.assertEqual(args[-2]["reuse_sentinel_status"], "passed")
            with ExitStack() as stack:
                stack.enter_context(patch.object(client, "resource_estimate", return_value=dict(host_reserve_bytes=0)))
                stack.enter_context(patch.object(client, "check_host_admission"))
                stack.enter_context(patch.object(client, "ProbeBackend", return_value=backend))
                stack.enter_context(patch("experiments.verification_transition.budget.connect_budget", return_value=budget))
                stack.enter_context(patch.object(client, "qualify_reuse", side_effect=qualification))
                stack.enter_context(patch.object(client, "evaluate_controls", side_effect=scoring))
                result = client.run(plan, exports, root / "run", {})
            self.assertEqual(calls, ["qualification", "scoring"])
            self.assertEqual(result["requests_reconstructed"], 64)
            self.assertEqual(backend.reconstruct.call_count, 64)
            self.assertEqual({call.args[0]["record_id"] for call in backend.reconstruct.call_args_list}, set(paths))
            backend.close.assert_called_once()

    def test_alias_has_one_physical_evaluation_but_two_logical_rows(self):
        record = record_fixture()
        keys = keys_fixture()[:2]
        exports = exports_fixture()[:2]
        exports[1].update(checkpoint=exports[0]["checkpoint"],
            weights_identity=exports[0]["weights_identity"], reused_from="S_low")
        backend = SimpleNamespace(one_block=Mock(side_effect=lambda snapshot, checkpoint: observed(record, snapshot)))
        snapshots = {client.probe_key(key): key["round_ordinal"] for key in keys}
        with tempfile.TemporaryDirectory() as folder:
            outcomes, receipt = [], receipt_fixture()
            client.evaluate_controls(backend, {(row["window"], row["state"]): row for row in exports},
                keys, {"r": record}, snapshots, Path(folder), Mock(), receipt, outcomes)
            self.assertEqual(backend.one_block.call_count, 2)
            self.assertEqual((len(outcomes), receipt["physical_control_blocks"], receipt["aliased_control_rows"]), (4, 2, 2))
            for low, high in zip(outcomes[:2], outcomes[2:]):
                self.assertEqual(low["execution_id"], high["execution_id"])
                self.assertEqual(high["reused_from"], "S_low")

    def test_raw_invalid_measurement_is_saved_before_validation(self):
        record, key = record_fixture(), keys_fixture()[0]
        invalid = observed(record, 0)
        invalid["output_matches_canonical_prefix"] = False
        exports = {(row["window"], row["state"]): row for row in exports_fixture()[:2]}
        with tempfile.TemporaryDirectory() as folder:
            root, outcomes, receipt = Path(folder), [], receipt_fixture()
            with self.assertRaises(ValueError):
                client.evaluate_controls(SimpleNamespace(one_block=Mock(return_value=invalid)), exports,
                    [key], {"r": record}, {client.probe_key(key): 0}, root, Mock(), receipt, outcomes)
            self.assertEqual(outcomes[0]["status"], "invalid")
            self.assertEqual(receipt["invalid_control_rows"], 1)
            self.assertEqual(json.loads((root / outcomes[0]["execution_id"]).read_text()), invalid)
            self.assertIn('"valid": false', (root / "PROBE_OUTCOMES.jsonl").read_text())

    def test_transport_failure_is_unknown_not_unexecuted(self):
        record, key = record_fixture(), keys_fixture()[0]
        exports = {(row["window"], row["state"]): row for row in exports_fixture()[:2]}
        with tempfile.TemporaryDirectory() as folder:
            outcomes, receipt = [], receipt_fixture()
            with self.assertRaises(RuntimeError):
                client.evaluate_controls(SimpleNamespace(one_block=Mock(side_effect=RuntimeError("lost worker"))),
                    exports, [key], {"r": record}, {client.probe_key(key): 0}, Path(folder), Mock(), receipt, outcomes)
            self.assertEqual(outcomes[0]["status"], "unknown")
            self.assertEqual(receipt["control_attempts"], 1)
            self.assertEqual(receipt["physical_control_blocks"], 0)

    def test_sentinel_gate_compares_all_proposals_and_verdicts(self):
        raw = observed(record_fixture(), 0)
        for field in ("draft_tokens", "target_argmax", "positions"):
            changed = deepcopy(raw)
            changed["blocks"][0][field][-1] += 1
            self.assertEqual(client.sentinel_parity(changed, raw)["status"], "failed")
        changed = deepcopy(raw)
        changed.update(wall_s=99, reconstruction=dict(status="passed", diagnostic_clock=1))
        self.assertEqual(client.sentinel_parity(changed, raw)["status"], "passed")

    def test_qualified_reuse_is_bounded_at_24_plus_eight_without_parent_reconstruction(self):
        keys, parents, snapshots = [], {}, {}
        for request in (0, 1):
            rid = f"r{request}"
            parents[rid] = dict(record_fixture(), record_id=rid)
            for base in keys_fixture():
                key = dict(base, request_id=rid)
                keys.append(key)
                snapshots[client.probe_key(key)] = (rid, key["round_ordinal"])
        backend = SimpleNamespace(one_block=Mock(side_effect=lambda snapshot, checkpoint:
            observed(parents[snapshot[0]], snapshot[1])))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_root = root / "old"
            history, exports = [], []
            for state in client.HISTORICAL_STATES:
                identity = dict(bytes=128, sha256=state)
                export = dict(window=8, state=state, checkpoint=f"/{state}", weights_identity=identity)
                exports.append(export)
                for key in keys:
                    raw = observed(parents[key["request_id"]], key["round_ordinal"])
                    row = client.score_observation(parents[key["request_id"]], key, raw,
                        window=8, state=state, checkpoint=export["checkpoint"])
                    execution_id = f"{state}/{key['request_id']}/{key['round_ordinal']}.json"
                    write_json(old_root / execution_id, raw)
                    history.append(dict(row, historical_weights_identity=identity, execution_id=execution_id))
            plan = dict(neutral_qualifier_blocks=8, sentinel_exports=exports,
                checkpoints=dict(K6=dict(export="/K6")), historical_probe_root=str(old_root),
                reuse_identity_receipt="/identity.json")
            receipt = receipt_fixture()
            client.qualify_reuse(backend, plan, keys, parents, snapshots, history, root, Mock(), receipt)
            self.assertEqual(backend.one_block.call_count, 32)
            self.assertEqual((receipt["neutral_blocks"], receipt["sentinel_blocks"]), (8, 24))
            self.assertEqual(receipt["reuse_sentinel_status"], "passed")
            self.assertIsNone(receipt["active_qualification"])
            plan["neutral_qualifier_blocks"] = 9
            with self.assertRaises(ValueError):
                client.qualify_reuse(backend, plan, keys, parents, snapshots, history, root, Mock(), receipt)
            self.assertEqual(backend.one_block.call_count, 32)


if __name__ == "__main__":
    unittest.main()
