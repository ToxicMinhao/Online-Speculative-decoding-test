"""CPU tests for retained author checkpoint replay; no GPU or teacher calls."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


fixtures = module("author_replay_fixtures", "test_experiment.py")
experiment = fixtures.experiment
native = module("author_replay_native", "native.py")
report = module("author_replay_report", "report.py")
native_fixtures = module("author_replay_native_fixtures", "test_native.py")


def author_reference(rows, reference):
    """Make a completed author run, including distinct bound A1/A2 manifests."""
    result = experiment.run_author_episode(rows, reference, fixtures.FakeBackend())
    assert result["status"] == "complete", result
    for event, update in enumerate(result["updates"], 1):
        label = f"A{event}"
        files = {name: {"bytes": 123, "sha256": fixtures.identity(f"author:{label}:{name}")}
                 for name in ("config.json", "model.safetensors", "version.txt")}
        identity = experiment.digest(files)
        location = f"/retained/author/{result['episode_id']}/checkpoints/U{event}"
        update.update(status="passed", checkpoint_path=location)
        update["checkpoint"] = {"checkpoint_id": label, "path": location, "files": files,
                                "checkpoint_identity_sha256": identity}
        for collection in (result["canonical"], result["scores"]):
            for item in collection:
                if item["checkpoint_id"] == label:
                    item["checkpoint_identity_sha256"] = identity
    return fixtures.rebind(result)


class ReplayBackend(fixtures.FakeBackend):
    def start_episode(self, episode, *, reference, author_reference):
        self.author_reference = copy.deepcopy(author_reference)
        self.author_identities = {update["checkpoint_id"]: update["checkpoint"]["checkpoint_identity_sha256"]
                                  for update in author_reference["updates"]}
        super().start_episode(episode, reference=reference)

    def serve(self, input_ids, checkpoint, *, capture, turn_index, repeat=None):
        result = super().serve(input_ids, checkpoint, capture=capture, turn_index=turn_index, repeat=repeat)
        wrong_identity = self.score_fault and self.score_fault == (checkpoint, turn_index, repeat, "identity")
        if checkpoint in self.author_identities and not wrong_identity:
            result["checkpoint_identity_sha256"] = self.author_identities[checkpoint]
        return result

    def encode(self, messages):
        raise AssertionError("Replay must not encode or regenerate saved conversation inputs")

    def prepare_support(self, item):
        raise AssertionError("Replay must not collect training support")

    def train_event(self, event, support):
        raise AssertionError("Replay must not train")


def replay(*, replica=1, scenario="STABLE_CODE", backend=None):
    rows = fixtures.episode_rows(replica, scenario)
    reference = fixtures.original_reference(rows)
    author = author_reference(rows, reference)
    backend = backend or ReplayBackend()
    events = []
    result = experiment.run_author_replay(rows, reference, author, backend,
                                         emit=lambda kind, data: events.append((kind, copy.deepcopy(data))))
    return result, backend, events, reference, author


class AuthorReplayProtocolTests(unittest.TestCase):
    def test_one_episode_is_72_fresh_scores_on_eight_saved_contexts_without_learning(self):
        result, backend, events, reference, author = replay()
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(result["mode"], "author-replay")
        self.assertEqual(len(result["scores"]), 72)
        self.assertEqual((result["canonical"], result["updates"], result["support"]), ([], [], []))
        self.assertEqual(backend.encoded, [])
        calls = [event for event in backend.events if event["kind"] == "serve"]
        self.assertEqual(len(calls), 72)
        self.assertEqual({event["checkpoint"] for event in calls}, {"F", "A1", "A2"})
        self.assertEqual({event["turn"] for event in calls}, set(range(25, 33)))
        self.assertEqual({event["steps"] for event in calls}, {0})
        self.assertFalse(any(event["capture"] for event in calls))
        for event in calls:
            self.assertEqual(event["input_ids"], reference["canonical"][event["turn"] - 1]["input_ids"])
        expected = {"F": reference["canonical"][0]["checkpoint_identity_sha256"],
                    **{entry["checkpoint_id"]: entry["checkpoint"]["checkpoint_identity_sha256"]
                       for entry in author["updates"]}}
        for row in result["scores"]:
            self.assertEqual(row["checkpoint_identity_sha256"], expected[row["checkpoint_id"]])
            self.assertFalse(row["training_data"])
            self.assertTrue(row["same_generation_valid"])
        self.assertFalse(any(event["kind"] in ("train", "prepare") for event in backend.events))
        self.assertEqual(backend.events[-1]["kind"], "finish")

    def test_plan_is_ten_development_episodes_720_scores_and_zero_updates(self):
        streams = [row for replica in (1, 2, 3, 4) for scenario in experiment.SCENARIOS
                   for row in fixtures.episode_rows(replica, scenario)]
        result = experiment.plan(streams, mode="author-replay")
        self.assertEqual(len(result["episodes"]), 10)
        self.assertEqual(result["total_score_executions"], 720)
        for field in ("adaptation_events", "optimizer_steps_per_event", "optimizer_steps_total",
                      "response_presentations_total", "canonical_request_executions"):
            self.assertEqual(result[field], 0, field)
        self.assertFalse(result["gpu_launch_authorized"])
        self.assertEqual({entry["replica"] for entry in result["episodes"]}, {1, 2})

    def test_rotation_covers_each_first_position_without_reusing_historical_measurements(self):
        for replica, scenario in ((1, "STABLE_CODE"), (1, "SHORT_CODE_FIRST"), (2, "RANDOM_MIX")):
            with self.subTest(replica=replica, scenario=scenario):
                result, backend, events, _, _ = replay(replica=replica, scenario=scenario)
                self.assertEqual(result["status"], "complete")
                orders = [data for kind, data in events if kind == "method_order"]
                offset = (replica - 1) * len(experiment.SCENARIOS) + experiment.SCENARIOS.index(scenario)
                self.assertEqual([row["methods"] for row in orders],
                                 [experiment.method_order(["F", "A1", "A2"], repeat, offset) for repeat in range(3)])
                self.assertEqual({row["methods"][0] for row in orders}, {"F", "A1", "A2"})
                self.assertEqual(len([event for event in backend.events if event["kind"] == "serve"]), 72)

    def test_output_or_identity_mismatch_is_emitted_before_failure_and_cleanup(self):
        for fault, field in (("output", "canonical_output_match"), ("identity", "checkpoint_identity_match"),
                             ("input", "input_match")):
            with self.subTest(fault=fault):
                result, backend, events, _, _ = replay(backend=ReplayBackend(score_fault=("A1", 27, 0, fault)))
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["scores"][-1][field])
                self.assertEqual(result["scores"][-1]["turn"], 27)
                score_at = max(i for i, (kind, _) in enumerate(events) if kind == "score")
                fail_at = next(i for i, (kind, _) in enumerate(events) if kind == "failure")
                self.assertLess(score_at, fail_at)
                self.assertEqual(backend.events[-1]["kind"], "finish")

    def test_author_binding_tampering_or_wrong_episode_is_rejected_before_backend(self):
        changes = [lambda value: value.update(status="running"),
                   lambda value: value.update(episode_id="R2_STABLE_CODE"),
                   lambda value: value["updates"][0].update(checkpoint_id="C1"),
                   lambda value: value["updates"][1].update(optimizer_steps=8),
                   lambda value: value["scores"][0].update(output_ids=[999]),
                   lambda value: value["reference_source_binding"].update(file_sha256="0" * 64),
                   lambda value: value["updates"][0]["checkpoint"]["files"]["model.safetensors"].update(sha256="0" * 64)]
        for mutate in changes:
            rows = fixtures.episode_rows()
            reference = fixtures.original_reference(rows)
            author = author_reference(rows, reference)
            mutate(author)
            fixtures.rebind(author)  # Check semantic binding, not only its outer self-digest.
            backend = ReplayBackend()
            with self.assertRaises(ValueError):
                experiment.run_author_replay(rows, reference, author, backend)
            self.assertEqual(backend.events, [])

    def test_unbound_changed_author_object_is_rejected_before_backend(self):
        rows = fixtures.episode_rows()
        reference = fixtures.original_reference(rows)
        author = author_reference(rows, reference)
        author["canonical"][24]["input_ids"].append(999)
        backend = ReplayBackend()
        with self.assertRaises(ValueError):
            experiment.run_author_replay(rows, reference, author, backend)
        self.assertEqual(backend.events, [])


class AuthorReplayNativeAdmissionTests(unittest.TestCase):
    def test_replay_start_validates_author_payloads_without_loading_current_updater_weights(self):
        rows = fixtures.episode_rows()
        reference = fixtures.original_reference(rows)
        author = author_reference(rows, reference)
        backend = native.NativeBackend({})

        def fake_start(instance, episode_id):
            instance.checkpoints = {"F": {"checkpoint_identity_sha256": reference["canonical"][0]["checkpoint_identity_sha256"]}}
            return {"status": "passed"}

        with patch.object(native.base, "verify_checkpoint") as verify, \
                patch.object(native, "check_followup_sources"), \
                patch.object(native.base.NativeBackend, "start_episode", fake_start):
            backend.start_episode(rows[0]["stream_id"], reference=reference, author_reference=author)
        self.assertEqual([call.args[0]["checkpoint_id"] for call in verify.call_args_list], ["A1", "A2"])
        self.assertEqual(set(backend.checkpoints), {"F", "A1", "A2"})
        self.assertEqual(backend.config["external_checkpoints"], {})
        self.assertEqual(set(backend.config["retained_author_checkpoints"]), {"A1", "A2"})

    def test_retained_replay_backend_rejects_training_and_support_before_any_worker(self):
        backend = native.NativeBackend({"retained_author_checkpoints": {"A1": {"path": "/old-author"}}})
        with patch.object(native.base, "_WorkerClient") as worker:
            with self.assertRaisesRegex(ValueError, "forbids support"):
                backend.prepare_support({})
            with self.assertRaisesRegex(ValueError, "forbids training"):
                backend.train_event(1, [])
            worker.assert_not_called()

    def test_retained_author_admission_requires_exact_mapping_not_a_current_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retained = native_fixtures.checkpoint(root / "old-author" / "U1", "A1")
            retained2 = native_fixtures.checkpoint(root / "old-author" / "U2", "A2")
            config = {"episode_out": str(root / "replay"),
                      "retained_author_checkpoints": {"A1": copy.deepcopy(retained), "A2": retained2}}
            self.assertEqual(native.admit_checkpoint(config, retained), "A1")
            alternate = native_fixtures.checkpoint(root / "other-author" / "U1", "A1")
            with self.assertRaises(ValueError):
                native.admit_checkpoint(config, alternate)
            current = native_fixtures.checkpoint(root / "current" / "U1", "C1")
            current["checkpoint_id"] = "A1"
            with self.assertRaises(ValueError):
                native.admit_checkpoint(config, current)
            (Path(retained["path"]) / "model.safetensors").write_bytes(b"mutated weight payload")
            with self.assertRaises(ValueError):
                native.admit_checkpoint(config, retained)

    def test_bad_author_weights_stop_before_serving_worker_starts(self):
        rows = fixtures.episode_rows()
        reference = fixtures.original_reference(rows)
        author = author_reference(rows, reference)
        backend = native.NativeBackend({})
        with patch.object(native.base.NativeBackend, "start_episode") as start:
            with self.assertRaises((ValueError, FileNotFoundError)):
                backend.start_episode(rows[0]["stream_id"], reference=reference, author_reference=author)
            start.assert_not_called()


def write_bound(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class CampaignFixture:
    """Synthetic custody chain; every byte lives under a temporary directory."""
    def __init__(self, directory):
        self.root = Path(directory) / "campaign"
        self.root.mkdir()
        self.manifest = {"campaign_root": str(self.root), "episodes": []}
        self.state = {"status": "complete", "episodes": {}}
        for name in ("selection", "streams", "runtime_config"):
            self.manifest[name] = write_bound(self.root / "inputs" / (name + ".json"), {"kind": name})
        self.references, self.results, self.retentions = {}, {}, {}
        for replica in (1, 2):
            for scenario in experiment.SCENARIOS:
                rows = fixtures.episode_rows(replica, scenario)
                episode = rows[0]["stream_id"]
                original = fixtures.original_reference(rows)
                original.pop("source_binding")
                bound = write_bound(self.root / "inputs" / "references" / (episode + ".json"), original)
                reference = experiment.load_reference(bound["path"], bound["sha256"])
                self.references[episode] = reference
                self.manifest["episodes"].append({"episode_id": episode, "reference": bound})
                author = author_reference(rows, reference)
                author.pop("source_binding")
                for event, update in enumerate(author["updates"], 1):
                    location = self.root / "runs" / episode / "native" / "checkpoints" / ("U" + str(event))
                    update["checkpoint_path"] = str(location)
                    update["checkpoint"]["path"] = str(location)
                self.results[episode] = author
                result_binding = write_bound(self.root / "runs" / episode / "RESULT.json", author)
                retention = {"status": "complete", "episode_id": episode, "pending": None,
                             "result_binding": result_binding}
                self.retentions[episode] = retention
                receipt = write_bound(self.root / "evidence" / (episode + ".retention.json"), retention)
                self.state["episodes"][episode] = {"status": "complete", "returncode": 0, "retention": receipt}
        self.reseal()

    def reseal(self):
        manifest = write_bound(self.root / "MANIFEST.json", self.manifest)
        self.manifest_sha = manifest["sha256"]
        self.state["manifest_sha256"] = self.manifest_sha
        self.state_sha = write_bound(self.root / "STATE.json", self.state)["sha256"]

    def rewrite_result(self, episode):
        self.retentions[episode]["result_binding"] = write_bound(
            self.root / "runs" / episode / "RESULT.json", self.results[episode])
        self.state["episodes"][episode]["retention"] = write_bound(
            self.root / "evidence" / (episode + ".retention.json"), self.retentions[episode])
        self.reseal()

    def load(self, **override):
        loader = module("author_replay_binding_test", "replay_binding.py")
        episode = "R1_STABLE_CODE"
        arguments = {"episode_id": episode, "reference": self.references[episode],
                     "selection_path": self.manifest["selection"]["path"],
                     "streams_path": self.manifest["streams"]["path"],
                     "runtime_config_path": self.manifest["runtime_config"]["path"]}
        arguments.update(override)
        return loader.load_author_replay(self.root, self.manifest_sha, self.state_sha, **arguments)


class AuthorReplayCampaignBindingTests(unittest.TestCase):
    def test_complete_terminal_chain_loads_exact_author_result_without_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(directory)
            author, provenance = fixture.load()
            self.assertEqual(author["episode_id"], "R1_STABLE_CODE")
            self.assertEqual(author["mode"], "author-path")
            self.assertEqual(author["updates"][0]["checkpoint"]["path"],
                             fixture.results["R1_STABLE_CODE"]["updates"][0]["checkpoint"]["path"])
            self.assertEqual(author["source_binding"]["file_sha256"],
                             fixture.retentions["R1_STABLE_CODE"]["result_binding"]["sha256"])
            self.assertTrue(provenance)
            self.assertFalse(list(fixture.root.rglob("*optimizer*")))

    def test_dataset_runtime_or_reference_byte_changes_reject(self):
        for field in ("selection", "streams", "runtime_config", "reference"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = CampaignFixture(directory)
                binding = fixture.manifest["episodes"][0]["reference"] if field == "reference" else fixture.manifest[field]
                Path(binding["path"]).write_bytes(b"tampered bytes")
                with self.assertRaises(ValueError):
                    fixture.load()

    def test_wrong_supplied_runtime_and_reference_bindings_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(directory)
            wrong = fixture.root / "different_runtime.json"
            write_bound(wrong, {"kind": "different runtime"})
            with self.assertRaises(ValueError):
                fixture.load(runtime_config_path=wrong)
            changed = copy.deepcopy(fixture.references["R1_STABLE_CODE"])
            changed["source_binding"]["file_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                fixture.load(reference=changed)

    def test_manifest_state_retention_or_result_hash_tampering_rejects(self):
        for part in ("manifest", "state", "retention", "result"):
            with self.subTest(part=part), tempfile.TemporaryDirectory() as directory:
                fixture = CampaignFixture(directory)
                paths = {"manifest": fixture.root / "MANIFEST.json", "state": fixture.root / "STATE.json",
                         "retention": Path(fixture.state["episodes"]["R1_STABLE_CODE"]["retention"]["path"]),
                         "result": fixture.root / "runs" / "R1_STABLE_CODE" / "RESULT.json"}
                with paths[part].open("a") as stream:
                    stream.write(" ")  # JSON semantics unchanged; byte provenance changed.
                with self.assertRaises(ValueError):
                    fixture.load()

    def test_incomplete_campaign_other_failed_episode_or_reserved_episode_rejects(self):
        changes = [lambda fixture: fixture.state.update(status="running"),
                   lambda fixture: fixture.state["episodes"]["R2_RANDOM_MIX"].update(returncode=1),
                   lambda fixture: fixture.manifest["episodes"].pop(),
                   lambda fixture: fixture.state["episodes"].pop("R2_RANDOM_MIX")]
        for mutate in changes:
            with tempfile.TemporaryDirectory() as directory:
                fixture = CampaignFixture(directory)
                mutate(fixture)
                fixture.reseal()
                with self.assertRaises(ValueError):
                    fixture.load()
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(directory)
            with self.assertRaises(ValueError):
                fixture.load(episode_id="R3_STABLE_CODE")

    def test_bound_but_foreign_author_checkpoint_cannot_replace_same_episode_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(directory)
            episode = "R1_STABLE_CODE"
            checkpoint = fixture.results[episode]["updates"][0]
            checkpoint["checkpoint"]["path"] = str(fixture.root / "runs" / "R2_STABLE_CODE" / "native" / "checkpoints" / "U1")
            checkpoint["checkpoint_path"] = checkpoint["checkpoint"]["path"]
            fixture.rewrite_result(episode)
            with self.assertRaises(ValueError):
                fixture.load()

    def test_retention_symlink_escape_is_rejected_even_with_valid_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(directory)
            episode = "R1_STABLE_CODE"
            old = fixture.state["episodes"][episode]["retention"]
            outside = Path(directory) / "outside-retention.json"
            outside.write_bytes(Path(old["path"]).read_bytes())
            escaped = fixture.root / "evidence" / "escaped.json"
            escaped.symlink_to(outside)
            fixture.state["episodes"][episode]["retention"] = {**old, "path": str(escaped)}
            fixture.reseal()
            with self.assertRaises(ValueError):
                fixture.load()


class AuthorReplayReportTests(unittest.TestCase):
    def scored(self, replica=1, scale=1.):
        result, _, _, _, _ = replay(replica=replica)
        self.assertEqual(result["status"], "complete", result)
        for row in result["scores"]:
            row["request_s"] = scale * {"F": 2., "A1": 1.8, "A2": 1.5}[row["checkpoint_id"]]
            row["verification_calls"] = {"F": 10, "A1": 9, "A2": 7}[row["checkpoint_id"]]
            row["verification_committed_tokens"] = 1
            row["activation_s"], row["cache_maintenance_s"] = 999., 999.
        return result

    def test_report_primary_comparison_and_no_reconstructed_training_or_session_savings(self):
        result = report.summarize([self.scored()])
        windows = result["episodes"][0]["windows"]
        self.assertEqual({row["window"] for row in windows}, {"25-32"})
        pooled = next(row for row in windows if row["repeat"] == "pooled")
        self.assertEqual(set(pooled["metrics"]), {"F", "A1", "A2"})
        pairs = {(row["candidate"], row["control"]): row for row in pooled["comparisons"]}
        self.assertEqual(set(pairs), {("A1", "F"), ("A2", "F"), ("A2", "A1")})
        self.assertAlmostEqual(pairs[("A2", "A1")]["latency_reduction_percent"], 100 * (1 - 1.5 / 1.8))
        self.assertAlmostEqual(pairs[("A2", "A1")]["verification_call_reduction_percent"], 100 * (1 - 7 / 9))
        self.assertEqual(pooled["metrics"]["F"]["executions"], 24)
        self.assertEqual(pooled["metrics"]["F"]["distinct_request_contexts"], 8)
        self.assertAlmostEqual(pooled["metrics"]["F"]["complete_request_s"], 48.)
        costs = result["episodes"][0]["cost_breakdown"]
        self.assertIsNone(costs["accounted_author_episode_s"])
        self.assertFalse(costs["complete_nonoverlapping_accounting"])
        self.assertIn("unavailable", costs["frozen_full_session_comparison"])

    def test_capture_on_last_score_keeps_all_rows_but_invalidates_only_affected_pairs(self):
        for field, payload in (("feedback", {"synthetic_capture": True}),
                               ("capture_costs", {"append_capture_s": .1})):
            with self.subTest(field=field):
                class LastScoreCapture(ReplayBackend):
                    def serve(self, input_ids, checkpoint, *, capture, turn_index, repeat=None):
                        result = super().serve(input_ids, checkpoint, capture=capture,
                                               turn_index=turn_index, repeat=repeat)
                        if (checkpoint, turn_index, repeat) == ("A1", 32, 2):
                            result[field] = copy.deepcopy(payload)
                        return result

                result, backend, events, _, _ = replay(backend=LastScoreCapture())
                self.assertEqual(result["status"], "failed")
                self.assertEqual(len(result["scores"]), 72)
                self.assertEqual(len([event for event in backend.events if event["kind"] == "serve"]), 72)
                last = result["scores"][-1]
                self.assertEqual((last["checkpoint_id"], last["turn"], last["repeat"]), ("A1", 32, 2))
                self.assertEqual(last[field], payload)
                self.assertTrue(last["same_generation_valid"])  # Output equality alone is insufficient.
                score_at = max(i for i, (kind, _) in enumerate(events) if kind == "score")
                fail_at = next(i for i, (kind, _) in enumerate(events) if kind == "failure")
                self.assertLess(score_at, fail_at)
                before_report = copy.deepcopy(result["scores"])
                summary = report.summarize([result])
                self.assertEqual(result["scores"], before_report)
                self.assertEqual(summary["episodes"][0]["status"], "failed")
                pooled = next(row for row in summary["pooled"] if row["repeat"] == "pooled")
                pairs = {(row["candidate"], row["control"]): row for row in pooled["comparisons"]}
                for key in (("A2", "A1"), ("A1", "F")):
                    self.assertFalse(pairs[key]["paired_same_generation_valid"])
                    self.assertIsNone(pairs[key]["latency_reduction_percent"])
                    self.assertIsNone(pairs[key]["verification_call_reduction_percent"])
                    self.assertIsNone(pairs[key]["calls_per_1000_token_reduction_percent"])
                self.assertTrue(pairs[("A2", "F")]["paired_same_generation_valid"])
                self.assertIsNotNone(pairs[("A2", "F")]["latency_reduction_percent"])
                self.assertEqual({key: value["executions"] for key, value in pooled["metrics"].items()},
                                 {"F": 24, "A1": 24, "A2": 24})

    def test_pooled_gain_uses_summed_times_and_mismatches_remain_visible(self):
        first, second = self.scored(), self.scored(replica=2, scale=10.)
        for row in second["scores"]:
            if row["checkpoint_id"] == "A2":
                row["request_s"] = 19.  # A2 regresses versus A1=18 on this episode.
        result = report.summarize([first, second])
        pooled = next(row for row in result["pooled"] if row["repeat"] == "pooled")
        comparison = next(row for row in pooled["comparisons"] if (row["candidate"], row["control"]) == ("A2", "A1"))
        self.assertAlmostEqual(comparison["latency_reduction_percent"], 100 * (1 - (1.5 + 19.) / (1.8 + 18.)))
        self.assertLess(comparison["latency_reduction_percent"], 0)
        self.assertEqual(pooled["expected_distinct_contexts"], 16)
        second["scores"][0]["output_ids"] = [999]
        invalid = report.summarize([first, second])
        invalid_pool = next(row for row in invalid["pooled"] if row["repeat"] == "pooled")
        affected = next(row for row in invalid_pool["comparisons"]
                        if second["scores"][0]["checkpoint_id"] in (row["candidate"], row["control"]))
        self.assertFalse(affected["paired_same_generation_valid"])
        self.assertIsNone(affected["latency_reduction_percent"])
        self.assertEqual(invalid_pool["metrics"][second["scores"][0]["checkpoint_id"]]["executions"], 48)

    def test_csv_keeps_each_repeat_episode_scenario_and_pooled_comparison_without_overwrite(self):
        result = report.summarize([self.scored()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "COMPARISONS.csv"
            report.write_comparison_csv(path, result)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 36)  # 3 scopes x 4 repeats x 3 comparisons.
            self.assertEqual({row["scope"] for row in rows}, {"episode", "scenario", "pooled"})
            self.assertEqual({row["repeat"] for row in rows}, {"0", "1", "2", "pooled"})
            self.assertEqual({row["mode"] for row in rows}, {"author-replay"})
            self.assertEqual(sum((row["candidate"], row["control"]) == ("A2", "A1") for row in rows), 12)
            self.assertTrue(all(row["paired_same_generation_valid"] == "True" for row in rows))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                report.write_comparison_csv(path, result)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
