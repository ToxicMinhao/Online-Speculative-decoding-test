"""Exact, fail-fast comparison of one current-pool replay to retained PR51 E5.

Only declared timing and publication metadata may differ. Scientific values are
compared without a numerical tolerance. Large tensors produce compact equality
and fingerprint receipts, never full parameter or gradient dumps.
"""
from pathlib import Path
import hashlib
import math

from online_feedback.storage import write_json

SCHEMA = "mathqa_e5_exact_replay_comparison_v1"
CATEGORIES = (
    "draft_tensors", "optimizer_moments_counters_options", "rng",
    "scientific_configuration", "original_event_ancestry",
    "sampler_accumulation_boundary", "event_scientific_receipts",
)
RESPONSE_FIELDS = frozenset(("scope", "record_id", "event_id", "pass_index",
    "presentation_ordinal", "sampled_anchors", "loss", "ratio_metrics", "sum_metrics"))
WINDOW_FIELDS = frozenset(("scope", "event_id", "window", "global_step", "record_ids",
    "gradient_norm", "learning_rate", "mean_response_loss"))
PRESENTATION_FIELDS = frozenset(("record_id", "pass", "optimizer_step_after", "optimizer_stepped",
    "learning_rate", "metrics", "training_s", "sampled_anchors", "metrics_scope",
    "response_observation", "prompt_tokens", "response_tokens", "active_anchors"))
EVENT_FIELDS = frozenset(("event_id", "features", "passes", "response_presentations", "start_step",
    "end_step", "optimizer_steps", "reduction", "objective", "objective_work", "gradient_denominator",
    "records", "selection_totals", "selection_totals_source", "method", "dataset", "kplus_loss",
    "mathqa_fixed_lr_transfer_v1"))
STATE_FIELDS = frozenset(("event_schema", "draft_state_dict", "backend", "rng", "global_step",
    "presentations", "events", "strategy", "run_id", "world_size", "config_text",
    "algorithm_contract", "reduction", "accumulation_steps", "horizon"))
OBJECTIVE_TIMES = frozenset(("reference_setup_s", "reference_forward_s", "reference_head_s",
    "projection_s", "teacher_head_s", "selection_s", "kplus_chunk_s", "kplus_response_s"))
COST_TIMES = frozenset(("training_s", "checkpoint_save_s", "export_s", "event_wall_s_before_commit"))
PUBLICATION_FIELDS = frozenset(("status", "checkpoint_path", "export_path", "full_state_path", "costs"))
RATIO_FIELDS = frozenset(("objective_loss", "forward_kl", "selection_rate"))
SUM_FIELDS = frozenset((
    "supervised_position_exposure", "available_position_occurrences", "selected_position_occurrences",
    "available_position_weight", "selected_position_weight", "no_error_blocks", "empty_blocks",
    "main_loss_numerator", "loss_denominator", "kplus_N_K", "kplus_S", "kplus_W1", "kplus_W2",
    "kplus_N1", "kplus_N2", "kplus_N_new", "kplus_priority1_error_count", "kplus_priority2_error_count",
    "kplus_coefficient1", "kplus_coefficient2", "kplus_first_error_weight_after",
    "kplus_conservation_error", "kplus_error_weight_moved", "kplus_error_weight_moved_share",
    "kplus_mixed_response", "kplus_response_count",
)) | frozenset(f"first_error_position_{index}" for index in range(1, 16))
METRIC_FIELDS = SUM_FIELDS | frozenset(("author_weighted_objective", "forward_kl", "loss",
    "objective_loss", "selection_rate", "response_presentations", "acc"))
EXCLUSIONS = {
    "presentation": ["training_s"],
    "event": sorted(PUBLICATION_FIELDS),
    "event.objective_work": sorted(OBJECTIVE_TIMES),
    "event.costs": sorted(COST_TIMES),
    "endpoint": ["run_id"],
    "endpoint.config_text": ["run_id", "output_dir"],
}


def _fingerprint(value):
    """One tensor/array at a time; preserve represented bytes and dtype."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        finite = not cpu.is_floating_point() or bool(torch.isfinite(cpu).all())
        data = memoryview(cpu.reshape(-1).view(torch.uint8).numpy())
        return dict(kind="tensor", shape=list(cpu.shape), dtype=str(cpu.dtype),
                    sha256=hashlib.sha256(data).hexdigest(), finite=finite)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        finite = not np.issubdtype(array.dtype, np.number) or bool(np.isfinite(array).all())
        return dict(kind="ndarray", shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(memoryview(array)).hexdigest(), finite=finite)
    return None


def _brief(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        return value if not isinstance(value, str) or len(value) <= 240 else value[:240] + "..."
    return dict(type=type(value).__name__, length=len(value) if hasattr(value, "__len__") else None)


class ReplayComparator:
    def __init__(self, historical_state, historical_event_receipt, historical_responses,
                 historical_windows, *, arm, output_dir):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=False)
        self.state = historical_state
        self.event = historical_event_receipt
        self.responses = historical_responses
        self.windows = historical_windows
        self.presentations = self.event.get("records", [])
        self.receipt = dict(schema=SCHEMA, status="running", arm=arm,
            responses_compared=0, windows_compared=0, presentations_compared=0,
            event_compared=False, endpoint_compared=False, categories={},
            historical_feature_identity_verified=False, first_divergence=None,
            comparison="exact scientific values; no numerical tolerance",
            exclusions=EXCLUSIONS, tensor_comparisons=0, extra_optimizer_steps=0,
            historical_event_counters_checked_separately=True)
        self._save()
        self._keys(self.state, STATE_FIELDS, "historical_state")
        event_view = self._event_view(self.event)
        if arm not in ("N", "S"):
            self._fail("arm", "N or S", arm)
        self._equal((len(self.responses), len(self.windows), len(self.presentations)),
                    (256, 128, 256), "historical_receipt_counts")
        self._equal((self.state["global_step"], self.state["presentations"], len(self.state["events"])),
                    (640, 1280, 5), "historical_boundary")
        self._equal(self.event["method"], arm, "historical_event.method")
        self._equal(self.event.get("status"), "completed", "historical_event.status")
        for index, row in enumerate(self.responses):
            self._response_schema(row, f"historical_response[{index}]")
        for index, row in enumerate(self.windows):
            self._keys(row, WINDOW_FIELDS, f"historical_window[{index}]")
        for index, row in enumerate(self.presentations):
            self._keys(row, PRESENTATION_FIELDS, f"historical_presentation[{index}]")
            self._equal(row["response_observation"], self.responses[index],
                        f"historical_presentation[{index}].response_observation")
        self._equal(self._event_view(self.state["events"][-1]), event_view,
                    "historical_state.event5_vs_receipt")
        self._save()

    def _save(self):
        write_json(self.root / "COMPARISON.json", self.receipt)

    def _fail(self, field, expected, actual):
        mismatch = dict(field=field, expected=_brief(expected), actual=_brief(actual))
        self.receipt.update(status="failed", first_divergence=mismatch)
        self._save()
        raise ValueError("Exact replay mismatch at " + field)

    def _keys(self, value, required, field, optional=frozenset()):
        if not isinstance(value, dict):
            self._fail(field + ".type", "dict", type(value).__name__)
        missing, unknown = required - value.keys(), value.keys() - required - optional
        if missing or unknown:
            self._fail(field + ".fields", sorted(required),
                       {"missing": sorted(missing), "unknown": sorted(unknown)})

    def _equal(self, actual, expected, field):
        import numpy as np
        import torch
        if isinstance(expected, (torch.Tensor, np.ndarray)):
            left, right = _fingerprint(actual), _fingerprint(expected)
            equal = left == right and left is not None and left["finite"]
            record = dict(field=field, equal=equal, actual=left, expected=right)
            with (self.root / "TENSOR_COMPARISONS.jsonl").open("a") as stream:
                import json
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            self.receipt["tensor_comparisons"] += 1
            if not equal:
                self._fail(field, right, left)
            return
        if type(actual) is not type(expected):
            self._fail(field + ".type", type(expected).__name__, type(actual).__name__)
        if isinstance(expected, dict):
            if actual.keys() != expected.keys():
                self._fail(field + ".fields", sorted(map(str, expected)), sorted(map(str, actual)))
            for key in sorted(expected, key=lambda item: (type(item).__name__, str(item))):
                self._equal(actual[key], expected[key], field + "." + str(key))
        elif isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                self._fail(field + ".length", len(expected), len(actual))
            for index, (a, e) in enumerate(zip(actual, expected)):
                self._equal(a, e, f"{field}[{index}]")
        elif isinstance(expected, float):
            if not math.isfinite(actual) or not math.isfinite(expected) or actual.hex() != expected.hex():
                self._fail(field, expected, actual)
        elif actual != expected:
            self._fail(field, expected, actual)

    def _presentation_view(self, row):
        self._keys(row, PRESENTATION_FIELDS, "presentation")
        self._keys(row["metrics"], METRIC_FIELDS | ({"grad_norm"} if row["optimizer_stepped"] else set()),
                   "presentation.metrics")
        self._response_schema(row["response_observation"], "presentation.response_observation")
        return {key: value for key, value in row.items() if key != "training_s"}

    def _response_schema(self, row, field):
        self._keys(row, RESPONSE_FIELDS, field)
        self._keys(row["ratio_metrics"], RATIO_FIELDS, field + ".ratio_metrics")
        self._keys(row["sum_metrics"], SUM_FIELDS, field + ".sum_metrics")
        for key, pair in row["ratio_metrics"].items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                self._fail(field + ".ratio_metrics." + key, "numerator/denominator pair", pair)

    def _event_view(self, event):
        self._keys(event, EVENT_FIELDS, "event", PUBLICATION_FIELDS | {"counters"})
        if "costs" in event:
            self._keys(event["costs"], COST_TIMES, "event.costs")
        if "counters" in event:
            self._equal(event["counters"], dict(global_step=640, presentations=1280, completed_events=5),
                        "event.counters")
        work = event["objective_work"]
        self._keys(work, {"work", "reference_operations"}, "event.objective_work", OBJECTIVE_TIMES)
        self._keys(work["work"], {"draft_forwards", "layer_forwards", "objective_chunks",
                   "vocabulary_head_calls", "response_forwards"}, "event.objective_work.work")
        result = {key: value for key, value in event.items() if key not in PUBLICATION_FIELDS | {"counters"}}
        result["objective_work"] = {key: value for key, value in work.items() if key not in OBJECTIVE_TIMES}
        result["records"] = [self._presentation_view(row) for row in event["records"]]
        return result

    def _observe(self, row, field, expected_rows, required):
        key = field + "s_compared"
        index = self.receipt[key]
        if self.receipt["status"] != "running" or index >= len(expected_rows):
            self._fail(field + ".unexpected_call", "one ordered observation per retained row", index)
        self._keys(row, required, f"{field}[{index}]")
        current, expected = row, expected_rows[index]
        if field == "presentation":
            current, expected = self._presentation_view(current), self._presentation_view(expected)
        self._equal(current, expected, f"{field}[{index}]")
        self.receipt[key] += 1
        self._save()

    def observe_response(self, row):
        self._observe(row, "response", self.responses, RESPONSE_FIELDS)

    def observe_window(self, row):
        self._observe(row, "window", self.windows, WINDOW_FIELDS)

    def observe_presentation(self, row):
        self._observe(row, "presentation", self.presentations, PRESENTATION_FIELDS)

    def compare_event(self, event):
        self._equal(tuple(self.receipt[key] for key in
            ("responses_compared", "windows_compared", "presentations_compared")),
            (256, 128, 256), "complete_observation_counts")
        self._equal(self._event_view(event), self._event_view(self.event), "event5")
        self.receipt["event_compared"] = True
        self.receipt["categories"]["event_scientific_receipts"] = "exact"
        self._save()
        return dict(status="passed", phase="event_scientific_receipts",
            responses_compared=256, windows_compared=128, presentations_compared=256,
            event_compared=True, endpoint_compared=False,
            full_gate_pending_endpoint=True)

    def compare_endpoint(self, live_state, trainer):
        import yaml
        self._keys(live_state, STATE_FIELDS, "endpoint")
        if not self.receipt["event_compared"]:
            self._fail("event_compared", True, False)
        for field, category in (("draft_state_dict", "draft_tensors"),
                                ("backend", "optimizer_moments_counters_options"), ("rng", "rng")):
            self._equal(live_state[field], self.state[field], "endpoint." + field)
            self.receipt["categories"][category] = "exact"
            self._save()
        self._equal(live_state["events"][:4], self.state["events"][:4], "original_event_ancestry")
        self._equal(len(live_state["events"]), 5, "endpoint.events.length")
        self._equal(self._event_view(live_state["events"][-1]), self._event_view(self.event), "endpoint.event5")
        self.receipt["categories"]["original_event_ancestry"] = "exact"
        special = {"draft_state_dict", "backend", "rng", "events", "config_text", "run_id"}
        self._equal({k:v for k,v in live_state.items() if k not in special},
                    {k:v for k,v in self.state.items() if k not in special}, "scientific_configuration")
        configs = [yaml.safe_load(state["config_text"]) for state in (live_state, self.state)]
        for config in configs:
            for key in ("run_id", "output_dir"):
                if key not in config:
                    self._fail("config_text." + key, "declared configuration field", "missing")
                config.pop(key)
        self._equal(configs[0], configs[1], "scientific_config_text")
        self.receipt["categories"]["scientific_configuration"] = "exact"
        model = trainer.bundle.model
        actual = dict(global_step=trainer.global_step, presentations=trainer.presentations,
            micro=trainer.core._micro, accumulation=trainer.core.accumulation_remainder,
            ratio_totals=dict(trainer.core._ratio_totals), sum_totals=dict(trainer.core._sum_totals),
            schedule_absent=trainer._presentation_schedule is None, training=bool(model.training),
            reference_absent=model.reference is None, ranking_enabled=bool(model.ranking_enabled),
            objective=model.objective, closed=bool(trainer.closed), failed=bool(trainer.failed),
            gradients_clear=all(parameter.grad is None for parameter in trainer.bundle.draft_model.parameters()))
        expected = dict(global_step=640, presentations=1280, micro=1280, accumulation=0,
            ratio_totals={}, sum_totals={}, schedule_absent=True, training=True, reference_absent=True,
            ranking_enabled=False, objective="osd_first_error", closed=False, failed=False, gradients_clear=True)
        self._equal(actual, expected, "sampler_accumulation_boundary")
        self._equal(trainer._anchor_observations, self.presentations[-1]["sampled_anchors"], "last_natural_anchors")
        try:
            trainer._check_native_path()
            trainer._check_actual_lr()
        except ValueError as error:
            self._fail("sampler_accumulation_boundary.native_path", "unchanged PR51 native path", str(error))
        self.receipt["categories"]["sampler_accumulation_boundary"] = "exact"
        self.receipt.update(status="passed", endpoint_compared=True)
        self._save()
        return self.receipt
