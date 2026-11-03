"""CPU-only, outcome-blind selection and durable Algorithm R memory metadata.

The caller owns immutable prepared packets and verifies all arm commits before
insertion. This module accepts only identities, chronology and prompt descriptors;
domain composition is joined to its receipts by reporting code, after selection.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np


SEED = 20261102
EVENT_SIZE = 128
MEMORY_CAPACITY = 128
FRESH_COUNT = 96
REPLAY_COUNT = 32
ARMS = ("K", "U", "R")
ARITHMETIC = {
    "dtype": "float64",
    "similarity": "numpy.einsum('qd,md->qm', optimize=False), clip(-1,1), maximum(0)",
    "coverage": "numpy.sum(per_query_maximum, dtype=float64) / 128",
    "gain": "candidate_coverage - previous_coverage",
    "tie_break": "exact numeric equality, smaller global_ordinal",
    "numpy_version": np.__version__,
}


@dataclass(frozen=True)
class ReplayRecord:
    """Compact immutable metadata; no feature packet, label or outcome access."""

    record_id: str
    global_ordinal: int
    event_index: int
    descriptor: tuple[float, ...]

    def __post_init__(self):
        descriptor = tuple(float(value) for value in self.descriptor)
        object.__setattr__(self, "descriptor", descriptor)
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("A nonempty original record_id is required")
        if not isinstance(self.global_ordinal, int) or not 0 <= self.global_ordinal < 1024:
            raise ValueError("global_ordinal must be an integer in 0..1023")
        if self.event_index != self.global_ordinal // EVENT_SIZE + 1:
            raise ValueError("event_index disagrees with global fresh-record chronology")
        values = np.asarray(descriptor, dtype=np.float64)
        norm = float(np.linalg.norm(values))
        if not descriptor or not np.isfinite(values).all() or not np.isfinite(norm):
            raise ValueError("Descriptor must contain finite values")
        if abs(norm - 1.0) > 1e-12:
            raise ValueError("Descriptor must already be normalized in float64")


def prompt_descriptor(target_last_hidden_states, prompt_length: int):
    """Read only the last prompt row already present in a prepared CPU packet."""
    started = perf_counter()
    shape = target_last_hidden_states.shape
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError("Prepared target hidden states must have shape [1, tokens, hidden]")
    if not isinstance(prompt_length, int) or not 1 <= prompt_length <= shape[1]:
        raise ValueError("prompt_length must identify a present last prompt token")
    row = target_last_hidden_states[0, prompt_length - 1, :]
    if hasattr(row, "detach"):
        import torch

        if row.device.type != "cpu":
            raise ValueError("Descriptor extraction requires an already-loaded CPU packet")
        row = row.detach().to(dtype=torch.float64).numpy()
    values = np.array(row, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Last prompt hidden state contains nonfinite values")
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Last prompt hidden state must have finite positive norm")
    normalized = values / norm
    descriptor = tuple(float(value) for value in normalized)
    receipt = {
        "prompt_length": prompt_length,
        "row_index": prompt_length - 1,
        "hidden_dimensions": int(values.size),
        "all_finite": True,
        "original_norm": norm,
        "normalized_norm": float(np.linalg.norm(normalized)),
        "dtype": "float64",
        "descriptor_bytes": int(normalized.nbytes),
        "array_working_bytes": int(values.nbytes + normalized.nbytes),
        "wall_seconds": perf_counter() - started,
    }
    return descriptor, receipt


def _ordered_unique(records: Sequence[ReplayRecord]) -> tuple[ReplayRecord, ...]:
    ordered = tuple(sorted(records, key=lambda record: record.global_ordinal))
    if len({record.record_id for record in ordered}) != len(ordered):
        raise ValueError("Repeated original record_id")
    if len({record.global_ordinal for record in ordered}) != len(ordered):
        raise ValueError("Repeated global_ordinal")
    return ordered


def _fresh_event(event_index: int, records: Sequence[ReplayRecord]):
    ordered = _ordered_unique(records)
    expected = list(range((event_index - 1) * EVENT_SIZE, event_index * EVENT_SIZE))
    if not 1 <= event_index <= 8 or [record.global_ordinal for record in ordered] != expected:
        raise ValueError("Fresh event must contain exactly its 128 original global ordinals")
    return ordered


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "quantiles_linear": {
            str(percentile): float(np.percentile(values, percentile, method="linear"))
            for percentile in (0, 25, 50, 75, 100)
        },
        "zero_count": int(np.count_nonzero(values == 0)),
    }


def _record_receipt(record: ReplayRecord, event_index: int) -> dict:
    return {
        "record_id": record.record_id,
        "global_ordinal": record.global_ordinal,
        "event_index": record.event_index,
        "age_reference_event_index": event_index,
        "age_events": event_index - record.event_index,
        "age_fresh_records": (event_index - 1) * EVENT_SIZE - record.global_ordinal,
    }


def _selection_receipt(similarities, indices, memory, event_index):
    covered = np.zeros(EVENT_SIZE, dtype=np.float64)
    previous_score = 0.0
    steps = []
    for step, index in enumerate(indices, 1):
        covered = np.maximum(covered, similarities[:, index])
        score = float(np.sum(covered, dtype=np.float64) / EVENT_SIZE)
        steps.append({
            "step": step,
            **_record_receipt(memory[index], event_index),
            "coverage": score,
            "marginal_gain": score - previous_score,
        })
        previous_score = score

    # Assign a positive best similarity to its smallest-ordinal maximizer.
    # Zero-coverage queries are explicitly unassigned, not credited to a record.
    ordered_indices = sorted(indices, key=lambda index: memory[index].global_ordinal)
    selected = similarities[:, ordered_indices]
    positive = covered > 0
    winners = np.argmax(selected, axis=1)
    counts = np.bincount(winners[positive], minlength=REPLAY_COUNT)
    assigned = int(np.count_nonzero(positive))
    shares = counts / assigned if assigned else np.zeros(REPLAY_COUNT)
    return {
        "record_ids": [memory[index].record_id for index in indices],
        "global_ordinals": [memory[index].global_ordinal for index in indices],
        "coverage": previous_score,
        "steps": steps,
        "similarity_distribution": _distribution(selected),
        "best_similarity_distribution": _distribution(covered),
        "concentration": {
            "assignment_tie_break": "smallest global_ordinal among exact positive maxima",
            "assigned_queries": assigned,
            "zero_coverage_queries": EVENT_SIZE - assigned,
            "maximum_share": float(np.max(shares)),
            "herfindahl": float(np.sum(shares * shares, dtype=np.float64)),
            "effective_records": float(1 / np.sum(shares * shares)) if assigned else 0.0,
            "record_query_counts": [
                {"record_id": memory[index].record_id, "count": int(counts[position])}
                for position, index in enumerate(ordered_indices)
            ],
        },
        "ages_events": _distribution(np.array([event_index - memory[i].event_index for i in indices])),
        "ages_fresh_records": _distribution(np.array([
            (event_index - 1) * EVENT_SIZE - memory[i].global_ordinal for i in indices
        ])),
    }


def build_event_plan(event_index, fresh_records, memory_records):
    """Allocate the fixed two passes and return auditable JSON-ready selections."""
    started = perf_counter()
    fresh = _fresh_event(event_index, fresh_records)
    memory = _ordered_unique(memory_records)
    if any(record.event_index >= event_index for record in memory):
        raise ValueError("Replay memory may contain only strictly past fresh records")
    if {record.record_id for record in fresh} & {record.record_id for record in memory}:
        raise ValueError("Fresh and memory identities overlap")
    if len(memory) != (0 if event_index == 1 else MEMORY_CAPACITY):
        raise ValueError("Event 1 requires empty memory; later events require all 128 slots")
    dimensions = {len(record.descriptor) for record in fresh + memory}
    if len(dimensions) != 1:
        raise ValueError("All full-width prompt descriptors must share one dimension")

    first_pass = [record.record_id for record in fresh]
    first_ordinals = [record.global_ordinal for record in fresh]
    result = {
        "event_index": event_index,
        "arithmetic": deepcopy(ARITHMETIC),
        "seed_sequences": {
            "fresh": [SEED, event_index, 0],
            "uniform_old": [SEED, event_index, 1],
            "reservoir": [SEED, 0, 2],
        },
        "pass_record_ids": {arm: [list(first_pass), list(first_pass)] for arm in ARMS},
        "pass_global_ordinals": {arm: [list(first_ordinals), list(first_ordinals)] for arm in ARMS},
        "memory_records": [_record_receipt(record, event_index) for record in memory],
        "common_fresh_record_ids": list(first_pass),
        "uniform": None,
        "coverage": None,
    }
    if event_index == 1:
        result["cost"] = {"scoring_seconds": 0.0, "greedy_seconds": 0.0,
                          "array_working_bytes": 0, "total_wall_seconds": perf_counter() - started}
        return result

    fresh_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, event_index, 0])))
    uniform_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, event_index, 1])))
    fresh_indices = sorted(int(i) for i in fresh_rng.choice(EVENT_SIZE, FRESH_COUNT, replace=False))
    uniform_indices = [int(i) for i in uniform_rng.choice(MEMORY_CAPACITY, REPLAY_COUNT, replace=False)]
    common_fresh = [fresh[index] for index in fresh_indices]
    result["common_fresh_record_ids"] = [record.record_id for record in common_fresh]

    scoring_started = perf_counter()
    queries = np.asarray([record.descriptor for record in fresh], dtype=np.float64)
    candidates = np.asarray([record.descriptor for record in memory], dtype=np.float64)
    similarities = np.einsum("qd,md->qm", queries, candidates, optimize=False, dtype=np.float64)
    np.clip(similarities, -1.0, 1.0, out=similarities)
    np.maximum(similarities, 0.0, out=similarities)
    scoring_seconds = perf_counter() - scoring_started

    greedy_started = perf_counter()
    covered = np.zeros(EVENT_SIZE, dtype=np.float64)
    remaining = list(range(MEMORY_CAPACITY))
    selected = []
    for _ in range(REPLAY_COUNT):
        best_index = remaining[0]
        best_score = -1.0
        for index in remaining:
            candidate_score = float(np.sum(np.maximum(covered, similarities[:, index]), dtype=np.float64) / EVENT_SIZE)
            if candidate_score > best_score:
                best_index = index
                best_score = candidate_score
        selected.append(best_index)
        remaining.remove(best_index)
        covered = np.maximum(covered, similarities[:, best_index])
    greedy_seconds = perf_counter() - greedy_started

    for arm, indices in (("U", uniform_indices), ("R", selected)):
        second_pass = sorted(common_fresh + [memory[index] for index in indices],
                             key=lambda record: record.global_ordinal)
        result["pass_record_ids"][arm][1] = [record.record_id for record in second_pass]
        result["pass_global_ordinals"][arm][1] = [record.global_ordinal for record in second_pass]
    result["uniform"] = _selection_receipt(similarities, uniform_indices, memory, event_index)
    result["coverage"] = _selection_receipt(similarities, selected, memory, event_index)
    result["memory_similarity_distribution"] = _distribution(similarities)
    result["descriptor_checks"] = {
        "dimensions": queries.shape[1],
        "all_finite": True,
        "fresh_norms": _distribution(np.linalg.norm(queries, axis=1)),
        "memory_norms": _distribution(np.linalg.norm(candidates, axis=1)),
    }
    result["cost"] = {
        "scoring_seconds": scoring_seconds,
        "greedy_seconds": greedy_seconds,
        "array_working_bytes": int(queries.nbytes + candidates.nbytes + similarities.nbytes + covered.nbytes),
        "working_bytes_scope": "persistent numeric arrays; excludes transient arrays, Python metadata and allocator overhead",
        "total_wall_seconds": perf_counter() - started,
    }
    return result


class ReservoirMemory:
    """Fixed 128-slot Algorithm R state, advanced once per verified fresh event."""

    def __init__(self):
        self._rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, 0, 2])))
        self._slots: list[ReplayRecord] = []
        self._seen_ids: list[str] = []
        self.completed_event = 0

    @property
    def records(self) -> tuple[ReplayRecord, ...]:
        """Return the slot order; selectors canonicalize candidates by ordinal."""
        return tuple(self._slots)

    @property
    def seen_count(self) -> int:
        return len(self._seen_ids)

    def insert_after_event(self, event_index, fresh_records, *, verified_arms: Iterable[str]):
        if set(verified_arms) != set(ARMS):
            raise ValueError("All three arms must have verified commits before memory insertion")
        if event_index != self.completed_event + 1:
            raise ValueError("Insert each fresh event once, in chronological order")
        fresh = _fresh_event(event_index, fresh_records)
        if {record.record_id for record in fresh} & set(self._seen_ids):
            raise ValueError("A fresh record may be admitted only once, even after eviction")
        before = [_record_receipt(record, event_index) for record in self._slots]
        insertions = []
        for record in fresh:
            self._seen_ids.append(record.record_id)
            count = self.seen_count
            slot = count - 1 if count <= MEMORY_CAPACITY else int(self._rng.integers(0, count))
            evicted = None
            if slot < MEMORY_CAPACITY:
                if count <= MEMORY_CAPACITY:
                    self._slots.append(record)
                else:
                    evicted = self._slots[slot].record_id
                    self._slots[slot] = record
            insertions.append({
                "record_id": record.record_id,
                "global_ordinal": record.global_ordinal,
                "seen_count": count,
                "draw": slot if count > MEMORY_CAPACITY else None,
                "admitted_slot": slot if slot < MEMORY_CAPACITY else None,
                "evicted_record_id": evicted,
            })
        self.completed_event = event_index
        return {
            "event_index": event_index,
            "verified_arms": list(ARMS),
            "before_slots": before,
            "insertions": insertions,
            "after_slots": [_record_receipt(record, event_index + 1) for record in self._slots],
            "seen_count": self.seen_count,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def state_dict(self):
        return {
            "schema": "track2-reservoir-v1",
            "capacity": MEMORY_CAPACITY,
            "seed_sequence": [SEED, 0, 2],
            "completed_event": self.completed_event,
            "seen_count": self.seen_count,
            "last_global_ordinal": self.seen_count - 1,
            "seen_record_ids": list(self._seen_ids),
            "slots": [asdict(record) for record in self._slots],
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state):
        if state["schema"] != "track2-reservoir-v1" or state["capacity"] != MEMORY_CAPACITY:
            raise ValueError("Reservoir schema or fixed capacity differs")
        if state["seed_sequence"] != [SEED, 0, 2]:
            raise ValueError("Reservoir seed differs from the fixed campaign")
        result = cls()
        result.completed_event = state["completed_event"]
        result._seen_ids = list(state["seen_record_ids"])
        result._slots = [ReplayRecord(**row) for row in state["slots"]]
        count = result.seen_count
        if not 0 <= result.completed_event <= 8 or count != result.completed_event * EVENT_SIZE:
            raise ValueError("Reservoir event/counter continuity differs")
        if state["seen_count"] != count or state["last_global_ordinal"] != count - 1:
            raise ValueError("Reservoir saved counter or chronology differs")
        if len(set(result._seen_ids)) != count or len(result._slots) != min(count, MEMORY_CAPACITY):
            raise ValueError("Reservoir identities or slot count differ")
        _ordered_unique(result._slots)
        for record in result._slots:
            if record.global_ordinal >= count or result._seen_ids[record.global_ordinal] != record.record_id:
                raise ValueError("Reservoir slot is not bound to an inserted original record")
        result._rng.bit_generator.state = deepcopy(state["rng_state"])
        return result
