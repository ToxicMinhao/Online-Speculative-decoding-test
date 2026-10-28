"""Persistent exact-token feedback for one synchronous online runner.

One JSON file holds serving records in arrival order and their update events.
Pending, sealed, and failed records count toward capacity. An unfinished event
keeps its records reserved and blocks further updates, including after restart.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .storage import write_json


class FeedbackError(ValueError):
    """Invalid feedback or event transition."""


class BufferFullError(FeedbackError):
    """Outstanding feedback has reached the configured capacity."""


SCHEMA_VERSION = 2


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise FeedbackError(f"{name} must be a positive integer")


def _identifier(value, name):
    if not isinstance(value, str) or not value:
        raise FeedbackError(f"{name} must be a nonempty string")


class FeedbackBuffer:
    """Preserve chronological serving records and publication status.

    Generation validation belongs to the backend; this buffer checks ownership,
    exact token types, ordering, and capacity. Public reads return copies. There
    is no concurrent-writer or automatic recovery protocol. Use a fresh directory
    for this schema; old run directories remain evidence for the old code.
    """

    def __init__(self, root: Path, capacity: int):
        _positive_integer(capacity, "capacity")
        self.root = Path(root)
        self.capacity = capacity
        self._state_path = self.root / "state.json"
        if not self._state_path.exists():
            self._save({
                "schema_version": SCHEMA_VERSION,
                "capacity": capacity,
                "records": [],
                "events": [],
            })
        self._load()

    def _load(self):
        state = json.loads(self._state_path.read_text())
        if state.get("schema_version") != SCHEMA_VERSION:
            raise FeedbackError("Unsupported feedback schema; use a fresh output directory")
        if state["capacity"] != self.capacity:
            raise FeedbackError("buffer capacity does not match saved configuration")
        return state

    def _save(self, state):
        write_json(self._state_path, state)

    @staticmethod
    def _pending_records(state):
        reserved = {record_id for event in state["events"] for record_id in event["record_ids"]}
        return [record for record in state["records"] if record["record_id"] not in reserved]

    @staticmethod
    def _find_event(state, event_id):
        for event in state["events"]:
            if event["event_id"] == event_id:
                return event
        raise KeyError(event_id)

    def add(self, record: dict):
        """Append a completed serving record without decoding its token IDs."""
        if record["purpose"] != "serving":
            raise FeedbackError("only serving feedback may enter the training buffer")
        for name in ("record_id", "session_id", "checkpoint_id"):
            _identifier(record[name], name)
        if type(record["request_index"]) is not int or record["request_index"] < 0:
            raise FeedbackError("request_index must be a nonnegative integer")
        for name in ("prompt_ids", "response_ids"):
            tokens = record[name]
            if not isinstance(tokens, list) or not tokens:
                raise FeedbackError(f"{name} must be a nonempty list of integer token IDs")
            if any(type(token) is not int or token < 0 for token in tokens):
                raise FeedbackError(f"{name} must contain nonnegative integer token IDs")

        state = self._load()
        for previous in state["records"]:
            if previous["record_id"] == record["record_id"]:
                raise FeedbackError(f"duplicate record_id: {record['record_id']}")
            if (previous["session_id"] == record["session_id"]
                    and previous["request_index"] >= record["request_index"]):
                raise FeedbackError("request_index must increase within a session")
        committed_count = sum(
            len(event["record_ids"])
            for event in state["events"] if event["status"] == "committed"
        )
        if len(state["records"]) - committed_count >= self.capacity:
            raise BufferFullError("outstanding feedback reached capacity")
        state["records"].append(record)
        self._save(state)
        return copy.deepcopy(record)

    def pending(self) -> list[dict]:
        return self._pending_records(self._load())

    def records(self, ids) -> list[dict]:
        """Read selected records in their original arrival order."""
        state = self._load()
        by_id = {record["record_id"]: record for record in state["records"]}
        records = [by_id[record_id] for record_id in ids]
        selected = set(ids)
        chronological_ids = [
            record["record_id"] for record in state["records"]
            if record["record_id"] in selected
        ]
        if chronological_ids != ids:
            raise FeedbackError("requested records must be unique and in arrival order")
        return records

    def seal_event(self, event_id, count) -> dict:
        """Reserve the earliest pending records before training can begin."""
        _identifier(event_id, "event_id")
        _positive_integer(count, "count")
        state = self._load()
        if any(event["event_id"] == event_id for event in state["events"]):
            raise FeedbackError("duplicate event_id")
        if any(event["status"] != "committed" for event in state["events"]):
            raise FeedbackError("an unfinished/failed event blocks further updates; inspect the saved event")
        pending = self._pending_records(state)
        if count > len(pending):
            raise FeedbackError("insufficient pending records")
        event = {
            "event_id": event_id,
            "status": "sealed",
            "record_ids": [record["record_id"] for record in pending[:count]],
            "checkpoint_id": None,
            "receipt": None,
            "error": None,
        }
        state["events"].append(event)
        self._save(state)
        return event

    def event(self, event_id):
        return self._find_event(self._load(), event_id)

    def commit_event(self, event_id, checkpoint_id, receipt):
        """Mark publication complete; the controller must publish first."""
        _identifier(checkpoint_id, "checkpoint_id")
        state = self._load()
        event = self._find_event(state, event_id)
        if event["status"] == "committed":
            if event["checkpoint_id"] == checkpoint_id and event["receipt"] == receipt:
                return event
            raise FeedbackError("committed event cannot be changed")
        if event["status"] != "sealed":
            raise FeedbackError("only a sealed event can commit")
        event.update(status="committed", checkpoint_id=checkpoint_id, receipt=copy.deepcopy(receipt))
        self._save(state)
        return event

    def fail_event(self, event_id, error):
        """Keep an unsuccessful event reserved; never requeue it for training."""
        _identifier(error, "error")
        state = self._load()
        event = self._find_event(state, event_id)
        if event["status"] == "failed" and event["error"] == error:
            return event
        if event["status"] != "sealed":
            raise FeedbackError("only a sealed event can fail")
        event.update(status="failed", error=error)
        self._save(state)
        return event
