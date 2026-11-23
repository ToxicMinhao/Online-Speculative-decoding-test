"""Persistent serving feedback and publication states, adapted from DFlash-Online.

One writer owns the buffer and loads its state once. Failed or sealed events stay
reserved across restart; there is no automatic retry, reset, or migration.
"""

from copy import deepcopy
import json
from pathlib import Path

from graduate.feedback import ResponseRecord
from graduate.storage import write_json


class FeedbackError(ValueError):
    """Invalid feedback ownership, ordering, or event transition."""


class BufferFullError(FeedbackError):
    """Pending, sealed, and failed feedback have reached capacity."""


class FeedbackBuffer:
    def __init__(self, root: Path, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise FeedbackError("capacity must be a positive integer")
        self.capacity = capacity
        self.state_path = Path(root) / "state.json"
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state["schema"] != "graduate_feedback_v1":
                raise FeedbackError("Unsupported feedback schema; use a fresh directory")
            if state["capacity"] != capacity:
                raise FeedbackError("Buffer capacity differs from its saved configuration")
            records = [ResponseRecord.model_validate(row) for row in state["records"]]
            self._records = {record.record_id: record for record in records}
            self._events = {event["event_id"]: event for event in state["events"]}
        else:
            self._save({}, {})

    def _save(self, records, events):
        # Adopt a replacement only after the complete state reaches disk.
        write_json(self.state_path, {
            "schema": "graduate_feedback_v1", "capacity": self.capacity,
            "records": [record.model_dump(mode="json") for record in records.values()],
            "events": list(events.values()),
        })
        self._records, self._events = records, events

    def _pending(self):
        reserved = {record_id for event in self._events.values() for record_id in event["record_ids"]}
        return [record for record_id, record in self._records.items() if record_id not in reserved]

    def add(self, record: ResponseRecord) -> ResponseRecord:
        """Import a serving response once; reject changed content under the same ID."""
        previous = self._records.get(record.record_id)
        if previous is not None:
            # JSON comparison also distinguishes changed numeric/boolean metadata.
            saved = json.dumps(previous.model_dump(mode="json"), sort_keys=True, allow_nan=False)
            incoming = json.dumps(record.model_dump(mode="json"), sort_keys=True, allow_nan=False)
            if saved != incoming:
                raise FeedbackError("Changed content for existing record_id: " + record.record_id)
            return previous.model_copy(deep=True)
        if any(previous.session_id == record.session_id
               and previous.request_index >= record.request_index
               for previous in self._records.values()):
            raise FeedbackError("request_index must increase within a session")
        committed_count = sum(len(event["record_ids"]) for event in self._events.values()
                              if event["status"] == "committed")
        if len(self._records) - committed_count >= self.capacity:
            raise BufferFullError("Outstanding feedback reached capacity")
        self._save({**self._records, record.record_id: record.model_copy(deep=True)}, self._events)
        return record.model_copy(deep=True)

    def pending(self) -> list[ResponseRecord]:
        return [record.model_copy(deep=True) for record in self._pending()]

    def records(self, ids: list[str]) -> list[ResponseRecord]:
        """Return unique selected responses in their original arrival order."""
        records = [self._records[record_id] for record_id in ids]
        selected = set(ids)
        if [record_id for record_id in self._records if record_id in selected] != ids:
            raise FeedbackError("Record IDs must be unique and in arrival order")
        return [record.model_copy(deep=True) for record in records]

    def seal_event(self, event_id: str, count: int):
        """Reserve the earliest pending responses before preparation or training."""
        if not event_id:
            raise FeedbackError("event_id must not be empty")
        if type(count) is not int or count < 1:
            raise FeedbackError("count must be a positive integer")
        if event_id in self._events:
            raise FeedbackError("Duplicate event_id")
        if any(event["status"] != "committed" for event in self._events.values()):
            raise FeedbackError("An unfinished or failed event blocks further updates")
        pending = self._pending()
        if count > len(pending):
            raise FeedbackError("Insufficient pending responses")
        event = {
            "event_id": event_id, "status": "sealed",
            "record_ids": [record.record_id for record in pending[:count]],
            "checkpoint_id": None, "receipt": None, "error": None,
        }
        self._save(self._records, {**self._events, event_id: event})
        return deepcopy(event)

    def event(self, event_id):
        return deepcopy(self._events[event_id])

    def commit_event(self, event_id, checkpoint_id, receipt):
        """Record successful publication; the caller must publish first."""
        if not checkpoint_id:
            raise FeedbackError("checkpoint_id must not be empty")
        event = self.event(event_id)
        if event["status"] == "committed":
            if event["checkpoint_id"] == checkpoint_id and event["receipt"] == receipt:
                return event
            raise FeedbackError("A committed event cannot be changed")
        if event["status"] != "sealed":
            raise FeedbackError("Only a sealed event can commit")
        event.update(status="committed", checkpoint_id=checkpoint_id, receipt=deepcopy(receipt))
        self._save(self._records, {**self._events, event_id: event})
        return deepcopy(event)

    def fail_event(self, event_id, error):
        """Preserve a failed event without returning its responses to pending."""
        if not error:
            raise FeedbackError("error must not be empty")
        event = self.event(event_id)
        if event["status"] == "failed" and event["error"] == error:
            return event
        if event["status"] != "sealed":
            raise FeedbackError("Only a sealed event can fail")
        event.update(status="failed", error=error)
        self._save(self._records, {**self._events, event_id: event})
        return deepcopy(event)
