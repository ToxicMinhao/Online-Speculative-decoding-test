"""The exact completed response that may become feedback for a later update."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator


class ResponseRecord(BaseModel):
    """Keep generated token IDs and provenance; never decode and retokenize them."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    record_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    request_index: int = Field(ge=0)
    checkpoint_id: str = Field(min_length=1)
    purpose: Literal["serving"]
    prompt_ids: tuple[NonNegativeInt, ...] = Field(min_length=1, strict=False)
    response_ids: tuple[NonNegativeInt, ...] = Field(min_length=1, strict=False)
    finish_reason: Literal["eos", "length"]
    stop_token_ids: tuple[NonNegativeInt, ...] = Field(min_length=1, strict=False)
    max_new_tokens: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_termination(self):
        if len(self.response_ids) > self.max_new_tokens:
            raise ValueError("Response exceeds its original output cap.")
        if any(token in self.stop_token_ids for token in self.response_ids[:-1]):
            raise ValueError("Response continues after its first EOS.")
        ended_at_eos = self.response_ids[-1] in self.stop_token_ids
        if (self.finish_reason == "eos") != ended_at_eos:
            raise ValueError("Termination disagrees with the final token.")
        if not ended_at_eos and len(self.response_ids) != self.max_new_tokens:
            raise ValueError("Response ended before EOS or the output cap.")
        return self

    @property
    def input_ids(self):
        return self.prompt_ids + self.response_ids

    @property
    def eligible_anchor_count(self):
        # Each seed and its immediate successor must both be response tokens.
        return len(self.response_ids) - 1


def read_response(path: Path) -> ResponseRecord:
    """Accept a saved serving record; evaluation records are rejected by the schema."""
    return ResponseRecord.model_validate_json(path.read_text())
