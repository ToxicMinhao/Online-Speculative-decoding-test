"""Explicit configuration for our single method; no experiment-arm registry."""

from pathlib import Path
import tomllib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    # Validate once at the input boundary. Misspelled options must not be ignored.
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class ModelSettings(Settings):
    target: Path = Field(strict=False)
    draft: Path = Field(strict=False)


class FeedbackSettings(Settings):
    manifest: Path = Field(strict=False)
    buffer_dir: Path = Field(strict=False)


class CollectionSettings(Settings):
    requests: Path = Field(strict=False)
    output_dir: Path = Field(strict=False)


class TrainingSettings(Settings):
    output_dir: Path = Field(strict=False)
    seed: int = Field(ge=0)
    responses_per_event: int = Field(gt=0)
    passes_per_event: int = Field(gt=0)
    microbatch_size: int = Field(ge=1, le=1)
    accumulation_steps: int = Field(gt=0)
    context_limit: int = Field(gt=0)
    max_anchors_per_response: int = Field(gt=0)
    projection_chunk_blocks: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    loss_decay_gamma: float = Field(gt=0)
    shuffle_responses: bool
    parameter_dtype: Literal["bfloat16"]
    probability_dtype: Literal["bfloat16"]
    optimizer: Literal["bf16_adamw"]
    learning_rate_schedule: Literal["constant"]
    attention_backend: Literal["flex_attention"]

    @model_validator(mode="after")
    def check_accumulation(self):
        if self.presentations_per_event % self.accumulation_steps:
            raise ValueError("The complete event must contain whole accumulation windows.")
        return self

    @property
    def presentations_per_event(self):
        return self.responses_per_event * self.passes_per_event


class ObjectiveSettings(Settings):
    # These identify the existing method, rather than introducing tuning options.
    first_error_redistribution_fraction: Literal[0.5]
    prefix_probability_smoothing: Literal[0.5]
    prefix_relative_cap: Literal[0.25]
    denominator_epsilon: Literal[1e-6]


class ServingSettings(Settings):
    runtime: Literal["sglang"]
    context_limit: int = Field(gt=0)
    block_size: int = Field(gt=1)
    max_new_tokens: int = Field(gt=0)
    concurrency: int = Field(ge=1, le=1)
    greedy: bool
    parameter_dtype: Literal["bfloat16"]
    attention_backend: Literal["fa3"]
    memory_fraction: float = Field(gt=0, lt=1)
    chunked_prefill_tokens: int = Field(gt=0)
    cuda_graphs: bool
    cuda_graph_max_batch_size: int = Field(gt=0)


class Configuration(Settings):
    models: ModelSettings
    training: TrainingSettings
    objective: ObjectiveSettings
    serving: ServingSettings
    feedback: FeedbackSettings
    collection: CollectionSettings

    @model_validator(mode="after")
    def check_consistency(self):
        if not (self.training.shuffle_responses and self.serving.greedy and self.serving.cuda_graphs):
            raise ValueError("This recipe requires response shuffling, greedy decoding and CUDA graphs.")
        if self.training.context_limit != self.serving.context_limit:
            raise ValueError("Training and serving must retain the same full context.")
        if self.serving.max_new_tokens >= self.serving.context_limit:
            raise ValueError("The context must leave room for the prompt.")
        if self.serving.block_size > self.serving.context_limit:
            raise ValueError("A draft block must fit within the context limit.")
        return self


def load_config(path: Path) -> Configuration:
    """Read TOML; resolve file paths relative to it, never the working directory."""
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        config = Configuration.model_validate(tomllib.load(stream))

    def resolve_path(file_path):
        return (path.parent / file_path.expanduser()).resolve()

    updates = {
        "models": ModelSettings(
            target=resolve_path(config.models.target),
            draft=resolve_path(config.models.draft),
        )
    }
    updates["training"] = config.training.model_copy(update={
        "output_dir": resolve_path(config.training.output_dir),
    })
    updates["feedback"] = FeedbackSettings(
        manifest=resolve_path(config.feedback.manifest),
        buffer_dir=resolve_path(config.feedback.buffer_dir),
    )
    updates["collection"] = CollectionSettings(
        requests=resolve_path(config.collection.requests),
        output_dir=resolve_path(config.collection.output_dir),
    )
    return config.model_copy(update=updates)
