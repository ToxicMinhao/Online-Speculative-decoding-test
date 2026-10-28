"""Online-only recipe choices, separate from the unchanged SpecForge config."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml

class OnlineRecipe(BaseModel):
    # Defaults preserve the historical configs that predate this recipe.
    model_config = ConfigDict(extra="forbid")
    optimizer: Literal["fp32_master", "bf16_adamw"] = "fp32_master"
    weight_decay: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    shuffle_responses: bool = False
    kl_precision: Literal["float32", "bfloat16"] = "float32"
    responses_per_event: int | None = Field(default=None, ge=1)
    passes_per_event: int = Field(default=2, ge=1)
    max_events: int | None = Field(default=None, ge=1)


def load_event_config(path):
    """Remove only our options, then use the author's migration and validation."""
    from specforge.config import Config, migrate_legacy_config

    path = Path(path)
    raw = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    recipe = OnlineRecipe.model_validate(raw.pop("online_feedback", {}))
    config = Config.model_validate(migrate_legacy_config(raw))
    if recipe.max_events is not None:
        if recipe.responses_per_event is None:
            raise ValueError("A bounded event schedule requires responses_per_event")
        presentations = recipe.responses_per_event * recipe.passes_per_event
        steps, remainder = divmod(presentations, config.training.accumulation_steps)
        if remainder or steps * recipe.max_events != config.training.max_steps:
            raise ValueError("Online event schedule and optimizer horizon disagree")
    return config, recipe
