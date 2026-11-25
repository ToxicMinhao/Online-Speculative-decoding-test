"""BASE/PAIR use the exact native C recipe; pilot length is a runner boundary."""

from experiments.opencode_native_comparison.configuration import (
    build_config as build_native_config,
    validate_config_text as validate_native_config,
)


def validate_config_text(config_text, original_config_text, *, maximum_events=2):
    if type(maximum_events) is not int or maximum_events not in (1, 2):
        raise ValueError("Repair conditional KL pilot requires one or two declared events")
    receipt = validate_native_config(config_text, original_config_text)
    return {**receipt, "pilot_maximum_events": maximum_events,
            "native_horizon_preserved": True}


def build_config(original_config_text, *, arm, output_dir, maximum_events=2, run_id=None):
    if arm not in ("BASE", "PAIR"):
        raise ValueError("Repair conditional KL pilot arm must be BASE or PAIR")
    config_text = build_native_config(
        original_config_text, arm="C_CODE58", output_dir=output_dir,
        run_id=run_id or "repair-conditional-kl-" + arm.lower(),
    )
    validate_config_text(config_text, original_config_text, maximum_events=maximum_events)
    return config_text
