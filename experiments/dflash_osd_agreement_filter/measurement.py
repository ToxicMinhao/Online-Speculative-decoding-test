"""Reusable receipt for the qualified native measurement contract."""
from __future__ import annotations

import math
import struct

from experiments.dflash_per_turn.experiment import validate_measurement


def measurement_validation(output):
    """Run the existing validator; never infer validity from a saved status.

    The qualified validator fails on its first error. Preserve that exact error
    in a serializable receipt so execution and reporting use the same contract.
    """
    try:
        validate_measurement(output)
    except (ValueError, TypeError, OverflowError) as error:
        return {"status": "failed", "errors": [str(error)]}
    return {"status": "passed", "errors": []}


def preserve_invalid_numbers(value):
    """Make a rejected receipt JSON-safe without turning invalid costs valid.

    Explicit markers keep NaN/Inf visible; the IEEE-754 bytes preserve even the
    sign and payload of a NaN. Call only after validation failed. These markers
    are not numbers and cannot pass the qualified measurement validator.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value), "ieee754_binary64_hex": struct.pack(">d", value).hex()}
    if isinstance(value, dict):
        return {key: preserve_invalid_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [preserve_invalid_numbers(item) for item in value]
    return value
