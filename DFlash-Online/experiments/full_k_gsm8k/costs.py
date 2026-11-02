"""Passive training-resource observations; no model or optimizer operations."""
import math


def compiler_snapshot():
    from torch._dynamo.utils import compilation_time_metrics

    return {name: list(durations) for name, durations in compilation_time_metrics.items()}


def compiler_delta(before, after):
    """Compiler timers may nest: retain each timer, never add their totals."""
    result = {}
    for name in sorted(set(before) | set(after)):
        previous, current = before.get(name, []), after.get(name, [])
        if current[:len(previous)] != previous or len(current) < len(previous):
            raise ValueError("Compiler timing history changed during one event")
        durations = current[len(previous):]
        if any(not math.isfinite(value) or value < 0 for value in durations):
            raise ValueError("Invalid compiler duration")
        if durations:
            result[name] = dict(calls=len(durations), durations_s=durations, total_s=sum(durations))
    return result


def begin_event_profile():
    import torch

    if not torch.cuda.is_initialized():
        raise ValueError("Profile only the already initialized training device")
    torch.cuda.reset_peak_memory_stats()
    return compiler_snapshot()


def finish_event_profile(before):
    import torch

    return dict(
        training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        training_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        training_compiler_timers=compiler_delta(before, compiler_snapshot()),
        compiler_timer_scope="Native TorchDynamo timers; nested, included in training time",
        native_worker_compilation="Included in worker loading/serving; not separately identified",
        extra_forwards=0, extra_backwards=0, extra_optimizer_steps=0,
    )
