"""Keep model loading, replay and export from changing future training draws."""

from contextlib import contextmanager
import random


def capture_rng():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)
