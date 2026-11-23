"""One continuous optimizer, two shuffled passes, separately normalized responses."""

import json
import random
import time

import numpy as np
import torch

from graduate.objective import response_loss


def seed_training(seed):
    """Seed once before model construction, never between responses or passes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_event(model, optimizer, feedback, config, log_path):
    """Train the supplied model in place; publication is a separate later step.

    Prepared feedback stays on CPU. The model moves one response at a time to
    its device and draws fresh anchors from that response's complete context.
    """
    settings = config.training
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    presentations = len(feedback) * settings.passes_per_event
    # A partial accumulation window silently changes the effective batch size.
    if presentations % settings.accumulation_steps:
        raise ValueError("Training must end on a complete accumulation window.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    presentation = 0
    optimizer_steps = 0
    started = time.perf_counter()
    with log_path.open("w") as log:
        for pass_index in range(settings.passes_per_event):
            order = torch.randperm(len(feedback)).tolist()
            for response_index in order:
                response = feedback[response_index]
                blocks = model(response) # Samples anchor positions; Constructs each 16-token input block; Creates position IDs and attention masks.; Runs the current draft model
                loss, metrics = response_loss(
                    blocks.hidden, blocks.teacher_hidden, blocks.labels, blocks.weights,
                    model.lm_head, response.stop_token_ids, config.objective,
                    settings.projection_chunk_blocks,
                )
                # Mean of response losses, not one mean over pooled token weights.
                (loss / settings.accumulation_steps).backward()
                presentation += 1
                row = {
                    "pass": pass_index + 1, "presentation": presentation,
                    "record_id": response.record_id,
                    "anchors": blocks.anchors.detach().cpu().tolist()[0],
                    "loss": loss.detach().item(),
                    **{name: value.item() for name, value in metrics.items()},
                }
                if presentation % settings.accumulation_steps == 0:
                    # A NaN gradient need not raise during backward or Adam.
                    # Check here, once, as part of the already required clipping.
                    norm = torch.nn.utils.clip_grad_norm_(
                        parameters, settings.max_grad_norm, error_if_nonfinite=True,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    row.update(optimizer_step=optimizer_steps, gradient_norm=norm.item())
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(f"pass {pass_index + 1}, response {presentation}/{presentations}, "
                      f"loss {row['loss']:.6f}, optimizer steps {optimizer_steps}", flush=True)
                # Do not keep the previous response's graph alive into the next forward.
                del loss, metrics, blocks

    return {
        "responses": len(feedback), "presentations": presentation,
        "optimizer_steps": optimizer_steps,
        "training_seconds": time.perf_counter() - started,
    }
