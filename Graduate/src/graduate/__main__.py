"""The entry point shared by terminal execution and F5."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from graduate.config import load_config
from graduate.feedback import read_response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--debug-responses", type=int,
                        help="Use a small real subset, without claiming a full event.")
    arguments = parser.parse_args(argv)

    # First breakpoint: model preparation and training are downstream of this line.
    config = load_config(arguments.config)
    if arguments.command == "collect":
        from graduate.live import collect_feedback
        result = collect_feedback(config, arguments.debug_responses)
        print(json.dumps(result, indent=2), flush=True)
        return 0

    from graduate.batches import load_feedback, read_feature_layout
    from graduate.buffer import FeedbackBuffer

    layout = read_feature_layout(config)
    manifest = config.feedback.manifest
    entries = json.loads(manifest.read_text())
    count = (config.training.responses_per_event if arguments.debug_responses is None
             else arguments.debug_responses)
    if count <= 0 or len(entries) < count:
        parser.error(f"Requested {count} distinct responses; manifest contains {len(entries)}.")
    entries = entries[:count]
    records = [read_response(manifest.parent / entry["response"]) for entry in entries]
    # Buffer imports are idempotent, but duplicate entries must not train twice.
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("Training manifest contains duplicate response identities.")
    prepared = [load_feedback(record, manifest.parent / entry["features"], layout)
                for record, entry in zip(records, entries)]
    buffer = FeedbackBuffer(config.feedback.buffer_dir, config.training.responses_per_event)
    for record in records:
        buffer.add(record)

    import torch
    from graduate.model import load_model
    from graduate.storage import write_json
    from graduate.training import seed_training, train_event

    run_dir = config.training.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "inputs.json", {
        "configuration": config.model_dump(mode="json"),
        "debug_responses": arguments.debug_responses,
        "record_ids": [record.record_id for record in records],
        "manifest": str(manifest),
    })
    seed_training(config.training.seed)
    torch.cuda.reset_peak_memory_stats()
    model = load_model(config, "cuda")
    # Native BF16 parameters imply BF16 Adam moments; there are no master weights.
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate, weight_decay=config.training.weight_decay,
    )
    result = train_event(model, optimizer, prepared, config, run_dir / "presentations.jsonl")
    result.update(
        status="debug_training_completed" if arguments.debug_responses else "training_completed",
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), output_directory=str(run_dir),
        publication="not_implemented_in_step_4",
    )
    write_json(run_dir / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
