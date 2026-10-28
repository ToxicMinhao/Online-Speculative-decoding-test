"""Extract inspectable token/mask evidence from saved feedback, without inference."""
import argparse
import json
from pathlib import Path
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for path in sorted(args.run.glob('*/F/features/*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        row = {key: payload[key] for key in ('record_id', 'prompt_length', 'label_binding')}
        row['group'] = path.parents[2].name
        row['source_path'] = str(path)
        row['input_ids'] = payload['input_ids'].tolist()
        row['loss_mask'] = payload['loss_mask'].tolist()
        row['tensors'] = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                               'finite': bool(torch.isfinite(value).all())}
                          for key, value in payload.items() if isinstance(value, torch.Tensor)}
        rows.append(row)
        del payload
    args.output.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps({'feedback_files': len(rows)}))
