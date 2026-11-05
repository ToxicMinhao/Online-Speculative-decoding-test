"""CPU token-probability walkthrough; no DFlash training or campaign execution.

The constructor and chunk loss are imported from the installed component's
review copy. An identity vocabulary head exposes four freely adjustable logits,
so this checks the objective directly, not backbone/AdamW learning dynamics.
"""
from pathlib import Path
import csv
import json
import sys
import tarfile

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
COMPONENT = ROOT.parent / 'projected_target_component_20261022'
sys.path.insert(0, str(COMPONENT / 'worktree'))
from online_feedback.objective import TargetObjectiveModel
from online_feedback.projected_target import projected_targets

TOKENS = 'ABCD'
STEPS = 100
LEARNING_RATE = 1.0  # Illustration only: direct logits, not the training recipe.
CASES = [
    ('clear_error', [.40, .35, .05, .20], 2),
    ('near_miss', [.36, .34, .20, .10], 1),
    ('already_correct', [.15, .60, .15, .10], 1),
    ('tie_selected', [.40, .40, .10, .10], 0),
    ('tie_not_selected', [.40, .40, .10, .10], 1),
]


class LogitProbe(nn.Module):
    """Use the production loss with an identity head and one supervised token."""
    _chunk_loss = TargetObjectiveModel._chunk_loss

    def __init__(self):
        super().__init__()
        self.lm_head = nn.Identity()
        self.objective = 'projected'
        self.costs = {'reference_head_s': 0.0, 'projection_s': 0.0}

    def forward(self, current_logits, reference_logits, label):
        numerator, denominator, _, _ = self._chunk_loss(
            current_logits, reference_logits, label, torch.ones_like(label).float(),
        )
        return numerator / denominator


def run_case(name, probabilities, label, initial_probabilities=None):
    reference = torch.tensor([probabilities], dtype=torch.float32).log()
    labels = torch.tensor([label])
    reference_probabilities = reference.softmax(-1)
    target = projected_targets(reference, labels)
    reference_was_correct = int(reference.argmax(-1)) == label
    assert not target.requires_grad
    torch.testing.assert_close(target.sum(-1), torch.ones(1), atol=2e-6, rtol=0)
    if reference_was_correct:
        assert torch.equal(target, reference_probabilities)
    else:
        rivals = target[0, torch.arange(4) != label]
        assert float(target[0, label] - 1.1 * rivals.max()) >= -2e-6
        assert int(target.argmax(-1)) == label

    current = torch.tensor([initial_probabilities or probabilities]).log().requires_grad_()
    probe = LogitProbe()
    rows = []
    for step in range(STEPS + 1):
        # Exactly the same reference is used after every synthetic logit step.
        assert torch.equal(projected_targets(reference, labels), target)
        loss = probe(current, reference, labels)
        (gradient,) = torch.autograd.grad(loss, current)
        q = current.detach().softmax(-1)
        error = (gradient - (q - target)).abs().max()
        torch.testing.assert_close(gradient, q - target, atol=2e-7, rtol=2e-5)
        row = {
            'case': name, 'step': step, 'target_token': TOKENS[label],
            'argmax': TOKENS[int(current.argmax(-1))], 'soft_ce': float(loss.detach()),
            'max_q_minus_r_error': float(error),
            'max_distance_to_target': float((q - target).abs().max()),
            **{f'q_{t}': float(q[0, i]) for i, t in enumerate(TOKENS)},
            **{f'r_{t}': float(target[0, i]) for i, t in enumerate(TOKENS)},
            **{f'gradient_{t}': float(gradient[0, i]) for i, t in enumerate(TOKENS)},
        }
        rows.append(row)
        if step < STEPS:
            with torch.no_grad():
                current -= LEARNING_RATE * gradient
    assert rows[-1]['max_distance_to_target'] < 2e-6
    assert all(b['soft_ce'] <= a['soft_ce'] + 1e-6 for a, b in zip(rows, rows[1:]))
    return {
        'name': name, 'reference_q0': reference_probabilities[0].tolist(),
        'label': TOKENS[label], 'initial_q': initial_probabilities or probabilities,
        'target_r': target[0].tolist(), 'reference_was_correct': reference_was_correct,
        'target_unchanged_at_all_steps': True, 'rows': rows,
    }


def main():
    torch.set_num_threads(1)
    # Byte comparison with the exact seven-file bundle installed on A800.
    with tarfile.open(COMPONENT / 'source_update.tar.gz') as archive:
        for relative in ('online_feedback/objective.py', 'online_feedback/projected_target.py'):
            assert archive.extractfile(relative).read() == (COMPONENT / 'worktree' / relative).read_bytes()
    cases = [run_case(*case) for case in CASES]
    cases.append(run_case('correct_but_disturbed', [.15, .60, .15, .10], 1,
                          initial_probabilities=[.35, .35, .20, .10]))
    torch.testing.assert_close(
        torch.tensor(cases[0]['target_r']),
        torch.tensor([.258064516, .258064516, .283870968, .20]),
        atol=2e-7, rtol=2e-5,
    )
    rows = [row for case in cases for row in case['rows']]
    with (ROOT / 'trajectory.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        'scope': 'synthetic CPU free-logit diagnostic; not DFlash model training',
        'production_code_matches_installed_bundle': True,
        'loss': 'actual TargetObjectiveModel._chunk_loss, identity head, one unit-weight position',
        'dtype': 'float32', 'margin_ratio': 1.1,
        'optimizer': 'direct-logit gradient descent', 'learning_rate': LEARNING_RATE,
        'steps_per_case': STEPS, 'tolerances': {'probability_atol': 2e-6, 'gradient_atol': 2e-7, 'gradient_rtol': 2e-5},
        'passed': True, 'cases': cases,
        'max_gradient_identity_error': max(row['max_q_minus_r_error'] for row in rows),
    }
    (ROOT / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    for case in cases:
        print(case['name'], 'q0=', case['reference_q0'], 'r=', case['target_r'])
        for row in case['rows']:
            if row['step'] in (0, 1, 2, 5, 10, 20, 50, 100):
                print(row['step'], [round(row[f'q_{t}'], 6) for t in TOKENS], row['argmax'], round(row['soft_ce'], 6))
    print('All checks passed; maximum q-r gradient error:', payload['max_gradient_identity_error'])


if __name__ == '__main__':
    main()
