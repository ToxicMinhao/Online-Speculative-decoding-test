"""The prospectively chosen one-event screen; historical CODE contracts stay fixed."""
from experiments.opencode_native_comparison.contract import SERVING, digest

ARMS = ('BASE', 'PAIR')
BASE_COMMIT = '1551e5e5d0dd7736d8441977aa3dea250535da7b'
LIMITS = dict(elapsed_s=10800, occupied_card_s=18000,
              new_allocated_bytes=90_000_000_000, free_reserve_bytes=400_000_000_000)
WINDOW, PANEL, REPEATS = 128, 128, 3


def validate_inputs(inputs):
    groups = [('train_requests', WINDOW), ('evaluation_requests', PANEL),
              ('warmup_requests', 2), ('qualification_requests', 2)]
    identifiers, prompts = set(), set()
    for name, count in groups:
        rows = inputs[name]
        if len(rows) != count:
            raise ValueError(f'{name} requires {count} sealed inputs')
        for ordinal, row in enumerate(rows):
            key, tokens = row['request_id'], tuple(row['prompt_ids'])
            if key in identifiers or tokens in prompts:
                raise ValueError('Repeated prompt or identity across pilot roles')
            if (row['request_index'] != ordinal or row['max_new_tokens'] != 2048
                    or row['stop_token_ids'] != [151645, 151643]
                    or not 1 <= len(tokens) <= 2048):
                raise ValueError('Native order/token/termination contract changed')
            identifiers.add(key)
            prompts.add(tokens)
    return dict(status='passed', distinct_inputs=len(identifiers))
