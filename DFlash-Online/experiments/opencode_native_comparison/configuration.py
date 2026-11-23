"""The common CODE recipe, with one declared context-length change."""
from copy import deepcopy

import yaml

from experiments.ok_coding.training import validate_config_text as validate_original
from experiments.ok_continuation.resume import differences

from .contract import ARMS


def historical_view(config_text):
    """A validation view only; the running and saved CODE config stays 4,096."""
    raw = yaml.safe_load(config_text)
    if raw.get('data', {}).get('max_length') not in (2272, 4096):
        raise ValueError('Only the declared 2272-to-4096 context change is allowed')
    raw['data']['max_length'] = 2272
    return yaml.safe_dump(raw)


def validate_config_text(config_text, original_config_text):
    current = yaml.safe_load(config_text)
    original = yaml.safe_load(original_config_text)
    if current.get('data', {}).get('max_length') != 4096:
        raise ValueError('All CODE arms require the shared 4096 training context')
    validate_original(historical_view(original_config_text), 17)
    validate_original(historical_view(config_text), 17)
    checked = deepcopy(current)
    checked['data']['max_length'] = original['data']['max_length']
    for name in ('run_id', 'output_dir'):
        if name in original:
            checked[name] = original[name]
        else:
            checked.pop(name, None)
    mismatch = differences(checked, original)
    if mismatch:
        raise ValueError('CODE config changed beyond context/run/output: ' + repr(mismatch[:8]))
    return dict(status='passed', context_limit=4096, seed=17,
                allowed_changes=['data.max_length', 'run_id', 'output_dir'])


def build_config(original_config_text, *, arm, output_dir, run_id=None):
    if arm not in ARMS:
        raise ValueError('Unknown CODE training arm')
    raw = yaml.safe_load(original_config_text)
    raw['data']['max_length'] = 4096
    raw['run_id'] = run_id or 'opencode-' + arm.lower()
    raw['output_dir'] = str(output_dir)
    result = yaml.safe_dump(raw, sort_keys=False)
    validate_config_text(result, original_config_text)
    return result
