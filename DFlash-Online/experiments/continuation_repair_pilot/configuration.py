"""A fixed warm-event identity; native training settings remain unchanged."""
from copy import deepcopy
import hashlib
import json
import yaml

SCHEMA = 'continuation_repair_pilot_v1'
ARMS = ('BASE', 'TAIL_DISABLED', 'TAIL')
PARENT_BINDING = 'b120c76b0036b4786eb6b9d11b1594ec1dc6e57e3d20c9ed931924262b1c072a'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def experiment_contract(arm, binding):
    if arm not in ARMS or len(binding) != 64 or any(c not in '0123456789abcdef' for c in binding):
        raise ValueError('Require an explicit continuation arm and sealed binding')
    return dict(name=SCHEMA, arm=arm, binding=binding, parent_binding=PARENT_BINDING,
                parent_counters=[128, 256, 1], final_counters=[256, 512, 2],
                event_id='event_002', additional_presentations=256, additional_adam_steps=128,
                inherited_horizon=7424, objective='native_C_Kplus_PAIR',
                continuation_repair=arm == 'TAIL', score='native_FP32_running_minimum_support_area_over_14',
                coefficient_rule='within_HIGH_beta_centered_kappa_min_c_eta_minus_c',
                coefficient_precision='FP32', eta=.5, reset_optimizer=False, reset_seed=False,
                feedback='authenticated_retained_PR60_event2', schedule='natural_native_draws_checked')


def validate_config(config_text, parent_config_text):
    original, current = yaml.safe_load(parent_config_text), yaml.safe_load(config_text)
    for value in (original, current):
        value.pop('run_id', None)
        value.pop('output_dir', None)
    if current != original:
        raise ValueError('Warm continuation changes settings beyond run/output metadata')
    return dict(status='passed', changed_settings=[], inherited_horizon=7424)


def build_config(parent_text, arm, output):
    if arm not in ARMS:
        raise ValueError('Unknown continuation arm')
    value = deepcopy(yaml.safe_load(parent_text))
    value.update(run_id='continuation-repair-' + arm.lower(), output_dir=str(output))
    text = yaml.safe_dump(value, sort_keys=False)
    validate_config(text, parent_text)
    return text
