"""Native PR56 numerical recipe with honest BASE/TAIL output identities."""
import yaml
from experiments.opencode_native_comparison.configuration import validate_config_text as validate_native
from .contract import ARMS, experiment_contract


def validate_config_text(config_text, original_config_text):
    return validate_native(config_text, original_config_text)


def build_config(original_config_text, *, arm, output_dir):
    if arm not in ARMS:
        raise ValueError('Unknown full-run training arm')
    value = yaml.safe_load(original_config_text)
    value['data']['max_length'] = 4096
    value['run_id'] = 'continuation-repair-fullrun-' + arm.lower()
    value['output_dir'] = str(output_dir)
    result = yaml.safe_dump(value, sort_keys=False)
    validate_config_text(result, original_config_text)
    return result
