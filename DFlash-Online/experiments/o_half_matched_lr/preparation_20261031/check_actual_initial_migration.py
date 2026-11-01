"""Read-only CPU admission of both retained O initial states after JSON transport.

Run with CUDA_VISIBLE_DEVICES='' and DFlash-Online:training on PYTHONPATH.
Only the compact output receipt is written; no model is constructed or executed.
"""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import torch
import yaml

from experiments.o_half_matched_lr.contract import PANEL_BINDING, ORIGINAL_COMMIT
from experiments.o_half_matched_lr.training import experiment_contract, migrate_source, validate_snapshot
from experiments.ok_coding import training as original
from online_feedback.rng import capture_rng

parser = argparse.ArgumentParser()
parser.add_argument('--audit', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--commit', required=True)
args = parser.parse_args()
if torch.cuda.is_initialized():
    raise RuntimeError('CPU check entered with CUDA initialized')
audit = json.loads(args.audit.read_text())
if audit['status'] != 'passed' or audit['model_work'] != 0:
    raise ValueError('Require the complete authoritative CPU reuse audit')
binding = dict(upstream_data_binding=audit['original_data_binding'],
               evaluation_binding=PANEL_BINDING, execution_commit=args.commit,
               original_execution_commit=ORIGINAL_COMMIT)
result = dict(status='running', execution_commit=args.commit, model_work=0,
              optimizer_steps=0, scientific_state_writes=0, seeds={})
started = time.perf_counter()
try:
    for seed in (17, 29):
        key = str(seed)
        source_path = Path(audit['original_initial_states'][key])
        config_text = Path(audit['original_configs'][key]).read_text()
        before = capture_rng()
        source, source_check = original.load_snapshot(source_path, audit['draft'],
            arm='O', seed=seed, campaign_contract=audit['original_campaign_contracts'][key],
            config_text=config_text, completed_event=0)
        raw = yaml.safe_load(config_text)
        raw['training']['learning_rate'] = 1.5e-4
        own_config = yaml.safe_dump(raw)
        native_contract = deepcopy(source['algorithm_contract'])
        native_contract['o_half_matched_lr_v1'] = experiment_contract(seed, binding)
        kwargs = dict(seed=seed, binding=binding,
            original_contract=audit['original_campaign_contracts'][key],
            original_algorithm_contract=audit['original_algorithm_contracts'][key],
            original_optimizer_recipe=audit['original_optimizer_recipes'][key],
            original_config_text=config_text, config_text=own_config)
        candidate, migration = migrate_source(source, **kwargs, new_contract=native_contract)
        checked = validate_snapshot(candidate, **kwargs, completed_event=0)
        original._export_equal(candidate, audit['draft'])
        if original.inherited.differences(before, capture_rng()) or torch.cuda.is_initialized():
            raise ValueError('Read-only CPU check changed process RNG or initialized CUDA')
        if source['backend']['optimizer']['recipe']['optimizer_options']['lr'] != 3e-4:
            raise ValueError('Original source was mutated')
        result['seeds'][key] = dict(source=source_check, migration=migration, own=checked,
            source_path=str(source_path), export_exact=True,
            native_contract_types={name: type(source['algorithm_contract'][name]).__name__
                for name in ('dflash_target_layer_ids','specforge_step_options','specforge_model_provenance')},
            json_transport_types={name: type(audit['original_algorithm_contracts'][key][name]).__name__
                for name in ('dflash_target_layer_ids','specforge_step_options','specforge_model_provenance')},
            process_rng_unchanged=True, cuda_initialized=False)
        del candidate, source
        gc.collect()
    result['status'] = 'passed'
except BaseException as error:
    result.update(status='failed', error=repr(error))
    raise
finally:
    result['wall_s'] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
