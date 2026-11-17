"""Complete native-origin continuation with one explicit CPU/CUDA RNG fork."""
from copy import deepcopy
import math

from experiments.full_k_gsm8k import training as native
from experiments.mathqa_fixed_lr_transfer import state as mathqa
from experiments.recent_repair_retention import training as gsm8k
from experiments.role_loss_pilot import training as historical
from experiments.functional_step_geometry import training as geometry
from experiments.ok_continuation.resume import differences
from experiments.persistent_prefix_withholding.qualification import tree_identity
from online_feedback.kplus import kplus_contract
from online_feedback.prefix_allocation import prefix_allocation_contract
from experiments.unchanged_c_mathqa_transfer.state import validate_allocation_observation

CONTRACT_KEY = 'c_paired_confirmation_v1'
FORK_SEED = 29
ORIGIN_EVENT = {'gsm8k': 6, 'mathqa': 4}
NATIVE_LR = 3e-4
RELATIVE_CAP = .25


def fork_rng_state(saved, *, device, logical_cuda_index=0, generator_factory=None):
    """Private generators patch two fields; they do not seed the live runtime."""
    import torch
    factory = torch.Generator if generator_factory is None else generator_factory
    if logical_cuda_index not in range(len(saved['cuda'])):
        raise ValueError('Assigned logical CUDA RNG is absent from the complete origin')
    patched = deepcopy(saved)
    patched['torch'] = factory(device='cpu').manual_seed(FORK_SEED).get_state().clone()
    patched['cuda'][logical_cuda_index] = factory(device=device).manual_seed(FORK_SEED).get_state().clone()
    return patched


def fork_record(origin, patched_rng, *, dataset, origin_identity, physical_gpu_uuid,
                logical_cuda_index=0):
    if dataset not in ORIGIN_EVENT or not physical_gpu_uuid:
        raise ValueError('Require the sealed dataset and assigned physical training card')
    if differences(origin['rng']['python'], patched_rng['python']) or differences(origin['rng']['numpy'], patched_rng['numpy']):
        raise ValueError('Continuation fork cannot alter Python or NumPy RNG')
    for index, value in enumerate(origin['rng']['cuda']):
        if index != logical_cuda_index and differences(value, patched_rng['cuda'][index]):
            raise ValueError('Unassigned CUDA RNG changed')
    return dict(seed=FORK_SEED, origin_identity=deepcopy(origin_identity),
        first_event=ORIGIN_EVENT[dataset]+1, historical_initialization_seed=17,
        replaced_fields=['torch', f'cuda[{logical_cuda_index}]'],
        physical_gpu_uuid=physical_gpu_uuid, logical_cuda_index=logical_cuda_index,
        original_rng=tree_identity(origin['rng']), forked_rng=tree_identity(patched_rng),
        application='once_after_exact_startup_restore_before_first_natural_training_draw',
        second_event_reseed=False)


def experiment_contract(dataset, arm, binding, lineage):
    native.experiment_contract('K', binding)
    if dataset not in ORIGIN_EVENT or arm not in ('N', 'C'):
        raise ValueError('Paired continuation requires GSM8K/MathQA and N/C')
    if lineage['seed'] != 29 or lineage['first_event'] != ORIGIN_EVENT[dataset]+1:
        raise ValueError('Wrong continuation fork or event')
    return dict(schema=CONTRACT_KEY, dataset=dataset, arm=arm, binding=binding,
        first_event=ORIGIN_EVENT[dataset]+1, final_event=ORIGIN_EVENT[dataset]+2,
        learning_rate=NATIVE_LR, inherited_horizon=7424, native_loss=kplus_contract(),
        allocation=prefix_allocation_contract(RELATIVE_CAP) if arm == 'C' else None,
        continuation_rng_fork=deepcopy(lineage), initialization_seed=17,
        seed_reset=True, optimizer_reset=False, reference=None)


def continuation_contract(origin, *, dataset, arm, binding, lineage):
    return dict(deepcopy(origin['algorithm_contract']),
                **{CONTRACT_KEY:experiment_contract(dataset, arm, binding, lineage)})


def native_base_contract(contract):
    result = dict(contract)
    for name in (historical.CONTRACT_KEY, geometry.CONTRACT_KEY, CONTRACT_KEY):
        result.pop(name, None)
    return result


def validate_origin(state, original_config_text, *, dataset, expected_cuda_rng_states=1):
    if CONTRACT_KEY in state['algorithm_contract']:
        raise ValueError('Use the original complete origin, not an earlier paired fork')
    if dataset == 'gsm8k':
        return gsm8k.validate_origin(state, original_config_text,
                                    expected_cuda_rng_states=expected_cuda_rng_states)
    if dataset == 'mathqa':
        return mathqa.validate_origin(state, original_config_text,
                                     expected_cuda_rng_states=expected_cuda_rng_states)
    raise ValueError('Unknown paired-continuation dataset')


def fork_state(origin, *, dataset, arm, binding, lineage, config_text, run_id,
               original_config_text, expected_cuda_rng_states=1):
    """Keep restored RNG original here. Apply the declared fork after startup."""
    validate_origin(origin, original_config_text, dataset=dataset,
                    expected_cuda_rng_states=expected_cuda_rng_states)
    native.validate_config_text(config_text, 'K', original_config_text)
    return dict(origin, config_text=config_text, run_id=run_id,
        algorithm_contract=continuation_contract(origin, dataset=dataset, arm=arm,
                                                 binding=binding, lineage=lineage))


def validate_event_receipt(event, *, dataset, arm, binding, lineage, event_number,
                           require_completed=True, trainer_status=None):
    origin_event = ORIGIN_EVENT[dataset]
    expected = experiment_contract(dataset, arm, binding, lineage)
    if (event_number not in (origin_event+1, origin_event+2)
            or event.get('method') != arm or event.get('dataset') != dataset
            or event.get('kplus_loss') != kplus_contract()
            or event.get(CONTRACT_KEY) != expected):
        raise ValueError('Event lost its paired method or fork lineage')
    result = native.validate_event_receipt(event, arm='K', event_number=event_number,
        require_completed=require_completed, trainer_status=trainer_status)
    for row in event['records']:
        observation = row['response_observation']
        if (observation['record_id'] != row['record_id'] or observation['pass_index'] != row['pass']
                or observation['sampled_anchors'] != row['sampled_anchors']
                or observation['sum_metrics']['kplus_response_count'] != 1):
            raise ValueError('Response observation does not bind the native presentation')
        values = observation['sum_metrics']
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError('Nonfinite native response observation')
        weight = values['kplus_W1'] + values['kplus_W2']
        tolerance = kplus_contract()['mass_conservation_atol'] + kplus_contract()['mass_conservation_rtol']*weight
        if abs(values['kplus_first_error_weight_after']-weight) > tolerance:
            raise ValueError('Native first-error mass changed')
        if arm == 'C':
            validate_allocation_observation(observation)
        elif observation.get('prefix_allocation') != {'enabled':False}:
            raise ValueError('Native arm unexpectedly enabled prefix allocation')
    return dict(result, dataset=dataset, arm=arm, continuation_rng_fork_preserved=True)


def validate_snapshot(state, *, origin, dataset, arm, binding, lineage, config_text,
                      original_config_text, completed_event, expected_recipe,
                      expected_cuda_rng_states=1):
    first = ORIGIN_EVENT[dataset]
    if completed_event not in (first, first+1, first+2):
        raise ValueError('Only the origin and two complete paired events are allowed')
    expected = continuation_contract(origin, dataset=dataset, arm=arm, binding=binding, lineage=lineage)
    if differences(state['algorithm_contract'], expected) or differences(state['events'][:first], origin['events']):
        raise ValueError('Foreign fork lineage or rewritten original history')
    for number, event in enumerate(state['events'][first:], first+1):
        validate_event_receipt(event, dataset=dataset, arm=arm, binding=binding,
            lineage=lineage, event_number=number, require_completed=False)
    if dataset == 'gsm8k':
        view = gsm8k.native_view(origin)
    else:
        old = origin['algorithm_contract'][historical.CONTRACT_KEY]
        view = historical.native_contract_view(origin['algorithm_contract'], old['binding'], 'KPLUS', dataset)
    result = native.validate_snapshot(dict(state, algorithm_contract=view), arm='K',
        binding=view[native.CONTRACT_KEY]['binding'], config_text=config_text,
        original_config_text=original_config_text, completed_event=completed_event,
        expected_contract=view, expected_recipe=expected_recipe,
        expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(result, arm=arm, dataset=dataset, fork_seed=29, historical_initialization_seed=17)
