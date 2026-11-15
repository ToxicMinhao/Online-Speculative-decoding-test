"""CPU qualification and exact historical endpoint authentication.

Historical E measurements are reused, not regenerated. Missing P/N weights
become authenticated only when their newly serialized files match the original
SHA256 and byte count. A failed copy remains available for diagnosis.
"""
from collections import OrderedDict
import json
import math
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from experiments.threatened_repair_replay.prepare import identity, read
from experiments.threatened_repair_replay.runner import export_weights
from experiments.threatened_repair_replay.state import tensor_change
from .storage import check_write


NORM_ATOL, NORM_RTOL = 1e-12, 1e-10


def norm_agrees(measured, retained):
    return math.isfinite(measured) and math.isfinite(retained) and abs(measured-retained) <= NORM_ATOL + NORM_RTOL*abs(retained)


def check_identity(path, expected):
    actual = identity(path)
    if actual != {name: expected[name] for name in ('bytes', 'sha256')}:
        raise ValueError('Historical artifact identity differs: ' + str(path))
    return actual


def load_export(row, ordering):
    from safetensors.torch import load_file
    values = load_file(str(Path(row['checkpoint']) / 'model.safetensors'), device='cpu')
    if set(values) != set(ordering):
        raise ValueError('Export membership differs from original native state ordering')
    return OrderedDict((name, values[name]) for name in ordering)


def validate_copy(plan, row, weights, directory, receipt_path):
    """Original exporter, one rolling file, release only after a durable match."""
    companions = sum(item.stat().st_size for item in Path(plan['origin_export']).iterdir()
                     if item.is_file() and item.suffix not in ('.safetensors','.bin','.pt')
                     and not item.name.endswith('.index.json'))
    check_write(plan, row['weights_identity']['bytes'] + companions + 65_536, 'identity_export')
    directory = Path(directory)
    began = time.perf_counter()
    exported = export_weights(weights, plan['origin_export'], directory)
    actual, expected = exported['weights_identity'], row['weights_identity']
    passed = all(actual[name] == expected[name] for name in ('bytes','sha256'))
    receipt = dict(status='passed' if passed else 'failed', window=row['window'], state=row['state'],
                   original=row['checkpoint'], validation_copy=str(directory), original_identity=expected,
                   reproduced_identity=actual, tensors=exported['tensors'], cpu_io_s=time.perf_counter()-began,
                   release_condition='exact original file SHA256 and bytes; durable receipt',
                   copy_class='TEMP_VALIDATION' if passed else 'REVIEW')
    write_json(receipt_path, receipt)
    if not passed:
        raise ValueError('Unqualified file-identity procedure; preserve mismatch file: ' + str(directory))
    shutil.rmtree(directory)
    receipt['new_copy_removed_after_pass'] = True
    write_json(receipt_path, receipt)
    return receipt


def authenticate_endpoint(plan, window, state, weights, output):
    """Called during the single native trajectory for each P/N endpoint."""
    row = dict(plan['historical_windows'][str(window)][state], window=window, state=state)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    if row['available']:
        expected = load_export(row, weights)
        import torch
        mismatch = [name for name in weights if weights[name].dtype != expected[name].dtype
                    or weights[name].shape != expected[name].shape
                    or not torch.equal(weights[name].detach().cpu().view(torch.uint8), expected[name].view(torch.uint8))]
        result = dict(status='failed' if mismatch else 'passed', mode='direct original tensor comparison',
                      window=window, state=state, tensors=len(weights), mismatches=mismatch,
                      original_identity=check_identity(Path(row['checkpoint'])/'model.safetensors', row['weights_identity']))
        write_json(output/f'{state}_IDENTITY.json', result)
        if mismatch:
            raise ValueError('Reconstructed endpoint tensors differ: ' + str(mismatch[:4]))
    else:
        result = validate_copy(plan, row, weights, Path(plan['root'])/'rolling_validation', output/f'{state}_IDENTITY.json')
        result['mode'] = 'exact reconstructed original-file identity; no pre-cleanup tensor fingerprints'
    result['cpu_io_s'] = time.perf_counter()-began
    write_json(output/f'{state}_IDENTITY.json', result)
    return result


def compare_boundary_witness(snapshot, window, plan, output):
    import torch
    from experiments.ok_continuation.resume import differences
    path = plan['historical_windows'][str(window)].get('witness')
    if path is None:
        return dict(status='unavailable', window=window, reason='historical full witness exists only at8/128')
    old = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    # Timers, output locations and observer diagnostics are not native scientific
    # state. Compare all tensor/Adam/RNG/counter, accumulated-gradient and sampler
    # contents, plus modes/buffers; preserve a precise field list in the receipt.
    fields = ['draft_state_dict','backend','rng','global_step','presentations','strategy',
              'world_size','reduction','accumulation_steps','horizon']
    mismatch = differences({k:snapshot['scientific'][k] for k in fields},
                          {k:old['scientific'][k] for k in fields}, 'scientific')
    for key in ('core','gradients','schedule_state'):
        mismatch += differences(snapshot[key], old[key], key)
    for key in ('modes','buffers','nonpersistent'):
        mismatch += differences(snapshot['runtime'][key], old['runtime'][key], 'runtime.'+key)
    result = dict(status='failed' if mismatch else 'passed', window=window, historical_witness=path,
                  scientific_fields=fields, exact_fields=['core','gradients','schedule_state','runtime modes/buffers/nonpersistent'],
                  excluded_non_scientific_fields=['wall timers','output paths','role-observer diagnostics','opaque cache bindings'],
                  mismatches=mismatch[:64])
    Path(output).mkdir(parents=True,exist_ok=True); write_json(Path(output)/'WITNESS_PARITY.json',result)
    if mismatch:
        raise ValueError('Native boundary witness differs: '+str(mismatch[:6]))
    return result


def gate0(plan, output):
    """No model construction, CUDA operation, optimizer, or missing-E recreation."""
    import torch
    import safetensors
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    result = dict(status='running', cpu_only=True, gpu_work=0, optimizer_steps=0,
                  torch=torch.__version__, safetensors=safetensors.__version__,
                  norm_atol=NORM_ATOL, norm_rtol=NORM_RTOL, inputs=[], checks=[])
    write_json(output/'GATE0.json', result)
    try:
        if torch.cuda.is_initialized():
            raise ValueError('CPU Gate0 must start before CUDA initialization')
        for path, expected in plan['live_input_identities'].items():
            check_identity(path,expected)
            result['inputs'].append(dict(path=path,identity=expected,status='passed'))
        write_json(output/'LIVE_INPUTS.json',result['inputs'])
        for window in (8,128):
            boundary=plan['historical_windows'][str(window)]
            witness=torch.load(boundary['witness'],map_location='cpu',weights_only=False,mmap=True)
            original=witness['scientific']['draft_state_dict']
            for state in ('E','P','N'):
                row=dict(boundary[state],window=window,state=state)
                if not row['available']:
                    continue
                weights=load_export(row,original)
                check_identity(Path(row['checkpoint'])/'model.safetensors',row['weights_identity'])
                if state=='E':
                    change=tensor_change(original,weights)
                    passed=norm_agrees(change['difference_norm'],boundary['E_norm'])
                    receipt=dict(window=window,state=state,status='passed' if passed else 'failed',
                                 measured=change,retained_norm=boundary['E_norm'])
                    write_json(output/f'E_{window:03d}_NORM.json',receipt)
                    result['checks'].append(receipt)
                    if not passed:
                        raise ValueError('Retained E displacement does not reconcile')
                copied=validate_copy(plan,row,weights,Path(plan['root'])/'rolling_validation',output/f'{state}_{window:03d}_SERIALIZER.json')
                result['checks'].append(copied)
                del weights
                write_json(output/'GATE0.json',result)
            del original,witness
        result.update(status='passed', original_ordering='retained native boundary witness state_dict',
                      missing_PN_procedure_only=True, missing_endpoints_not_yet_authenticated=28)
    except BaseException as error:
        result.update(status='failed',error=repr(error),failure_type='unqualified CPU input/measurement/file-identity procedure')
        raise
    finally:
        result['cpu_wall_s']=time.perf_counter()-began
        result['cuda_initialized']=torch.cuda.is_initialized()
        write_json(output/'GATE0.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();gate0(read(args.plan),args.output)
