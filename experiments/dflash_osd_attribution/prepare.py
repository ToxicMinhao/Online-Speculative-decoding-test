"""Read-only inventory and deterministic anchor plan; never load a model."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.dflash_correction_bank.artifacts import binding, sha
from .protocol import STREAMS, SCHEMA, MAX_PAIRS, MAX_BLOCK_EVALUATIONS, digest, require, select_anchors


class Resolver:
    def __init__(self, mappings=()):
        self.mappings = [(Path(old), Path(new)) for old, new in mappings]

    def path(self, path):
        value = Path(path)
        for old, new in self.mappings:
            if value.is_relative_to(old):
                return new/value.relative_to(old)
        return value

    def checked(self, reference):
        path = self.path(reference['path'])
        require(path.is_file(), 'Missing retained artifact: '+str(path))
        require(path.stat().st_size == reference['bytes'] and sha(path) == reference['sha256'],
                'Retained artifact checksum mismatch: '+str(path))
        return path

    def load(self, reference):
        return json.loads(self.checked(reference).read_text())

    def present(self, reference):
        try:
            path = self.checked(reference)
            return dict(status='available', original=reference, resolved=binding(path))
        except (OSError, ValueError) as error:
            return dict(status='unavailable', original=reference, reason=str(error))


def inventory_checkpoint(checkpoint, resolver):
    """Every original checkpoint file is required; no substitute search/retraining."""
    require(digest(checkpoint['files']) == checkpoint['checkpoint_identity_sha256'], 'Stored checkpoint identity is inconsistent')
    result = dict(original=checkpoint, files=[], status='available')
    resolved = copy.deepcopy(checkpoint)
    resolved['path'] = str(resolver.path(checkpoint['path']).resolve())
    for name, value in checkpoint['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe checkpoint filename')
        row = resolver.present(dict(path=str(Path(checkpoint['path'])/name), **value))
        result['files'].append(dict(name=name, **row))
        if row['status'] != 'available':
            result['status'] = 'unavailable'
    result['resolved_checkpoint'] = resolved if result['status'] == 'available' else None
    return result


def eos_inventory(config, resolver):
    """EOS IDs come from pinned target configuration, never a guessed default."""
    target = config['target']; files = target['checkpoint_files']
    name = 'generation_config.json' if 'generation_config.json' in files else 'config.json'
    reference = dict(path=str(Path(target['model_path'])/name), **files[name])
    data = resolver.load(reference)
    ids = data.get('eos_token_id')
    ids = [ids] if type(ids) is int else ids
    require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids),
            'Pinned target config has no usable EOS identifiers')
    return sorted(set(ids)), reference


def inventory_campaign(campaign_root, resolver=None):
    resolver = resolver or Resolver()
    root = Path(campaign_root).resolve()
    plan = dict(schema=SCHEMA, scope='five completed main streams; final saved request per stream',
        campaign_root=str(root), mapping=[(str(a), str(b)) for a,b in resolver.mappings],
        cases=[], trace_inventory=[], anchors=[], excluded_streams=['qualification', 'personamem_persona404'],
        planned_requests=94, max_pairs=MAX_PAIRS, max_block_evaluations=MAX_BLOCK_EVALUATIONS,
        training_authorized=False, regeneration_authorized=False, automatic_execution=False)
    for stream, count in STREAMS.items():
        case = dict(stream=stream, final_turn=count, status='unavailable', issues=[])
        plan['cases'].append(case)
        result_path = root/'campaign'/stream/'RESULT.json'
        try:
            case['result'] = binding(result_path)
            result = resolver.load(case['result'])
            require(result['episode_id'] == stream and result['status'] == 'complete' and
                    result['qualification'] is False and result['planned_requests'] == count,
                    'Only the exact completed main stream is eligible')
            # Check every original F/OSD timing trajectory; analysis separately
            # validates all rows before choosing repeat0 for token observations.
            for row in result['scores']:
                if row['arm'] in ('F', 'OSD'):
                    plan['trace_inventory'].append(dict(stream=stream, turn=row['turn'], arm=row['arm'],
                        repeat=row['repeat'], **resolver.present(row['trajectory'])))
            native_dir = root/'campaign'/stream/'native'/stream
            case['native_config'] = binding(native_dir/'NATIVE_CONFIG.json')
            config = resolver.load(case['native_config'])
            require(config['episode_id'] == stream and not config.get('bank_qualification', False), 'Native config is not main-stream config')
            case['projections'] = result['startup']['bank_preparation']['projections']
            # Merely verify the original projection artifact; never rebuild it.
            resolver.checked(case['projections'])
            states = next(r['states'] for r in result['pre_request_states'] if r['turn'] == count)
            canonical = next(r for r in result['canonical'] if r['arm'] == 'BANK' and r['turn'] == count)
            case['input_ids_sha256'], case['output_ids_sha256'] = digest(canonical['input_ids']), digest(canonical['output_ids'])
            case['checkpoint_inventory'] = {}
            for arm in ('F', 'OSD'):
                state = states[arm]
                require(state['turn'] == count and state['after_turn'] == (0 if arm=='F' else count-1),
                        'Wrong pre-final-request checkpoint chronology')
                require(state['state_sha256'] == state['checkpoint']['checkpoint_identity_sha256'], 'State/checkpoint hash mismatch')
                case['checkpoint_inventory'][arm] = inventory_checkpoint(state['checkpoint'], resolver)
                if case['checkpoint_inventory'][arm]['status'] != 'available':
                    case['issues'].append(arm+': exact serving checkpoint unavailable')
            final_scores = [r for r in result['scores'] if r['turn'] == count and r['arm'] in ('F','OSD')]
            require(len(final_scores) == 6 and {(r['arm'], r['repeat']) for r in final_scores} ==
                    {(a,t) for a in ('F','OSD') for t in range(3)}, 'Final paired scores incomplete')
            for row in final_scores:
                require(row['same_generation_valid'] and row['capture'] is False and not row['training_data'] and
                        row['input_ids'] == canonical['input_ids'] and row['output_ids'] == canonical['output_ids'] and
                        row['state_sha256'] == states[row['arm']]['state_sha256'],
                        'Final saved paired comparison is not valid')
            frozen = next(r for r in final_scores if r['arm']=='F' and r['repeat']==0)
            case['frozen_trajectory'] = frozen['trajectory']
            trace = resolver.load(frozen['trajectory'])
            eos_ids, case['eos_configuration'] = eos_inventory(config, resolver)
            anchors = select_anchors(stream, count, canonical['input_ids'], canonical['output_ids'], trace, eos_ids)
            case['selected_anchor_count'] = len(anchors)
            case['selected_anchor_ids'] = [a['anchor_id'] for a in anchors]
            case['status'] = 'ready' if not case['issues'] else 'unavailable'
            if case['status'] == 'ready':
                for a in anchors:
                    a.update(checkpoints={arm:case['checkpoint_inventory'][arm]['resolved_checkpoint'] for arm in ('F','OSD')},
                        projections=binding(resolver.checked(case['projections'])),
                        native_config=case['native_config'], frozen_trajectory=case['frozen_trajectory'])
                plan['anchors'].extend(anchors)
        except (OSError, ValueError, KeyError, StopIteration) as error:
            case['issues'].append(f'{type(error).__name__}: {error}')
            case['status'] = 'unavailable'
    require(len(plan['anchors']) <= MAX_PAIRS, 'Pair budget exceeded')
    plan['planned_draft_block_evaluations'] = 2*len(plan['anchors'])
    plan['available_cases'] = sum(c['status']=='ready' for c in plan['cases'])
    plan['interpretation'] = 'Unavailable evidence remains missing. No checkpoint recreation, later substitution, anchor replacement or automatic launch.'
    return plan
