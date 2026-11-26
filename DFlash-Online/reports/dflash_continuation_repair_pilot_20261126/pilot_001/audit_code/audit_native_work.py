"""Independent compact-evidence audit; standard library only, no production imports.

Full tensors and coefficient vectors are deliberately not retrieved. Matching
fingerprints and runtime assertions are audited as receipts, not reconstructed
proof of the complete native calculation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def locate_bound_json(raw, reference, expected):
    """Resolve relocated compact evidence by bytes/hash, never rewrite locators."""
    raw=Path(raw)
    candidates=list((raw/'audit_inputs').glob('*.json'))+list(raw.glob('*.json'))
    candidates+=list(raw.rglob(Path(reference).name))
    for path in dict.fromkeys(candidates):
        if path.is_file() and path.stat().st_size==expected['bytes'] and identity(path)==expected:
            return path
    raise FileNotFoundError('Bound compact evidence unavailable: '+Path(reference).name)


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def finite_tree(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return True


def normalized_case(case):
    rows = deepcopy(case['rows'])
    for row in rows:
        row.pop('training_s', None)
        row['response_observation'].pop('event_id', None)
    return dict(rows=rows, gradient=case['gradient'], endpoint=case['endpoint'],
                gradient_norm=case['gradient_norm'])


def anchor_blocks(row):
    observations = row['sampled_anchors']
    if len(observations) != 1:
        raise ValueError('Expected one original sampler observation per response')
    positions = observations[0]['positions']
    if len(positions) != 1:
        raise ValueError('Expected microbatch one')
    return len(positions[0])


def scientific_rows_equal(actual, retained):
    keys = ('record_id', 'pass', 'sampled_anchors', 'optimizer_step_after', 'optimizer_stepped')
    return len(actual) == len(retained) and all(
        all(a[key] == b[key] for key in keys) for a, b in zip(actual, retained))


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.totals = {}
        self.limits = [
            'Qualification gradient/parameter/Adam/RNG equality is independently compared between saved fingerprints; tensor bytes are not available to this audit.',
            'Compact coefficient receipts omit beta/HIGH/score/candidate vectors. Their exact FP64 oracle cannot be independently recomputed offline; production flags and saved scalar bounds are checked, with source/CPU mathematics review separate.',
            'Restoration and final export exactness remain authenticated native runtime assertions supplemented by endpoint fingerprints and later external file postflight.',
            'Readout full-vocabulary rival optimality cannot be reconstructed from compact selected logits; this work audit does not claim it.',
            'Fresh per-response hashes bind retained readouts at continuation; historical per-response hashes did not exist. The zero-model rejected attempt is supported by authenticated traceback, unchanged aggregate receipts and pre-attempt file times, not an invented historical byte hash.',
            'Source manifest bytes and phase identities are independently checked; live remote file contents and full model tensors require the separately retained native/postflight checks.',
            'No CUDA, optimizer, teacher, serving or scientific rerun is performed by this auditor.',
        ]

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.failures.append(label)
        return condition

    def eq(self, actual, expected, label):
        return self.check(actual == expected, label)

    def near(self, actual, expected, label, *, absolute=1e-6, relative=2e-6):
        return self.check(math.isfinite(float(actual)) and math.isfinite(float(expected))
                          and abs(actual-expected) <= absolute + relative*abs(expected), label)

    def vector(self, value, label):
        self.check(isinstance(value.get('sha256'), str) and len(value['sha256']) == 64, label+'.digest')
        tensors = value['tensors']
        self.eq(len(tensors), 58, label+'.count')
        self.eq(len({t['name'] for t in tensors}), 58, label+'.unique')
        self.check(all(t['dtype'] == 'torch.bfloat16' for t in tensors), label+'.BF16')
        self.eq(sum(math.prod(t['shape']) for t in tensors), 1048626432, label+'.elements')

    def endpoint(self, value, step, presentations, label):
        for field in ('parameters', 'exp_avg', 'exp_avg_sq'):
            self.vector(value[field], label+'.'+field)
        self.eq(value['steps'], [float(step)]*58, label+'.Adam_counters')
        self.eq(value['global_step'], step, label+'.global_step')
        self.eq(value['presentations'], presentations, label+'.presentations')
        self.eq(value['micro'], presentations, label+'.micro_counter')
        self.eq(value['micro'] % 2, 0, label+'.empty_accumulation')
        self.check(value['gradients_cleared'], label+'.gradients_cleared')
        self.eq(value['options']['lr'], 3e-4, label+'.LR')
        self.eq(value['options']['weight_decay'], .01, label+'.weight_decay')
        self.check(len(value['rng_sha256']) == 64, label+'.RNG_digest')

    def startup(self, directory, receipt, label):
        self.eq(receipt['status'], 'completed', label+'.completed')
        self.eq(receipt['optimizer_steps'], 0, label+'.zero_Adam')
        expected = dict(response_forwards=2, response_backwards=2, optimizer_steps=0)
        self.eq(receipt['work']['attempts'], expected, label+'.attempts')
        self.eq(receipt['work']['completed'], expected, label+'.completed_work')
        self.eq(receipt['work']['unknown'], {k:0 for k in expected}, label+'.unknown')
        physical = receipt['physical_work']
        self.eq(physical['response_observations'], 2, label+'.original_observations')
        self.eq(physical['optimizer_window_observations'], 0, label+'.no_update_observation')
        self.eq(physical['counts']['draft_forwards'], 2, label+'.draft_forwards')
        restored = load(directory/'EXACT_RESTORATION.json')
        self.check(restored['exact_equal'], label+'.exact_restoration_receipt')
        self.eq(restored.get('tensor_differences', []), [], label+'.tensor_differences')
        self.eq(restored.get('metadata_differences', []), [], label+'.metadata_differences')

    def qualification(self, raw, plan, retained):
        summary = load(raw/'qualification/RESULT.json')
        self.eq(summary['status'], 'passed', 'qualification.status')
        self.eq(summary['binding'], plan['binding'], 'qualification.binding')
        self.eq(summary['execution_commit'], plan['execution_commit'], 'qualification.commit')
        self.check(summary['base_disabled_exact'] and summary['retained_BASE_reusable'], 'qualification.flags')
        cases = {}
        for item in summary['cases']:
            key = item['fixture'], item['arm']
            self.check(key not in cases, 'qualification.unique_case')
            path = raw/'qualification'/f'fixture_{key[0]}_{key[1]}'
            case = load(path/'RESULT.json')
            cases[key] = case
            label = f'qualification.fixture{key[0]}.{key[1]}'
            self.eq(case['status'], 'completed', label+'.status')
            self.eq(case['arm'], key[1], label+'.arm')
            self.eq(case['fixture'], key[0], label+'.fixture')
            self.startup(path/'startup', case['startup'], label+'.startup')
            self.endpoint(case['initial_state'], 128, 256, label+'.initial')
            self.endpoint(case['endpoint'], 129, 258, label+'.updated')
            self.vector(case['gradient'], label+'.gradient')
            self.eq(case['initial_state'], case['restored_state'], label+'.restored_digest_fields')
            self.check(case['restoration_exact'], label+'.restoration_flag')
            self.eq(case['counters'], [129,258], label+'.counters')
            self.eq(len(case['rows']), 2, label+'.presentations')
            self.eq(case['actual_work']['response_forwards'], 2, label+'.forwards')
            self.eq(case['actual_work']['draft_forwards'], 2, label+'.no_extra_draft')
            self.eq(case['actual_work']['backwards_attempted'], 2, label+'.backwards_attempted')
            self.eq(case['actual_work']['backwards_completed'], 2, label+'.backwards_completed')
            self.eq(case['schedule_advance_work'], dict(response_forwards=2*key[0], backwards=2*key[0], optimizer_steps=0), label+'.advance')
            for index, row in enumerate(case['rows']):
                reference = retained['records'][2*key[0]+index]
                for field in ('record_id', 'sampled_anchors'):
                    self.eq(row[field], reference[field], label+f'.response{index}.'+field)
                self.eq(row['optimizer_step_after'],128+int(index==1), label+f'.response{index}.step')
                self.eq(row['optimizer_stepped'], index==1, label+f'.response{index}.boundary')
                self.check(finite_tree(row['metrics']) and finite_tree(row['response_observation']),label+f'.response{index}.finite')
            if key == (0,'BASE'):
                self.check(case['retained_BASE_first_window_exact'], label+'.retained_gate')
                for index, row in enumerate(case['rows']):
                    for field in ('loss','ratio_metrics','sum_metrics','sampled_anchors','prefix_allocation'):
                        self.eq(row['response_observation'][field], retained['records'][index]['response_observation'][field], label+f'.historical{index}.'+field)
            if key[1] == 'TAIL':
                self.eq(len(case['coefficient_checks']), 2, label+'.coefficient_observation_count')
                for index, row in enumerate(case['coefficient_checks']):
                    self.eq(row['status'], 'passed', label+f'.coefficient{index}.status')
                    self.check(all(row['checks'].values()), label+f'.coefficient{index}.flags')
                    self.check(finite_tree(row), label+f'.coefficient{index}.finite')
                    self.check(0 <= row['changed_high_positions'] <= row['high_count'] <= row['repair_count'],label+f'.coefficient{index}.counts')
                    self.check(0 <= row['oracle_kappa'] <= .25,label+f'.coefficient{index}.kappa')
                    # Coarse necessary bound only; it does not replace the unavailable per-beta oracle.
                    self.check(row['coefficient_max_error'] <= 5.2e-6,label+f'.coefficient{index}.necessary_max_bound')
                    self.check(row['high_mass_error'] <= 5.2e-6*row['high_count'],label+f'.coefficient{index}.necessary_mass_bound')
        fixtures = sorted({key[0] for key in cases})
        self.check(fixtures in ([0],[0,1]), 'qualification.allowed_fixture_set')
        self.check(3 <= len(cases) <= 6, 'qualification.bounded_case_count')
        for fixture in fixtures:
            label=f'qualification.fixture{fixture}'
            self.eq({arm for f,arm in cases if f==fixture}, {'BASE','TAIL_DISABLED','TAIL'},label+'.arms')
            baseline, disabled, tail = [cases[fixture,a] for a in ('BASE','TAIL_DISABLED','TAIL')]
            self.eq(normalized_case(baseline), normalized_case(disabled),label+'.scientific_neutrality')
            for arm, other in [('disabled',disabled),('TAIL',tail)]:
                self.eq(baseline['initial_state'],other['initial_state'],label+'.'+arm+'.same_initial_state')
                self.eq(baseline['preservation_coefficients'],other['preservation_coefficients'],label+'.'+arm+'.C_coefficients')
                self.eq(baseline['actual_work'],other['actual_work'],label+'.'+arm+'.native_physical_work')
            self.eq(tail['preservation_coefficients'],[r['preservation_coefficients'] for r in tail['coefficient_checks']],label+'.actual_capture_crosscheck')
        if fixtures == [0,1]:
            self.check(not any(row['structurally_active'] for row in cases[0,'TAIL']['coefficient_checks']),'qualification.second_fixture_only_after_neutral')
        self.totals['qualification_cases']=len(cases)
        self.totals['qualification_startup_forwards']=2*len(cases)
        self.totals['qualification_core_forwards']=2*len(cases)
        self.totals['qualification_advance_forwards']=sum(c['schedule_advance_work']['response_forwards'] for c in cases.values())
        return cases

    def training(self, raw, plan, retained, features):
        folder=raw/'arms/TAIL'
        status=load(folder/'STATUS.json')
        event=load(folder/'trainer/events/event_002/RESULT.json')
        self.eq(status['status'],'completed','training.status')
        self.eq(status['training'],event,'training.full_receipt_identity')
        self.eq(status['execution_commit'],plan['execution_commit'],'training.commit')
        for key,value in dict(global_step=256,presentations=512,completed_event=2).items():
            self.eq(status[key],value,'training.'+key)
        for key,value in dict(passes=2,response_presentations=256,optimizer_steps=128,start_step=128,end_step=256,reduction='response_mean',objective='osd_first_error',method='TAIL').items():
            self.eq(event[key],value,'event.'+key)
        self.check(scientific_rows_equal(event['records'],retained['records']),'training.all_natural_schedule_bindings')
        self.eq([r['record_id'] for r in event['features']],[r['record_id'] for r in features],'training.feature_order')
        self.eq(len(set(r['record_id'] for r in features)),128,'training.distinct_feedback')
        self.eq(Counter(r['record_id'] for r in event['records']),Counter({r['record_id']:2 for r in features}),'training.two_presentations_each')
        work=status['actual_work']
        expected_chunks=2*sum(math.ceil(anchor_blocks(r)/8) for r in event['records'])
        for key,value in dict(response_forwards=256,draft_forwards=256,layer_forwards=1280,objective_chunks=expected_chunks,vocabulary_head_calls=expected_chunks*2,backwards_attempted=256,backwards_completed=256).items():
            self.eq(work[key],value,'training.work.'+key)
        self.eq(event['objective_work']['work'],work,'training.objective_work_reconciliation')
        self.eq(event['objective_work']['reference_operations'],0,'training.no_reference_work')
        self.startup(folder/'startup',status['startup'],'training.startup')
        restoration=load(folder/'trainer/PARENT_RESTORATION.json')
        self.eq(restoration['status'],'passed','training.parent_restoration.status')
        self.check(restoration['exact_weights_optimizer_rng_counters_history'],'training.parent_restoration.claim')
        boundary=status['boundary']
        self.eq(boundary['status'],'passed','training.boundary.status')
        self.eq(boundary['counters'],[256,512,2],'training.boundary.counters')
        self.check(boundary['immutable_parent_history'] and boundary['export_exact'],'training.parent_history_export')
        self.eq(boundary['native_adam_entries'],58,'training.Adam_entries')
        self.check(boundary['live']['exact_resume'],'training.live_exact')
        observations=lines(folder/'trainer/events/event_002/RESPONSE_OBSERVATIONS.jsonl')
        windows=lines(folder/'trainer/events/event_002/WINDOW_OBSERVATIONS.jsonl')
        self.eq(len(observations),256,'training.observations_once')
        self.eq(len(windows),128,'training.windows_once')
        self.eq([r['response_observation'] for r in event['records']],observations,'training.observation_reconciliation')
        moved=active=changed=0
        for ordinal,row in enumerate(event['records']):
            tag=f'training.presentation{ordinal}'
            self.eq(row['optimizer_step_after'],128+(ordinal+1)//2,tag+'.step')
            self.eq(row['optimizer_stepped'],ordinal%2==1,tag+'.boundary')
            self.eq(row['pass'],ordinal//128,tag+'.pass')
            self.eq(row['learning_rate'],3e-4 if ordinal%2 else None,tag+'.LR')
            self.check(finite_tree(row['metrics']) and finite_tree(row['response_observation']),tag+'.finite')
            observed=row['response_observation'];pool=observed['continuation_repair'];native=observed['sum_metrics']
            self.eq(pool['continuation_high_count'],native['kplus_priority2_error_count'],tag+'.HIGH_count')
            self.eq(pool['continuation_low_count'],native['kplus_priority1_error_count'],tag+'.LOW_count')
            self.check(0 <= pool['continuation_changed_high_positions'] <= pool['continuation_high_count'],tag+'.changed_count')
            self.check(0 <= pool['continuation_score_min'] <= pool['continuation_score_max'] <= 1,tag+'.score_support')
            low,high=native['kplus_W1'],native['kplus_W2']
            frac=high/(low+high) if low+high else 0
            c=.5*(1-frac)/(1+frac)
            kappa=min(c,.5-c) if low>0 and high>0 else 0
            self.near(pool['continuation_kappa'],kappa,tag+'.kappa')
            self.near(pool['continuation_high_mass_before'],high*native['kplus_coefficient2'],tag+'.native_HIGH_mass',absolute=1e-4)
            self.near(pool['continuation_high_mass_after']-pool['continuation_high_mass_before'],pool['continuation_high_mass_error'],tag+'.HIGH_mass_difference',absolute=1e-6,relative=0)
            self.check(abs(pool['continuation_high_mass_error']) <= 5.2e-6*pool['continuation_high_count'],tag+'.necessary_HIGH_mass_bound')
            self.check(pool['continuation_active_response'] in (0,1),tag+'.activity_boolean')
            active+=int(pool['continuation_active_response']);changed+=int(pool['continuation_changed_high_positions']);moved+=pool['continuation_moved_weight']
        self.eq(status['activity'],dict(original_responses=256,continuation_summaries=256,active_response_count=active,continuation_changed_high_positions=changed),'training.activity_once')
        for ordinal,row in enumerate(windows):
            tag=f'training.window{ordinal}'
            self.eq(row['global_step'],129+ordinal,tag+'.step')
            self.eq(row['window'],ordinal+1,tag+'.window')
            self.eq(row['record_ids'],[r['record_id'] for r in observations[ordinal*2:ordinal*2+2]],tag+'.response_binding')
            self.near(row['mean_response_loss'],sum(r['loss'] for r in observations[ordinal*2:ordinal*2+2])/2,tag+'.response_mean',absolute=0,relative=0)
            self.eq(row['learning_rate'],3e-4,tag+'.LR')
            self.eq(row['clipped'],row['gradient_norm']>1,tag+'.clipping')
        self.totals.update(training_steps=128,training_presentations=256,training_work=work,
                           training_startup_forwards=2,active_responses=active,changed_HIGH_occurrences=changed,moved_weight_sum=moved)
        return status

    def readout(self, raw, contract):
        root=raw/'mechanism';status=load(root/'STATUS.json');endpoint=load(root/'ENDPOINT_STATE.json')
        self.eq(status['status'],'completed','readout.status')
        self.eq(status['work'],endpoint['work'],'readout.work_reconciliation')
        self.eq(status['completed_primary_responses'],128,'readout.primary_responses')
        self.eq(status['completed_retention_responses'],8,'readout.retention_responses')
        self.eq(status['unattempted_response_forwards'],0,'readout.unattempted')
        primary = contract['features']
        retention = contract['legacy_binding']['sets']['parent_training']
        self.eq(len(primary),128,'readout.bound_primary_count')
        self.eq(len(retention),8,'readout.bound_retention_count')
        bindings = primary + retention + [primary[0]]
        rows=list(sorted((root/'primary').glob('*.json')))+list(sorted((root/'parent_training').glob('*.json')))+[root/'qualification_repeat.json']
        self.eq(len(rows),137,'readout.file_count')
        totals=Counter()
        for ordinal,path in enumerate(rows):
            row=load(path);tag=f'readout.response{ordinal}'
            binding=bindings[ordinal]
            for field in ('record_id','anchors','feature_identity'):
                self.eq(row[field],binding[field],tag+'.sealed_'+field)
            self.eq(row['status'],'passed',tag+'.status')
            self.eq(row['endpoint'],'TAIL',tag+'.label')
            self.eq(row['evaluation_mode'],'eval/no_grad',tag+'.mode')
            self.eq(row['rng_before'],row['rng_after'],tag+'.RNG')
            for field in ('parameters_unchanged','gradients_unchanged','runtime_restored','rng_preserved'):
                self.check(row[field],tag+'.'+field)
            n=len(row['anchors']);self.check(n in (8,64),tag+'.geometry')
            self.eq(len(row['blocks']),n,tag+'.blocks')
            expected=dict(response_reads_attempted=1,response_reads_completed=1,draft_forwards_attempted=1,draft_forwards_completed=1,head_calls_attempted=math.ceil(n/8),head_calls_completed=math.ceil(n/8),blocks_completed=n,backwards=0,optimizer_steps=0,teacher_forwards=0,feedback_generations=0,serving_generations=0)
            self.eq(row['work'],expected,tag+'.work')
            totals.update(row['work'])
        self.eq(dict(totals),status['work'],'readout.summed_work')
        self.eq(totals['response_reads_completed'],137,'readout.total_response_forwards')
        self.eq(totals['head_calls_completed'],1040,'readout.total_head_calls')
        self.eq(totals['blocks_completed'],8320,'readout.total_blocks')
        first=load(root/'primary/000.json');repeat=load(root/'qualification_repeat.json')
        for field in ('record_id','feature_identity','anchors','blocks','geometry','projection_shapes','rng_before','rng_after'):
            self.eq(first[field],repeat[field],'readout.repeat.'+field)
        loading=load(root/'LOADING.json')
        self.eq(loading['rng_before'],loading['rng_after'],'readout.loading_RNG')
        self.eq(loading['mapping']['tensors'],58,'readout.loading_mapping')
        self.check(loading['mapping']['exact_copy'] and not loading['optimizer_created'],'readout.exact_load_without_optimizer')
        self.eq(endpoint['status'],'passed','readout.endpoint_state')
        if status.get('resumed_unexecuted'):
            old_status=load(root/'attempts/001/STATUS.json')
            new_status=load(root/'attempts/002/STATUS.json')
            old_endpoint=load(root/'attempts/001/ENDPOINT_STATE.json')
            new_endpoint=load(root/'attempts/002/ENDPOINT_STATE.json')
            old_loading=load(root/'attempts/001/LOADING.json')
            new_loading=load(root/'attempts/002/LOADING.json')
            self.eq(new_status,status,'readout.final_attempt_status_copy')
            self.eq(old_endpoint['work'],old_status['work'],'readout.old_attempt_work')
            self.eq(new_endpoint['work'],status['attempt_work'],'readout.new_attempt_work')
            self.eq(new_endpoint['work'],endpoint['attempt_work'],'readout.new_endpoint_work')
            cumulative=Counter(old_endpoint['work']);cumulative.update(new_endpoint['work'])
            self.eq(dict(cumulative),status['work'],'readout.two_attempt_sum')
            self.eq(endpoint['cumulative_attempts'],2,'readout.cumulative_attempt_count')
            self.eq(old_loading['mapping'],new_loading['mapping'],'readout.same_exact_export_load')
            self.eq(new_loading,loading,'readout.current_loading_receipt')
            for attempt,receipt in enumerate((old_endpoint,new_endpoint),1):
                self.eq(receipt['status'],'passed',f'readout.attempt{attempt}.endpoint_state')
                self.check(receipt['parameters_unchanged'] and receipt['modes_unchanged'] and not receipt['optimizer_created'],f'readout.attempt{attempt}.state_preserved')
            self.near(status['wall_s'],old_status['wall_s']+status['attempt_wall_s'],'readout.wall_sum',absolute=1e-7,relative=0)
            self.eq(old_status['completed_primary_responses'],73,'readout.old_primary_coverage')
            self.eq(old_status['completed_retention_responses'],0,'readout.old_retention_coverage')
        self.totals['readout_work']=dict(totals)

    def resume_provenance(self, raw, plan):
        paths=sorted(raw.glob('resume_00[12]/REPAIR.json'))
        if not paths:
            return
        allowed={'source_root','source_manifest','source_identity','execution_commit',
                 'training_execution_commit','original_plan','original_plan_identity','prior_charged_roots',
                 'limits','elapsed_extension_authorization','elapsed_extension_authorization_identity'}
        allowed_exact={'experiments/continuation_repair_pilot/launch.py',
                       'experiments/continuation_repair_pilot/readout.py',
                       'experiments/continuation_repair_pilot/analysis.py',
                       'experiments/continuation_repair_pilot/README.md',
                       'reports/acl2027_method_development/SEQUENCE.md'}
        for path in paths:
            repair=load(path);execution=load(path.parent/'EXECUTION_PLAN.json')
            prefix=path.parent.name
            actual={key for key in set(plan)|set(execution) if plan.get(key)!=execution.get(key)}
            self.check(actual<=allowed,'resume.only_source_and_orchestration_plan_changes')
            self.eq(execution['binding'],plan['binding'],prefix+'.scientific_binding_unchanged')
            self.eq(execution['training_execution_commit'],plan['execution_commit'],prefix+'.training_source')
            self.eq(repair['original_training_commit'],plan['execution_commit'],prefix+'.receipt_training_source')
            self.eq(execution['execution_commit'],repair['orchestration_execution_commit'],prefix+'.orchestration_source')
            self.eq(execution['original_plan_identity'],identity(raw/'sealed/PLAN.json'),prefix+'.original_plan_bytes')
            self.eq(repair['original_plan_identity'],execution['original_plan_identity'],prefix+'.original_plan_identity_binding')
            self.check(repair['original_settings_unchanged'],prefix+'.settings_receipt')
            self.eq(execution['source_identity'],repair['repair_source_identity'],prefix+'.new_source_manifest_binding')
            self.eq(plan['source_identity'],repair['original_source_identity'],prefix+'.old_source_manifest_binding')
            self.check(bool(repair['changed_files']) and all(name in allowed_exact or
                name.startswith('tests/test_continuation_repair_') and name.endswith('.py') or
                name.startswith('reports/dflash_continuation_repair_pilot_20261126/')
                for name in repair['changed_files']),'resume.source_changes_within_narrow_repair')
            for relative,expected in repair['preserved_failures'].items():
                self.eq(identity(path.parent/'prior'/relative),expected,prefix+'.preserved_failure_hash.'+relative)
            if 'limits' in plan:
                expected_limits=dict(plan['limits'])
                if prefix=='resume_002':expected_limits['elapsed_s']=43200
                self.eq(execution['limits'],expected_limits,prefix+'.only_elapsed_limit_changed')
        for label,path in [('mechanism',raw/'mechanism/STATUS.json'),('campaign',raw/'STATUS.json')]:
            status=load(path)
            self.eq(status['execution_commit'],execution['execution_commit'],label+'.resumed_source')
            self.eq(status['training_execution_commit'],plan['execution_commit'],label+'.original_training_source')
        protocol=load(raw/'evaluation/PROTOCOL.json')
        self.eq(protocol['execution_commit'],execution['execution_commit'],'serving.resumed_source')
        self.totals['training_execution_commit']=plan['execution_commit']
        self.totals['orchestration_execution_commit']=execution['execution_commit']
        if len(paths)==2:
            authorization_path=locate_bound_json(raw,execution['elapsed_extension_authorization'],execution['elapsed_extension_authorization_identity'])
            authorization=load(authorization_path)
            clock=load(raw/'CLOCK.json')
            expected=dict(kind='elapsed_limit_extension',original_plan_binding=plan['binding'],
                original_clock_epoch=clock['first_gpu_action_epoch'],previous_elapsed_s=14400,
                approved_elapsed_s=43200,occupied_card_s=14400,
                authorized_scope='remaining readouts and evaluation',user_answer='Extend elapsed limit to 12 hours')
            for key,value in expected.items():self.eq(authorization[key],value,'authorization.'+key)
            self.eq(repair['elapsed_extension_authorization_identity'],identity(authorization_path),'authorization.repair_identity')
            self.eq(repair['original_limits'],plan['limits'],'authorization.original_limits')
            self.eq(repair['effective_limits'],execution['limits'],'authorization.effective_limits')
            self.check(repair['elapsed_extension']['only_elapsed_limit_changed'],'authorization.only_elapsed_flag')
            self.check(authorization['recorded_epoch']<=clock['processes'][-2]['start_epoch'],'authorization.precedes_resumed_work')
            source_specs=[('original',plan['source_manifest'],plan['source_identity'],plan['execution_commit']),
                          ('first_repair',load(paths[0].parent/'EXECUTION_PLAN.json')['source_manifest'],
                           load(paths[0].parent/'EXECUTION_PLAN.json')['source_identity'],load(paths[0].parent/'EXECUTION_PLAN.json')['execution_commit']),
                          ('CPU_preflight',authorization['prior_preflight_source_manifest'],authorization['prior_preflight_source_identity'],None),
                          ('resumed',execution['source_manifest'],execution['source_identity'],execution['execution_commit'])]
            sources={}
            for label,reference,bound,commit in source_specs:
                source=load(locate_bound_json(raw,reference,bound))
                self.check(source['clean'],label+'.clean_source')
                if commit is not None:self.eq(source['commit'],commit,label+'.source_commit')
                sources[label]=dict(identity=bound,commit=source['commit'])
            self.eq(len({row['identity']['sha256'] for row in sources.values()}),4,'sources.four_distinct_manifests')
            self.check(all(p.get('execution_commit')!=sources['CPU_preflight']['commit'] for p in clock['processes']),'sources.preflight_source_did_not_run_GPU_phase')
            self.totals['source_identity_chain']=sources
            self.totals['elapsed_limit_seconds']=43200
            self.totals['occupied_limit_seconds']=14400

    def ledger_and_clock(self, raw, cases):
        ledger=load(raw/'ADAM_LEDGER.json')['calls']
        self.eq([r['token'] for r in ledger],list(range(len(ledger))),'ledger.unique_monotone_tokens')
        self.check(all(r['status']=='completed' for r in ledger),'ledger.all_actual_attempts_completed')
        qualification=[r for r in ledger if r['phase']=='qualification']
        training=[r for r in ledger if r['phase']=='training']
        self.eq(len(qualification),len(cases),'ledger.qualification_cases')
        self.check(len(qualification)<=6,'ledger.qualification_ceiling')
        self.eq(Counter(r['arm'] for r in qualification),Counter(key[1] for key in cases),'ledger.qualification_arms')
        self.eq(len(training),128,'ledger.committed_steps')
        self.check(all(r['arm']=='TAIL' for r in training),'ledger.only_TAIL_training')
        self.eq(len(ledger),len(qualification)+len(training),'ledger.no_undeclared_optimizer_phase')
        clock=load(raw/'CLOCK.json');processes=clock['processes']
        resumed = (raw/'resume_001/REPAIR.json').exists()
        twice_resumed = (raw/'resume_002/REPAIR.json').exists()
        expected_phases = (['qualification','training','readout','readout','readout','evaluation'] if twice_resumed else
                           ['qualification','training','readout','readout','evaluation'] if resumed else
                           ['qualification','training','readout','evaluation'])
        self.eq([p['phase'] for p in processes],expected_phases,'clock.fixed_phases')
        self.eq([p['exit_code'] for p in processes],[0,0,1,1,0,0] if twice_resumed else [0,0,1,0,0] if resumed else [0,0,0,0],'clock.preserved_attempt_exit_codes')
        if resumed:
            prior=load(raw/'resume_001/prior/CLOCK.json')
            repair=load(raw/'resume_001/REPAIR.json')
            self.eq(clock['first_gpu_action_epoch'],prior['first_gpu_action_epoch'],'clock.original_epoch_preserved')
            self.eq(processes[:3],prior['processes'],'clock.original_processes_preserved')
            self.eq(repair['additional_adam_applications'],0,'resume.zero_extra_Adam')
            self.eq(repair['original_adam_applications'],len(ledger),'resume.Adam_ledger_unchanged')
            self.eq(load(raw/'resume_001/prior/ADAM_LEDGER.json')['calls'],ledger,'resume.original_Adam_calls_retained')
            old=load(raw/'mechanism/attempts/001/STATUS.json')
            new=load(raw/'mechanism/attempts/002/STATUS.json')
            self.eq(old['status'],'failed','resume.old_failure_preserved')
            self.eq(old['work']['response_reads_completed'],74,'resume.prior74')
            self.eq(new['attempt_work']['response_reads_completed'],63,'resume.new63')
            self.eq(new['work']['response_reads_completed'],137,'resume.cumulative137')
            self.eq(new['qualification_response_forwards'],0,'resume.no_repeat_qualification_forward')
            self.eq(new['reused_qualification_response_forwards'],1,'resume.original_repeat_reused')
            receipt=load(raw/'mechanism/RESUME_BINDING.json')
            for relative,expected in receipt['retained_file_identities'].items():
                if relative.startswith('primary/') or relative=='qualification_repeat.json':
                    target=raw/'mechanism'/relative
                else:
                    target=raw/'mechanism/attempts/001'/relative
                self.eq(identity(target),expected,'resume.authenticated_prior_file.'+relative)
            self.totals['readout_attempts']=sum(p['phase']=='readout' for p in processes)
            self.totals['readout_model_forward_attempts']=2
            self.totals['readout_forward_counts_by_process']=[74,0,63] if twice_resumed else [74,63]
            self.totals['readout_prior_completed']=74
            self.totals['readout_resumed_completed']=63
            if twice_resumed:
                second_prior=load(raw/'resume_002/prior/CLOCK.json')
                second_repair=load(raw/'resume_002/REPAIR.json')
                self.eq(processes[:4],second_prior['processes'],'clock.both_failed_processes_preserved')
                self.eq(clock['first_gpu_action_epoch'],second_prior['first_gpu_action_epoch'],'clock.second_resume_original_epoch')
                self.eq(load(raw/'resume_002/prior/ADAM_LEDGER.json')['calls'],ledger,'resume2.no_additional_Adam')
                evidence=second_repair['rejected_preparation']
                self.eq(evidence['status'],'passed','resume2.pre_model_evidence_status')
                self.eq(evidence['new_forward_evidence'],0,'resume2.pre_model_zero_new_forward_evidence')
                self.eq(evidence['existing_readouts'],74,'resume2.pre_model_retained_count')
                self.check(evidence['authenticated_traceback_before_adapter_load'],'resume2.pre_model_traceback_bound')
                self.check(evidence['retained_aggregate_hashes_unchanged'],'resume2.retained_aggregates_unchanged')
                self.check(not evidence['historical_individual_byte_equality_proven'],'resume2.historical_hash_limitation_disclosed')
                self.near(evidence['rejected_process_occupied_s'],processes[3]['end_epoch']-processes[3]['start_epoch'],'resume2.failed_checker_cost',absolute=1e-7,relative=0)
                rejected_log=raw/'resume_001/readout.log'
                self.eq(identity(rejected_log),evidence['rejected_log_identity'],'resume2.failed_checker_log_identity')
                text=rejected_log.read_text()
                self.check('resume_completed_readouts' in text and 'ValueError: Resume requires the identified storage-writer failure after73 primary readouts' in text,'resume2.failed_before_adapter_trace')
                for relative,expected in evidence['current_readout_identities'].items():
                    self.eq(identity(raw/relative),expected,'resume2.retained_current_binding.'+relative)
                self.totals['readout_process_attempts']=3
                self.totals['pre_model_rejection_occupied_seconds']=processes[3]['end_epoch']-processes[3]['start_epoch']
        self.check(all(a['end_epoch']<=b['start_epoch'] for a,b in zip(processes,processes[1:])),'clock.serial_phases')
        occupied=sum(p['end_epoch']-p['start_epoch'] for p in processes)
        self.near(clock['occupied_card_s'],occupied,'clock.card_seconds',absolute=1e-7,relative=0)
        self.check(clock['elapsed_s']<=(43200 if twice_resumed else 14400) and occupied<=14400,'clock.time_caps')
        self.eq(load(raw/'STATUS.json')['status'],'completed','campaign.completed')
        self.totals['Adam_applications']=len(ledger)
        self.totals['occupied_card_seconds']=occupied
        self.totals['campaign_elapsed_seconds']=clock['elapsed_s']


    def serving_work(self, raw):
        status=load(raw/'evaluation/STATUS.json');ledger=load(raw/'evaluation_work.json')
        for key,value in dict(status='completed',planned=792,attempted=792,returned=792,completed_valid=792,invalid=0,unknown=0,unattempted=0,pending=None).items():
            self.eq(status[key],value,'serving.'+key)
        expected=dict(scored_generation=768,warmup_generation=24)
        self.eq(ledger['attempted'],expected,'serving.attempted_budget')
        self.eq(ledger['completed'],expected,'serving.completed_budget')
        self.eq(ledger['pending'],{},'serving.pending_work')
        rows=lines(raw/'evaluation/MEASUREMENTS.jsonl')
        self.eq(len(rows),792,'serving.raw_row_count')
        self.eq(Counter(r['phase'] for r in rows),Counter(scored=768,warmup=24),'serving.raw_phase_counts')
        self.eq({r['method'] for r in rows},{'BASE','TAIL'},'serving.genuine_two_arms')
        self.check(all(r['valid_measurement'] and not r['errors'] for r in rows),'serving.all_measurement_validation_flags')
        scored=[r for r in rows if r['phase']=='scored']
        self.eq(Counter((r['request_id'],r['method']) for r in scored),Counter({(r['request_id'],arm):3 for r in scored for arm in ('BASE','TAIL')}),'serving.three_repeats_each')
        self.eq(len({r['request_id'] for r in scored}),128,'serving.fixed_request_count')
        self.eq(len({(r['request_id'],r['method'],r['repeat']) for r in scored}),768,'serving.no_duplicate_repeats')
        self.totals['serving_scored_generations']=768
        self.totals['serving_warmup_generations']=24
        self.totals['serving_total_generations']=792



def audit(raw, *, retained_event, feature_manifest, base_status, diagnostic_contract):
    raw=Path(raw);a=Audit();plan=load(raw/'sealed/PLAN.json')
    unsigned={k:v for k,v in plan.items() if k!='binding'}
    calculated=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.eq(calculated,plan['binding'],'plan.content_binding')
    a.eq(identity(diagnostic_contract),plan['retained_diagnostic_contract_identity'],'authenticated.diagnostic_contract')
    diagnostic=load(diagnostic_contract)
    for remote,local,label in [(plan['retained_event'],retained_event,'retained_event'),(plan['feature_manifest'],feature_manifest,'feature_manifest'),(plan['base_status'],base_status,'BASE_status')]:
        a.eq(identity(local),plan['sealed_files'][remote],'authenticated.'+label)
    retained=load(retained_event);features=load(feature_manifest);base=load(base_status)
    a.eq(base['training'],retained,'retained.BASE_training_receipt')
    a.eq(base['checkpoint_identity'],plan['base_export_identity'],'retained.BASE_export_binding')
    a.eq([base['global_step'],base['presentations'],base['completed_event']],[256,512,2],'retained.BASE_counters')
    a.check(plan['reuse_BASE'] and plan['production_admitted'],'plan.admitted_reuse_path')
    a.eq(plan['limits']['qualification_adam_steps'],6,'plan.disposable_limit')
    a.eq(plan['limits']['training_adam_steps'],128,'plan.training_limit')
    a.eq(plan['limits']['reserve_bytes'],400000000000,'plan.reserve')
    a.eq(plan['limits']['allocated_bytes'],24000000000,'plan.storage_cap')
    cases=a.qualification(raw,plan,retained)
    a.training(raw,plan,retained,features)
    a.readout(raw, diagnostic)
    a.ledger_and_clock(raw,cases)
    a.resume_provenance(raw,plan)
    a.serving_work(raw)
    return dict(status='passed' if not a.failures else 'failed',checks=a.checks,failures=a.failures,
                totals=a.totals,limitations=a.limits,execution_commit=plan['execution_commit'],
                execution_commit_scope='Original qualification/training and first74 readouts; resumed phase source appears in totals/source_identity_chain',
                independence='Independent standard-library compact-receipt audit; no production validators imported. Reviewer authored serving/readout implementation and reviewed native integration in shared context, so this is not a blind or author-independent experiment.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',required=True)
    parser.add_argument('--retained-event',required=True)
    parser.add_argument('--feature-manifest',required=True)
    parser.add_argument('--base-status',required=True)
    parser.add_argument('--diagnostic-contract',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve earlier audit output; select a new explicit output path')
    try:
        result=audit(args.raw,retained_event=args.retained_event,feature_manifest=args.feature_manifest,base_status=args.base_status,diagnostic_contract=args.diagnostic_contract)
    except Exception as error:
        result=dict(status='audit_incomplete',error=repr(error),scope='Audit-only failure; no experimental retry')
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        raise
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks','totals')}))
    if result['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
