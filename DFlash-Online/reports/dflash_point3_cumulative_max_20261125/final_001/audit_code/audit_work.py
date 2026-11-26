"""Independent CPU-only Point3 receipt/scalar/accounting audit.

This script imports no experimental implementation and never loads a model,
optimizer tensor or GPU runtime. Native tensor equality is a retained-receipt
claim; stored scalar arithmetic, branch decisions and work accounting are
recomputed here. Missing evidence is incomplete, never inferred as a pass.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ARMS = ('F_C_SUM', 'F_C_MAX')
METHODS = ('C', *ARMS)
HISTORICAL_CARD_S = 5790.93129579185
PARENT_HASH = '728770a7abf0ba3ef96805ef760997976ad7bedbbee1d8debe165e785a4d98a2'
EXPECTED = {
    'C': ('07df3714c2a24837b4bbc16bd2c81c6ea8d5aba4c7cc0cabfeef55271eaa0b75',
          '0ffe862ef3f0f181aa8048df73d25f7fe4f3d9170df77fe8bd831b7d67b8d5ef'),
    'F_C_SUM': ('435c74d89eb75d1f30e2beeffd9e12a85baf53d0e4ebf5387928e697d81c74d6',
                '9b7f2781275fb76a5465175aa887caf84284b979e9a29343714bc0b8b7b6637f'),
}


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def finite_values(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_values(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_values(item) for item in value)
    return True


def scientific_case(case):
    rows = deepcopy(case['rows'])
    for row in rows:
        row.pop('training_s', None)
        row['response_observation'].pop('event_id', None)
    return dict(rows=rows, **{name: case[name] for name in
        ('gradient_identity', 'endpoint', 'native_preclip_norm', 'adam_completed')})


class Audit:
    def __init__(self, root, historical_root=None, control_status=None):
        self.root = Path(root).resolve()
        self.historical_root = None if historical_root is None else Path(historical_root)
        self.control_status = None if control_status is None else Path(control_status)
        self.checked = 0
        self.inputs = {}
        self.sections = {}
        self.plan = self.load('CONTRACT.json')

    def load(self, path, lines=False):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        data = path.read_text()
        self.inputs[str(path)] = file_identity(path)
        return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)

    def check(self, condition, message):
        self.checked += 1
        if not condition:
            raise AssertionError(message)

    def equal(self, actual, expected, message):
        self.check(actual == expected, message + ': ' + repr(actual)[:220])

    def close(self, actual, expected, message, *, atol=1e-11, rtol=1e-10):
        self.check(math.isfinite(actual) and math.isfinite(expected)
                   and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), message)

    def section(self, name, function):
        before = self.checked
        try:
            details = function()
            result = dict(status='passed', **(details or {}))
        except FileNotFoundError as error:
            result = dict(status='incomplete', missing=str(error.filename))
        except (AssertionError, KeyError, TypeError, ValueError, IndexError) as error:
            result = dict(status='failed', error=repr(error))
        result['checks'] = self.checked-before
        self.sections[name] = result

    def contract(self):
        p = self.plan
        self.equal(p['schema'], 'point3_cumulative_max_v1', 'schema')
        self.equal(p['arms'], list(ARMS), 'arms')
        self.equal(p['cases'], list(METHODS), 'qualification cases')
        self.equal(p['parent_recovery_identity']['sha256'], PARENT_HASH, 'original parent')
        limits = p['limits']
        for key, value in dict(elapsed_s=21600, occupied_card_s=14400,
            allocated_bytes=25_000_000_000, reserve_bytes=400_000_000_000,
            qualification_adam_calls=6, qualification_attempts=2,
            oracle_backwards_total=60, event_adam_calls_per_arm=128,
            attempted_adam_calls=262, generation_calls=1188).items():
            self.equal(limits[key], value, 'limit '+key)
        self.check(p['production_admitted'] and p['resource_admission']['admitted'], 'plan admission')
        self.equal(p['parent_counters'], [128,256,1], 'parent counters')
        self.equal(p['final_counters'], [256,512,2], 'final counters')
        self.equal(p['committed_adam_calls'], 256, 'two event budget')
        self.equal(p['evaluation']['total_generations'], 1188, 'generation budget')
        self.equal(p['new_feedback_generations'], 0, 'no new feedback')
        encoded = json.dumps({k:v for k,v in p.items() if k != 'binding'},
                             sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                             allow_nan=False).encode()
        self.equal(p['binding'], hashlib.sha256(encoded).hexdigest(), 'sealed contract digest')
        return dict(execution_commit=p['execution_commit'], science='fixed SUM/MAX comparison')

    def retrieval(self):
        receipt = self.load(self.root.parent/'RETRIEVAL.json')
        self.equal(receipt['status'],'passed','complete evidence retrieval')
        self.check(not receipt['remote_archive_created'] and not receipt['model_tensors_transferred'],
                   'compact evidence retrieval only')
        manifest = self.load('RETRIEVAL_MANIFEST.json')
        self.equal(file_identity(self.root/'RETRIEVAL_MANIFEST.json')['sha256'],
                   receipt['manifest_sha256'],'retrieval manifest identity')
        self.equal(len(manifest),receipt['files'],'complete manifest file count')
        self.equal(sum(row['bytes'] for row in manifest.values()),receipt['bytes'],'complete compact-byte count')
        for relative,expected in manifest.items():
            path = self.root/relative
            self.check(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                       'relative retrieval path')
            self.equal(file_identity(path),expected,'retrieved artifact hash '+relative)
        return dict(files=len(manifest),bytes=receipt['bytes'],
                    independently_hashed_all_compact_artifacts=True)

    def work_ledger(self):
        ledger = self.load('STEP_LEDGER.json')
        self.equal(ledger['contract_identity'], file_identity(self.root/'CONTRACT.json'), 'work contract')
        attempts, calls, tokens = ledger['attempts'], ledger['calls'], ledger['token_backwards']
        self.check(1 <= len(attempts) <= 2, 'qualification attempt budget')
        self.equal([row['number'] for row in attempts], list(range(1,len(attempts)+1)), 'attempt order')
        self.equal([row['number'] for row in calls], list(range(1,len(calls)+1)), 'Adam reservation order')
        self.equal([row['number'] for row in tokens], list(range(1,len(tokens)+1)), 'oracle reservation order')
        self.check(len(calls) <= 262 and len(tokens) <= 60, 'aggregate attempted-work caps')
        qualification = [row for row in calls if row['phase']=='qualification']
        self.check(len(qualification) <= 6, 'disposable-step cap')
        for attempt in attempts:
            number = attempt['number']
            selected = [row for row in qualification if row['attempt']==number]
            selected_tokens = [row for row in tokens if row['attempt']==number]
            self.equal([row['case'] for row in selected], list(METHODS[:len(selected)]), 'case order')
            self.check(len(selected_tokens) <= 30, 'per-attempt oracle cap')
            if number == 2:
                self.check(bool(attempt['correction_note']) and attempts[0]['status']=='failed', 'retry evidence')
            if attempt['status']=='completed':
                self.equal(len(selected), 3, 'completed qualification Adam count')
                self.equal(len(selected_tokens), 28, 'completed qualification oracle count')
                self.check(all(row['status']=='completed' for row in selected+selected_tokens), 'unknown qualified work')
        self.equal(attempts[-1]['status'], 'completed', 'last qualification passed')
        for arm in ARMS:
            event = [row for row in calls if row['phase']=='event' and row['case']==arm]
            self.equal([row['window'] for row in event], list(range(1,129)), 'event window coverage '+arm)
            self.check(all(row['status']=='completed' for row in event), 'unknown event Adam work '+arm)
        self.equal(len(calls), len(qualification)+256, 'no unlabelled extra Adam work')
        return dict(attempts=len(attempts), attempted_adam=len(calls),
            disposable_adam=len(qualification), event_adam=256, oracle_backwards=len(tokens),
            unknown_adam=sum(row['status']!='completed' for row in calls),
            unknown_oracles=sum(row['status']!='completed' for row in tokens))

    def restoration(self, row, step=128, presentations=256):
        self.equal(row['status'], 'passed', 'restoration status')
        self.check(row['exact_resume'] and not row['mismatches'], 'exact restoration receipt')
        self.equal((row['global_step'],row['presentations']), (step,presentations), 'restored counters')
        self.equal(row['extra_optimizer_steps'], 0, 'no restoration Adam')
        self.equal(row['extra_model_passes'], 0, 'no restoration forward')
        for key in ('runtime_exact', 'ancillary_exact'):
            if key in row:
                self.check(row[key], 'restored '+key)

    def work(self, row, name):
        self.equal(row['pending'], [], name+' pending work')
        self.equal(row['attempted'], row['completed'], name+' all physical work completed')
        self.check(all(isinstance(v,int) and v >= 0 for v in row['completed'].values()), name+' finite counts')

    def startup(self, row):
        self.equal(row['status'], 'completed', 'startup complete')
        self.equal(row['optimizer_steps'], 0, 'startup Adam count')
        self.equal(row['work']['attempts'], dict(response_forwards=2, response_backwards=2, optimizer_steps=0), 'startup attempts')
        self.equal(row['work']['completed'], row['work']['attempts'], 'startup completed work')
        self.check(not any(row['work']['unknown'].values()), 'startup unknown work')

    def qualification(self):
        base = Path('qualification/native')
        result = self.load(base/'RESULT.json')
        self.equal(result['status'], 'completed', 'qualification complete')
        self.equal(result['execution_commit'], self.plan['execution_commit'], 'qualification source')
        self.equal(result['contract_identity'], file_identity(self.root/'CONTRACT.json'), 'qualification binding')
        self.equal(result['native_adam_calls'], 3, 'qualification Adam count')
        for key in ('exports','new_feedback','generations','full_vectors_written'):
            self.equal(result[key], 0, 'qualification '+key)
        self.restoration(result['final_restoration'])
        self.startup(result['startup'])
        self.work(result['work'], 'qualification')
        self.equal(result['work']['native_adam_steps'], 3, 'physical Adam count')
        for name in METHODS:
            row = self.load(base/(name+'.json'))
            self.equal(row['status'], 'completed', 'case complete '+name)
            self.equal((row['adam_attempts'],row['adam_completed']), (1,1), 'case Adam '+name)
            endpoint = row['endpoint']
            # TrainerCore's micro counter is cumulative; an even value means
            # empty two-response accumulation, not a reset to zero.
            self.equal((endpoint['global_step'],endpoint['presentations'],endpoint['micro']), (129,258,258), 'case counters '+name)
            self.equal(endpoint['steps'], [129.]*58, 'case Adam entries '+name)
            self.check(endpoint['gradients_cleared'], 'case gradients cleared '+name)
            self.restoration(row['restoration'])
            if name in EXPECTED:
                gradient, parameters = EXPECTED[name]
                self.equal(row['gradient_identity']['sha256'], gradient, 'retained preclip gradient '+name)
                self.equal(endpoint['parameters']['sha256'], parameters, 'retained native endpoint '+name)
                self.check(row['historical_reproduction']['complete_recorded_scientific_identity'], 'complete historical receipt '+name)
                self.equal(row['historical_reproduction']['mismatches'], [], 'historical mismatches '+name)
                if self.historical_root is not None:
                    old = self.load(self.historical_root/('C.json' if name=='C' else 'F_C.json'))
                    self.equal(scientific_case(row), scientific_case(old), 'independent historical scientific receipt '+name)
                    self.equal(row['coefficient_identity'], old['coefficient_identity'], 'historical coefficients '+name)
        reference = self.load(base/'NATIVE_C_REFERENCE.json')
        self.equal(reference['status'], 'passed', 'no-step C status')
        for key in ('bitwise_native_preclip_parity','scientific_state_unchanged','rng_restored','gradients_cleared'):
            self.check(reference[key], 'no-step C '+key)
        self.equal(reference['state_before'], reference['state_after'], 'no-step complete scientific state')
        self.equal(reference['gradient_identity'], result['cases']['C']['gradient_identity'], 'no-step actual gradient identity')
        for key, count in dict(response_forwards=2,response_backwards=2,native_adam_steps=0,
                               committed_presentations=0,sampling_calls=0).items():
            self.equal(reference[key], count, 'no-step work '+key)
        return dict(native_receipts='exact retained receipt comparison; full tensors not independently reloaded',
                    historical_receipt_files_supplied=self.historical_root is not None)

    def numerical_gates(self):
        base = Path('qualification/native/max_numerics')
        gates = self.load(base/'QUALIFICATION.json')
        self.equal(gates['status'], 'passed', 'MAX gates')
        self.equal(gates['reference_backward_count'], 28, 'MAX oracle count')
        self.equal(gates['old_hidden_state_gate'], 'failed_retained_diagnostic', 'historical failure not relabelled')
        self.equal(gates['repeat_exact'], dict(scores=True, allocations=True), 'fast repeat identity')
        self.check(gates['selection']['selected_before_scores'], 'pools selected before scores')
        self.equal([len(row['positions']) for row in gates['selected_pool_receipts']], [13,15], 'fixed complete pools')
        active = 0
        for pool in gates['selected_pool_receipts']:
            b, r = pool['beta'], pool['reference_coefficients_fp32']
            scores = pool['reference_scores_fp32']
            offsets = [value-min(scores) for value in scores]
            mean = math.fsum(x*w for x,w in zip(offsets,b))/math.fsum(b)
            centered = [x-mean for x in offsets]
            t = [2e-7+2e-6*abs(x) for x in b]
            bound = [.025*x+y for x,y in zip(b,t)]
            maximum = max(abs(x) for x in centered)
            independent = b if maximum==0 else [w*(1+.25*x/maximum) for w,x in zip(b,centered)]
            for x,y,allowance in zip(r,independent,t):
                self.check(abs(x-y)<=allowance, 'FP32 reference allocation versus independent FP64 formula')
            for observed, expected in zip(pool['gate_b']['per_position_bound'],bound):
                self.close(observed, expected, 'coordinate gate bound')
            reference_alignment = math.fsum((x-y)*z for x,y,z in zip(r,b,centered))
            tolerance = math.fsum(x*abs(y) for x,y in zip(t,centered))
            required = .9*reference_alignment-tolerance
            self.close(pool['gate_c']['reference_alignment'], reference_alignment, 'reference alignment')
            self.close(pool['gate_c']['tolerance'], tolerance, 'alignment allowance')
            self.close(pool['gate_c']['required_alignment'], required, 'required alignment')
            for coefficients in pool['fast_coefficients_fp32']:
                self.check(all(abs(x-y)<=z for x,y,z in zip(coefficients,r,bound)), 'full-pool fidelity')
                alignment = math.fsum((x-y)*z for x,y,z in zip(coefficients,b,centered))
                self.check((alignment>=required) if maximum else
                           all(abs(x-y)<=z for x,y,z in zip(coefficients,b,t)), 'reference-centered alignment')
                self.check(all(.75*y-z<=x<=1.25*y+z for x,y,z in zip(coefficients,b,t)), 'relative coefficient cap')
                self.check(abs(math.fsum(coefficients)-math.fsum(b))<=2e-7+2e-6*abs(math.fsum(b)), 'pool beta budget')
            first, second = pool['fast_coefficients_fp32']
            self.equal(first,second,'exact repeated coefficients')
            active += maximum>0 and all(any(x!=y for x,y in zip(a,b)) for a in (first,second))
        self.check(active>0, 'nonconstant active complete pool')
        progress = self.load(base/'REFERENCE_PROGRESS.json')
        self.equal(len(progress['rows']), 28, 'all oracle contractions retained')
        for row in progress['rows']:
            self.close(row['reference_score_fp64'], -row['primary_dot_fp64']/(2*row['denominator']), 'oracle normalization exactly once')
        for repeat in gates['invariants']:
            self.equal(len(repeat),2,'two native invariant responses')
            for row in repeat:
                self.check(row['passed'] and all(row['checks'].values()) and not row['failures'], 'native invariant receipts')
        return dict(independently_recomputed='coefficient formula, gates, mass/caps and retained scalar contractions',
                    unavailable='full native gradients and directions; their dot products cannot be independently rebuilt')

    def readout(self, readout):
        if 'sets' in readout:
            self.equal(set(readout['sets']), {'new_buffer','parent_training'}, 'both TRAIN witness roles')
            return sum(self.readout(value) for value in readout['sets'].values())
        self.equal(readout['status'], 'passed', 'readout completed')
        self.check(readout['state_preserved'], 'readout state preservation receipt')
        self.equal((readout['optimizer_steps'],readout['backwards'],readout['teacher_forwards']),
                   (0,0,0), 'readout does no training or teacher work')
        response_values = {'SUM':[], 'MAX':[]}
        blocks_seen = 0
        for response in readout['responses']:
            summed, maximized = [], []
            count = sum(block['origin']['m']>0 for block in response['blocks'])
            self.equal(response['nonempty_blocks'],count,'nonempty response denominator')
            for block in response['blocks']:
                blocks_seen += 1
                positions, origin = block['positions'], block['origin']
                proposals, labels = block['proposals'], origin['labels']
                matching = 0
                for j in range(1,origin['m']+1):
                    if proposals[j]!=labels[j]:
                        break
                    matching += 1
                damage = sum(flag and proposals[j]!=labels[j] for j,flag in enumerate(origin['prefix']))
                error = origin['e']
                repaired = error>=1 and proposals[error]==labels[error]
                followers = list(range(error+1,origin['K']+1)) if error>=1 else []
                expected_metrics = dict(raw_repair=int(repaired),useful_repair=int(repaired and not damage),
                    damaged_prefix_positions=damage,damaged_prefix_blocks=int(damage>0),
                    surviving_followers=sum(proposals[j]==labels[j] for j in followers),
                    reachable_followers=sum(matching>=j for j in followers),
                    matching_prefix_length=matching,fixed_K_shortfall=max(0,origin['K']-matching))
                for key,value in expected_metrics.items():
                    self.equal(block['metrics'][key],value,'native block '+key)
                self.equal([p['position'] for p in positions],list(range(1,origin['K']+1)),'fixed-K valid positions')
                minima, multiplicities = [], Counter()
                best = None
                for index,position in enumerate(positions):
                    margin = position['margin']
                    self.check(math.isfinite(margin),'finite native margin')
                    if best is None or margin < positions[best]['margin']:
                        best = index
                    minima.append(best)
                    multiplicities[best] += 1
                for index,position in enumerate(positions):
                    best = minima[index]
                    margin = positions[best]['margin']
                    phi = (max(-margin,0)+math.log1p(math.exp(-abs(margin))))/math.log(2)
                    own = position['margin']
                    sum_term = (origin['K']-index)*(max(-own,0)+math.log1p(math.exp(-abs(own))))/math.log(2)
                    self.equal(position['bottleneck_position'],best+1,'earliest running bottleneck')
                    self.equal(position['bottleneck_multiplicity'],multiplicities[index],'fixed-branch multiplicity')
                    tied = sum(p['margin']==margin for p in positions[:index+1])>1
                    self.equal(bool(position['tied_bottleneck_prefix']),tied,'native margin tie diagnostic')
                    self.close(position['max_assessment_contribution_fp32'],phi,'MAX softplus scalar',atol=2e-6,rtol=2e-6)
                    self.close(position['assessment_contribution_fp32'],sum_term,'SUM triangular scalar',atol=2e-6,rtol=2e-6)
                self.equal(sum(multiplicities.values()),origin['K'],'multiplicities sum to K')
                actual_sum = math.fsum(p['assessment_contribution_fp32'] for p in positions)
                actual_max = math.fsum(p['max_assessment_contribution_fp32'] for p in positions)
                self.close(block['assessment_sum_fp64'],actual_sum,'SUM represented scalar reduction')
                self.close(block['max_assessment_sum_fp64'],actual_max,'MAX represented scalar reduction')
                shortfall = max(0,origin['K']-matching)
                allowance = 2e-6*(1+abs(actual_sum))
                self.check(shortfall<=actual_max+allowance and actual_max<=actual_sum+allowance,'fixed-K shortfall bound')
                summed.append(actual_sum)
                maximized.append(actual_max)
            for kind,values in (('SUM',summed),('MAX',maximized)):
                expected = math.fsum(values)/count if count else 0.
                self.close(response['assessments'][kind]['fp64'],expected,'response block normalization '+kind)
                response_values[kind].append(expected)
        for kind,values in response_values.items():
            self.close(readout['assessments'][kind]['fp64'],math.fsum(values)/len(values),'equal response average '+kind)
        return blocks_seen

    def readout_scalars(self):
        files = [Path('qualification/native')/('READOUT_'+name+'.json') for name in ('parent',*METHODS)]
        files += [Path('qualification/native')/('WITNESS_'+name+'.json') for name in ('parent','C')]
        files += [Path('arms')/arm/'WITNESS.json' for arm in ARMS]
        total = sum(self.readout(self.load(path)) for path in files)
        return dict(blocks=total,files=len(files),
            scalar_audit_tolerance='2e-6 absolute +2e-6 relative for stored FP32 softplus only; not a changed native gate',
            independently_recomputed='running minima, earliest ties, multiplicity, SUM/MAX terms, fixed-K bound and response averaging')

    def endpoint(self, arm):
        base = Path('arms')/arm
        state = self.load(base/'STATUS.json')
        self.equal(state['status'], 'completed', 'event complete '+arm)
        self.equal((state['global_step'],state['presentations'],state['completed_event']), (256,512,2), 'event endpoint')
        parent = self.load(base/'trainer/PARENT_RESTORATION.json')
        self.check(parent['exact_weights_optimizer_rng_counters_history'], 'warm own-state restoration')
        self.equal(parent['parent_validation']['parent_identity']['sha256'], PARENT_HASH, 'shared parent recovery')
        self.startup(state['startup'])
        final = self.load(base/'FINAL_STATE_CHECK.json')
        self.equal(final['status'], 'passed', 'final recovery validation')
        self.equal((final['global_step'],final['presentations'],final['completed_event']), (256,512,2), 'final state counters')
        self.equal(final['adam_state_entries'], 58, 'all warm Adam entries')
        self.check(final['immutable_parent_history'] and final['export_exact'], 'parent history and export equality')
        self.restoration(final['live'],256,512)
        training = state['training']
        self.equal((training['response_presentations'],training['optimizer_steps'],training['start_step'],training['end_step']),
                   (256,128,128,256), 'continuous event accounting')
        rows = training['records']
        self.equal(len(rows),256,'one observation per presentation')
        schedule = self.plan['retained_C_schedule']['records']
        if self.control_status is not None:
            control = self.load(self.control_status)
            self.equal(file_identity(self.control_status),self.plan['retained_C_status_identity'],'retained C receipt authentication')
            self.equal(control['training']['records'],schedule,'plan schedule equals authenticated C')
            self.equal((control['global_step'],control['presentations'],control['completed_event']),
                       (256,512,2),'retained C endpoint counters')
        for index,(row,reference) in enumerate(zip(rows,schedule)):
            for key in ('record_id','pass','sampled_anchors','optimizer_step_after','optimizer_stepped'):
                self.equal(row[key],reference[key],f'{arm} exact schedule {index} {key}')
            self.equal(row['optimizer_step_after'],128+(index+1)//2,'optimizer counter progression')
            self.equal(row['optimizer_stepped'],bool(index%2),'two-response accumulation')
            self.check(finite_values(row['response_observation']), 'finite training observation')
        self.equal(len(schedule),256,'retained schedule length')
        for pass_number in (0,1):
            ids = [row['record_id'] for row in rows if row['pass']==pass_number]
            self.equal(len(ids),128,'response presentations per pass')
            self.equal(len(set(ids)),128,'no duplicate response within a pass')
        reference = training['point3_reference_work']
        for key,n in dict(windows=128,response_forwards=256,response_backwards=256,adam_steps=0).items():
            self.equal(reference[key],n,'no-step reference total '+key)
        window_work = Counter()
        cost = dict(scoring_s=0., window_s=0.)
        directories = sorted((self.root/base/'trainer/events/event_002').glob('allocation_*'))
        self.equal([p.name for p in directories], [f'allocation_{i:03d}' for i in range(1,129)], 'per-window evidence coverage')
        for window,path in enumerate(directories,1):
            work = self.load(path/'WORK.json')
            self.work(work,arm+' window '+str(window))
            self.equal(work['native_adam_steps'],1,'one candidate Adam per window')
            window_work.update(work['completed'])
            result = self.load(path/'RESULT.json')
            score = self.load(path/'SCORING.json')
            self.equal(result['window'],window,'window identity')
            self.equal(score['coefficient_identity'],result['coefficient_identity'],'sealed coefficients reused')
            self.check(result['coefficients_sealed_before_commit'],'coefficients sealed before commit')
            self.equal(result['reference']['native_adam_steps'],0,'no hidden reference Adam')
            self.check(result['reference']['rng_restored'] and result['reference']['gradients_cleared'],'reference isolation')
            cost['scoring_s'] += result['score_s']
            cost['window_s'] += result['total_window_s']
        self.equal(window_work['point3_native_C_reference_forward'],256,'physical reference forwards')
        self.equal(window_work['point3_native_C_reference_backward'],256,'physical reference backwards')
        return dict(counters=[256,512,2],new_adam=128,presentations=256,
            costs=dict(**cost,wall_s=state['wall_s'],**training['costs']),
            physical_work=dict(window_work), export_identity=state['checkpoint_identity'],
            recovery_identity=state['recovery_identity'],
            tensor_boundary='runtime exact-state/export receipts checked, full tensors not reloaded offline')

    def costs_and_storage(self):
        clock = self.load('CLOCK.json')
        self.equal(clock['status'],'completed','complete campaign clock')
        self.equal(clock['historical_card_s'],HISTORICAL_CARD_S,'historical costs retained')
        phases = clock['phases']
        self.equal([(row['phase'],row['arm']) for row in phases],
                   [('qualification',None),('train','F_C_SUM'),('train','F_C_MAX'),('evaluate',None)], 'phase order')
        self.check(all(row['status']=='completed' and row['exit_code']==0 for row in phases),'all child phases completed')
        self.equal(clock['first_gpu_action_epoch'],phases[0]['started_epoch'],'original GPU clock retained')
        for first,second in zip(phases,phases[1:]):
            self.check(first['started_epoch']+first['wall_s']<=second['started_epoch'],
                       'serialized phases with no training/evaluation overlap')
        total = math.fsum(row['wall_s'] for row in phases)
        self.close(clock['completed_card_s'],total,'occupied-card total')
        self.close(clock['cumulative_card_s'],total+HISTORICAL_CARD_S,'historical+new cost')
        finish = phases[-1]['started_epoch']+phases[-1]['wall_s']
        elapsed = finish-clock['first_gpu_action_epoch']
        self.check(0<=total<=14400 and 0<=elapsed<=21600,'hard time caps')
        self.equal([row['card'] for row in phases],['B','B','B','A'],'physical card schedule')
        admission = self.plan['resource_admission']
        self.check(admission['all_locations_peak_bytes']<=25_000_000_000,'whole-study peak budget')
        for fs in admission['filesystems']:
            self.check(fs['free_bytes']-fs['remaining_peak_bytes']>=fs['reserve_bytes'],'per-filesystem peak reserve')
        boundary = self.load('evaluation/BOUNDARIES.jsonl',lines=True)
        self.equal(len(boundary),19,'all serving boundary observations')
        for row in boundary:
            current = row['resources']
            self.check(current['charged_bytes']<=25_000_000_000,'observed allocation cap')
            self.check(current['free_bytes']-current['remaining_peak_bytes']>=400_000_000_000,'observed original-disk reserve')
            self.close(current['charged_bytes'],current['artifact_bytes']+current['compiler_growth_bytes']+
                       self.plan['forecast']['storage']['local_allowance_bytes'],'cache and artifacts both charged',atol=.5,rtol=0)
            quiet = row['quiet']
            self.equal(quiet['status'],'passed','quiet-window receipt')
            self.check(not quiet['transfers'] and not quiet['training'] and not quiet['foreign_gpu_processes'],
                       'no concurrent transfer/training/foreign GPU worker')
            self.equal(quiet['physical_gpu_uuid'],self.plan['devices']['A'],'timed physical GPU')
            self.check(all(item['physical_gpu_uuid']==self.plan['devices']['A'] for item in quiet['active']),
                       'second physical GPU stays idle')
        return dict(occupied_card_s=total,elapsed_s=elapsed,cumulative_card_s=total+HISTORICAL_CARD_S,
            storage_evidence='sealed admission and nineteen serving boundary samples; no claim of a continuous independent disk trace')

    def generation_coverage(self):
        status = self.load('evaluation/STATUS.json')
        for key,n in dict(planned=1188,attempted=1188,returned=1188,completed_valid=1188,invalid=0,unattempted=0,unknown=0).items():
            self.equal(status[key],n,'serving status '+key)
        self.equal(status['status'],'completed','serving completion')
        self.equal(status['pending'],None,'serving pending slot')
        ledger = self.load('evaluation_work.json')
        self.equal(ledger['attempted'],dict(scored_generation=1152,warmup_generation=36),'generation attempts')
        self.equal(ledger['completed'],ledger['attempted'],'generation completions')
        self.equal(ledger['pending'],{},'unknown generation work')
        records = self.load('evaluation/MEASUREMENTS.jsonl',lines=True)
        references = self.load('evaluation/REFERENCES.json')
        self.equal(len(records),1188,'returned measurement records')
        grouped = defaultdict(list)
        for row in records:
            self.check(row['valid_measurement'] and not row['errors'],'every measurement valid')
            self.equal(row['physical_gpu_uuid'],self.plan['devices']['A'],'raw measurement physical GPU')
            reference = references['warmup' if row['phase']=='warmup' else 'panel'][row['request_id']]
            record = self.load(Path('evaluation/raw')/Path(row['raw_path']).name)
            self.raw_record(row,record,reference)
            grouped[row['activation']].append(row)
        self.equal(set(grouped),set(range(18)),'all eighteen activations')
        coverage, positions = Counter(), Counter()
        warmup_ids = set()
        group_ids = {}
        for activation,rows in grouped.items():
            repeat,group = divmod(activation//3,2)
            order = METHODS[(repeat+group)%3:]+METHODS[:(repeat+group)%3]
            method = order[activation%3]
            self.check(all((row['method'],row['repeat'],row['group'])==(method,repeat,group)
                           for row in rows),'balanced activation identity')
            warmups=[row for row in rows if row['phase']=='warmup']
            scored=[row for row in rows if row['phase']=='scored']
            self.equal((len(warmups),len(scored)),(2,64),'activation work')
            warmup_ids.update(row['request_id'] for row in warmups)
            ids=[row['request_id'] for row in scored]
            self.equal(len(set(ids)),64,'unique requests per activation')
            if group in group_ids:
                self.equal(ids,group_ids[group],'identical ordered group across repeats/arms')
            group_ids[group]=ids
            for row in scored:
                coverage[row['request_id'],method,repeat]+=1
                positions[row['request_id'],method,activation%3]+=1
        self.equal(len(warmup_ids),2,'only original two warmups')
        self.check(not set(group_ids[0])&set(group_ids[1]),'disjoint evaluation groups')
        self.check(not (set(group_ids[0])|set(group_ids[1]))&warmup_ids,'separate warmups')
        self.equal(len(coverage),1152,'request-method-repeat coverage')
        self.check(all(n==1 for n in coverage.values()) and all(n==1 for n in positions.values()),'one of every paired execution position')
        return dict(scored=1152,warmups=36,activations=18,
            exact_raw_token_termination_checks=1188,
            paired_CI_and_freeze='audited separately by independent_serving_audit.py')

    def postflight(self):
        receipt = self.load(self.root.parent/'REMOTE_POSTFLIGHT.json')
        self.equal(receipt['status'],'passed','independent CPU postflight receipt')
        self.equal(receipt['gpu_work'],0,'postflight performs no GPU work')
        self.check(not receipt['experimental_modules_imported'] and not receipt['weights_loaded_into_framework'],
                   'independent streaming-byte verification')
        self.equal(receipt['source']['commit'],self.plan['execution_commit'],'actual execution source')
        self.equal(receipt['source']['manifest_identity'],self.plan['source_identity'],'authenticated source manifest')
        self.equal(receipt['source']['files_verified'],5346,'all staged source entries')
        protected = receipt['protected_files']
        self.equal(set(protected),{'parent_recovery','parent_export','control_recovery','control_export',
            'F_C_SUM/checkpoint','F_C_SUM/recovery','F_C_MAX/checkpoint','F_C_MAX/recovery'},'all eight protected endpoints')
        self.check(all(row['matched_record'] for row in protected.values()),'all protected file hashes match')
        self.equal(protected['parent_recovery']['identity'],self.plan['parent_recovery_identity'],'parent recovery hash')
        for arm in ARMS:
            status = self.load(Path('arms')/arm/'STATUS.json')
            self.equal(protected[arm+'/checkpoint']['identity'],status['checkpoint_identity'],'published export hash '+arm)
            self.equal(protected[arm+'/recovery']['identity'],status['recovery_identity'],'published recovery hash '+arm)
        summary = self.load('evaluation/SUMMARY.json')
        for method in METHODS:
            actual = receipt['tensor_content_identities'][method]
            self.equal(actual['sha256'],summary['parameter_identities'][method]['sha256'],
                       'independent tensor-content identity '+method)
            self.equal(actual['tensors'],58,'serving parameter tensors '+method)
        self.equal(len({value['sha256'] for value in receipt['tensor_content_identities'].values()}),3,
                   'three distinct serving models')
        resource = receipt['resource']
        self.equal(resource['charged_remote_bytes'],resource['remote_artifact_allocated_bytes']+
                   resource['compiler_growth_bytes'],'remote artifact/cache accounting')
        self.equal(resource['reserve_bytes'],400_000_000_000,'postflight reserve')
        self.check(resource['free_bytes']>=resource['reserve_bytes'],'postflight reserve retained')
        self.equal(receipt['gpu_processes'].strip(),'','both GPUs idle after completion')
        final = self.load(self.root.parent/'RESOURCE_FINAL.json')
        self.equal(final['charged_remote_bytes'],resource['charged_remote_bytes'],'final remote charge')
        self.equal(final['charged_all_locations_bytes'],final['charged_remote_bytes']+
                   final['actual_local_allocated_bytes']+final['new_shared_git_object_allocated_bytes'],
                   'measured all-location charge including shared Git growth')
        self.check(final['charged_all_locations_bytes']<=25_000_000_000,'actual all-location storage cap')
        self.equal(final['conservative_postflight_peak_bytes'],final['charged_all_locations_bytes']+
                   final['publication_growth_allowance_bytes'],'prospective publication overlap')
        self.check(final['conservative_postflight_peak_bytes']<=25_000_000_000,'publication storage cap')
        self.check(final['free_bytes']>=final['reserve_bytes']==400_000_000_000,'final original-disk reserve')
        self.check(final['fenrir_free_bytes']>=final['fenrir_reserve_bytes'],'final Fenrir reserve')
        self.equal(final['local_allowance_exceeded'],final['actual_local_allocated_bytes']>
                   final['original_local_delivery_allowance_bytes'],'local forecast overrun disclosed')
        return dict(source_files=5346,protected_file_hashes=8,distinct_parameter_identities=3,
            remote_allocated_bytes=resource['charged_remote_bytes'],
            actual_local_allocated_bytes=final['actual_local_allocated_bytes'],
            new_shared_git_object_allocated_bytes=final['new_shared_git_object_allocated_bytes'],
            charged_all_locations_bytes=final['charged_all_locations_bytes'],
            publication_peak_bytes=final['conservative_postflight_peak_bytes'],
            local_allowance_exceeded=final['local_allowance_exceeded'],
            scope='cross-check of independent CPU postflight receipts; auditor does not load full tensors')

    def raw_record(self, row, record, reference):
        self.check(reference['verified'] and reference['reference_kind']=='frozen_DFlash','authenticated F canonical type')
        for key in ('prompt_ids','response_ids','finish_reason','stop_token_ids','max_new_tokens'):
            self.equal(record[key],reference[key],'raw canonical '+key)
        self.equal(record['checkpoint_id'],row['checkpoint_id'],'raw checkpoint label')
        self.equal(len(record['response_ids']),row['output_tokens'],'raw output length')
        self.equal(len(record['prompt_ids']),row['prompt_tokens'],'raw prompt length')
        self.equal(record['max_new_tokens'],2048,'raw output cap')
        metadata = record['metadata']
        self.equal((metadata['block_size'],metadata['context_limit']),(16,4096),'native serving shape')
        self.check(not metadata['feedback_capture_in_timed_request'],'no timed feedback collection')
        self.equal(metadata['runtime'],'pinned_SGLang_FA3','qualified serving runtime')
        self.equal(metadata['cache_policy'],'idle_full_flush_before_every_request_and_publication','cold cache policy')
        traces = metadata['trace']
        self.equal(len(traces),row['verification_calls'],'trace verification calls')
        start = 0
        for trace in traces:
            self.equal(trace['output_start'],start,'absolute trace progress')
            self.equal(trace['anchor_token'],record['response_ids'][start],'canonical block anchor')
            progress = trace['committed_progress']
            self.equal(trace['committed_tokens'],record['response_ids'][start+1:start+progress+1],
                       'committed canonical verification suffix')
            self.equal(len(trace['committed_tokens']),progress,'trace progress length')
            accepted = 0
            for draft, verified in zip(trace['draft_tokens'],trace['target_argmax']):
                if draft != verified:
                    break
                accepted += 1
            self.equal(trace['raw_accepted_draft_tokens_before_stop'],accepted,
                       'native proposal agreement before stop clipping')
            accepted_path = trace['draft_tokens'][:accepted]+trace['target_argmax'][accepted:accepted+1]
            self.equal(trace['committed_tokens'],accepted_path[:progress],
                       'verified draft prefix and target replacement/bonus')
            start += progress
        self.equal(start,len(record['response_ids'])-1,'initial prefill token excluded exactly once')
        for key in ('accepted_draft_tokens','native_accepted_draft_tokens','committed_progress'):
            self.equal(sum(trace[key] for trace in traces),row[key],'trace-derived '+key)
        self.equal(metadata['verification_calls'],row['verification_calls'],'metadata call count')
        for key in ('request_latency_s','native_generate_s'):
            self.close(metadata[key],row[key],'raw measured '+key)
        self.close(metadata['capture']['cache_maintenance_s'],row['cache_maintenance_s'],'raw cache time')

    def finish(self):
        for name,function in [('retrieval',self.retrieval),('contract',self.contract),('qualification',self.qualification),
            ('numerical_gates',self.numerical_gates),('work_ledger',self.work_ledger),
            *[(arm,lambda arm=arm:self.endpoint(arm)) for arm in ARMS],
            ('readout_scalars',self.readout_scalars),
            ('costs_and_storage',self.costs_and_storage),('generation_coverage',self.generation_coverage),
            ('postflight',self.postflight)]:
            self.section(name,function)
        states = [row['status'] for row in self.sections.values()]
        status = 'failed' if 'failed' in states else 'incomplete' if 'incomplete' in states else 'passed'
        return dict(schema='point3_independent_receipt_audit_v1',status=status,
            auditor_identity=file_identity(__file__),checks=self.checked,sections=self.sections,
            input_identities=self.inputs,
            limitations=['No GPU operations or imported production analysis helpers',
                'No independent model-sized gradient, optimizer-state or parameter reconstruction',
                'Native tensor equality remains authenticated runtime receipt evidence',
                'No serving-speed or mechanism claim follows from qualification alone'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--historical-root')
    parser.add_argument('--control-status')
    args=parser.parse_args()
    audit=Audit(args.root,args.historical_root,args.control_status)
    result=audit.finish()
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],checks=result['checks'],
                         sections={k:v['status'] for k,v in result['sections'].items()})))
    raise SystemExit(0 if result['status']=='passed' else 2)


if __name__=='__main__':
    main()
