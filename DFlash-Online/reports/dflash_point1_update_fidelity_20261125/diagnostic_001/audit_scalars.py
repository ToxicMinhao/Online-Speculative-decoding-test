"""Independent stdlib-only audit of retained Point1 scalar and token receipts.

Does not import experimental code, run a model, or reconstruct unsaved vectors.
"""

import argparse
import csv
import hashlib
import json
import math
import datetime
from pathlib import Path


METRICS = ('raw_repair', 'useful_repair', 'damaged_prefix_positions',
           'damaged_prefix_blocks', 'surviving_followers', 'reachable_followers',
           'matching_prefix_length', 'fixed_K_shortfall')
LOWER = {'damaged_prefix_positions', 'damaged_prefix_blocks', 'fixed_K_shortfall'}
EXPECTED_COMMIT = '269f05644eeed2f0396602c47beb2a71166f8dd0'
CASES = ('BETA', 'C', 'F', 'C_REPEAT')
HISTORICAL_CARD_S = 5175.047027264


class Audit:
    def __init__(self, root, exported):
        self.root, self.exported = Path(root), Path(exported)
        self.checks = 0
        self.identities = {}
        self.readouts = {}
        self.score_summaries = {}
        self.pair_summaries = {}
        self.receipt_summary = {}

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            raise AssertionError(label)

    def equal(self, actual, expected, label):
        self.check(actual == expected, label)

    def close(self, actual, expected, label):
        self.check(math.isfinite(actual) and math.isfinite(expected)
                   and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12), label)

    def read(self, path):
        data = Path(path).read_bytes()
        key = str(Path(path).relative_to(self.root)) if Path(path).is_relative_to(self.root) else 'public/' + Path(path).name
        self.identities[key] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        return json.loads(data)

    def identity(self, path):
        data = Path(path).read_bytes()
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def endpoint(self, path):
        row = self.read(path)
        self.equal(row['status'], 'passed', 'endpoint status')
        self.check(row['state_preserved'], 'endpoint state preservation')
        for response in row['responses']:
            blocks = response['blocks']
            self.equal(len(blocks), len(response['anchors']), 'one block per retained anchor')
            self.equal(len(set(response['anchors'])), len(response['anchors']), 'unique within-response anchors')
            nonempty = sum(block['origin']['m'] > 0 for block in blocks)
            self.equal(nonempty, response['nonempty_blocks'], 'response denominator')
            for block in blocks:
                origin, proposed = block['origin'], block['proposals']
                labels, valid = origin['labels'], origin['valid']
                self.equal(len(proposed), 16, 'native sixteen-slot block')
                self.equal(valid, [False] + [True] * origin['m'] + [False] * (15-origin['m']), 'contiguous valid support')
                self.equal(block['current']['labels'], labels, 'current labels unchanged')
                self.equal(block['current']['valid'], valid, 'current valid support unchanged')
                prefix = 0
                for i in range(1, len(proposed)):
                    if not valid[i] or proposed[i] != labels[i]:
                        break
                    prefix += 1
                error = origin['e']
                damage = sum(old and proposed[i] != labels[i]
                             for i, old in enumerate(origin['prefix']))
                repaired = error > 0 and proposed[error] == labels[error]
                followers = range(error + 1, origin['K'] + 1) if error > 0 else ()
                expected = dict(raw_repair=int(repaired), useful_repair=int(repaired and not damage),
                    damaged_prefix_positions=damage, damaged_prefix_blocks=int(damage > 0),
                    surviving_followers=sum(proposed[i] == labels[i] for i in followers),
                    reachable_followers=sum(prefix >= i for i in followers),
                    matching_prefix_length=prefix, fixed_K_shortfall=max(origin['K'] - prefix, 0))
                for metric, value in expected.items():
                    self.equal(block['metrics'][metric], value, 'native proposal metric ' + metric)
                self.equal(block['current']['L'], prefix, 'current prefix length')
                current_error = prefix + 1 if prefix < origin['m'] else -1
                current_K = origin['m']
                if current_error > 0:
                    for i in range(current_error + 1, origin['m'] + 1):
                        if proposed[i] != labels[i]:
                            current_K = i - 1
                            break
                self.equal(block['current']['e'], current_error, 'current first error')
                self.equal(block['current']['K'], current_K, 'current usable endpoint')
                self.equal(block['current']['H'], current_K-prefix if current_error > 0 else 0, 'current usable horizon')
                self.equal(block['current']['m'], origin['m'], 'current valid length')
                self.equal(block['current']['prefix'], [False] + [i <= prefix for i in range(1,16)], 'current prefix mask')
                self.equal(block['current']['repair'], [i == current_error for i in range(16)], 'current repair mask')
                if row['endpoint'] == 'parent':
                    self.equal(origin,block['current'],'parent reproduces frozen origin roles')
                self.equal([p['position'] for p in block['positions']], list(range(1,origin['K']+1)), 'all frozen assessment positions')
                terms = []
                for position in block['positions']:
                    i, margin = position['position'], position['margin']
                    self.equal(position['canonical'], labels[i], 'position canonical binding')
                    self.equal(position['prediction'], proposed[i], 'position prediction binding')
                    self.equal(position['correct'], proposed[i] == labels[i], 'position correctness')
                    self.equal(position['margin_tie'], margin == 0, 'margin tie flag')
                    if margin > 0:
                        self.equal(proposed[i], labels[i], 'strict positive margin correctness')
                    elif margin < 0:
                        self.check(proposed[i] != labels[i], 'strict negative margin error')
                    else:
                        self.equal(proposed[i], min(labels[i], position['rival']), 'deterministic tied argmax')
                    value = position['assessment_contribution_fp32']
                    self.check(math.isfinite(value) and value >= 0, 'finite nonnegative assessment term')
                    terms.append(value)
                self.close(block['assessment_sum_fp64'], math.fsum(terms), 'block FP64 contribution sum')
            total = math.fsum(block['assessment_sum_fp64'] for block in blocks)
            self.close(response['response_assessment_fp64'], total / nonempty if nonempty else 0,
                       'response FP64 reduction')
        self.close(row['mean_response_assessment_fp64'],
                   math.fsum(r['response_assessment_fp64'] for r in row['responses']) / len(row['responses']),
                   'window FP64 mean')
        return row

    def paired(self, first, second, published):
        self.equal(len(first['responses']), len(second['responses']), 'paired response count')
        metrics = (*METRICS, 'assessment')
        benefits, terms, metric_deltas = [], [], {key: [] for key in metrics}
        response_deltas = {key: [] for key in metrics}
        block_cursor = 0
        count = len(first['responses'])
        for ordinal, (a, b) in enumerate(zip(first['responses'], second['responses'], strict=True)):
            self.equal(a['record_id'], b['record_id'], 'paired response identity')
            self.equal(a['anchors'], b['anchors'], 'paired anchors')
            self.equal(a['nonempty_blocks'], b['nonempty_blocks'], 'paired denominator')
            denominator = a['nonempty_blocks']
            local = []
            local_deltas = {key: [] for key in metrics}
            for x, y in zip(a['blocks'], b['blocks'], strict=True):
                self.equal(x['origin'], y['origin'], 'paired frozen origins')
                for left, right in zip(x['positions'], y['positions'], strict=True):
                    self.equal(left['position'], right['position'], 'paired position')
                    value = left['assessment_contribution_fp32'] - right['assessment_contribution_fp32']
                    local.append(value)
                    if denominator:
                        terms.append(value / denominator / count)
                for metric in metrics:
                    delta = y['metrics'][metric] - x['metrics'][metric]
                    metric_deltas[metric].append(delta)
                    local_deltas[metric].append(delta)
                    self.close(published['blocks'][block_cursor]['metric_changes'][metric], delta, 'published block delta')
                self.close(published['blocks'][block_cursor]['assessment_benefit_fp64'], math.fsum(
                    p['assessment_contribution_fp32'] - q['assessment_contribution_fp32']
                    for p,q in zip(x['positions'],y['positions'],strict=True)), 'published block benefit')
                block_cursor += 1
            benefits.append(math.fsum(local) / denominator if denominator else 0)
            self.close(published['responses'][ordinal]['assessment_benefit_fp64'], benefits[-1], 'published response benefit')
            for metric in metrics:
                delta = math.fsum(local_deltas[metric]) / (denominator or 1) if metric == 'assessment' else math.fsum(local_deltas[metric])
                response_deltas[metric].append(delta)
                self.close(published['responses'][ordinal]['metric_changes'][metric], delta, 'published response delta')
        benefit = math.fsum(benefits) / count
        self.close(published['assessment_benefit_fp64'], benefit, 'paired FP64 benefit')
        self.close(published['positive_contribution_sum_fp64'], math.fsum(x for x in terms if x > 0),
                   'positive matched contributions')
        self.close(published['negative_contribution_sum_fp64'], math.fsum(x for x in terms if x < 0),
                   'negative matched contributions')
        self.close(published['assessment_benefit_original_fp32'], first['mean_response_assessment_fp32']-second['mean_response_assessment_fp32'], 'original FP32 assessment subtraction')
        for key, predicate in [('positive',lambda x:x>0),('negative',lambda x:x<0),('zero',lambda x:x==0)]:
            self.equal(published[key+'_contributions'],sum(predicate(x) for x in terms),'contribution sign count')
        self.equal(published['assessed_occurrences'],len(terms),'matched contribution count')
        for metric, values in metric_deltas.items():
            sign = -1 if metric in LOWER or metric == 'assessment' else 1
            actual = published['block_metrics'][metric]
            for field, expected in dict(delta=math.fsum(values), gains=sum(sign*x > 0 for x in values),
                                       losses=sum(sign*x < 0 for x in values), ties=sum(x == 0 for x in values)).items():
                self.equal(actual[field], expected, metric + ' paired ' + field)
            values = response_deltas[metric]
            for field,expected in dict(delta=math.fsum(values),gains=sum(sign*x>0 for x in values),losses=sum(sign*x<0 for x in values),ties=sum(x==0 for x in values)).items():
                self.equal(published['response_metrics'][metric][field],expected,metric+' response '+field)
        self.pair_summaries[first['endpoint']+' to '+second['endpoint']+f' ({count} responses)'] = dict(
            assessment_benefit=benefit, positive=published['positive_contribution_sum_fp64'],
            negative=published['negative_contribution_sum_fp64'], per_response_benefits=benefits,
            block_metrics=published['block_metrics'], response_metrics=published['response_metrics'])
        return benefit

    def vectors(self, row):
        a, b, difference = row['reference_norm'], row['candidate_norm'], row['difference_norm']
        self.check(all(math.isfinite(v) and v >= 0 for v in (a, b, difference)), 'vector norms finite')
        self.check(0 <= row['differing_coordinates'] <= row['coordinates'], 'changed coordinates bounded')
        self.equal(row['zero_status'], 'both_zero' if a == b == 0 else 'reference_zero' if a == 0 else 'candidate_zero' if b == 0 else 'both_nonzero', 'zero case classification')
        # Squared norm identity can lose accuracy by cancellation for endpoint
        # vectors. Test on the summand scale, never floor a scientific ratio.
        residual = difference*difference-(a*a+b*b-2*row['dot'])
        self.check(abs(residual) <= 2e-12 * max(a*a+b*b+2*abs(row['dot']),difference*difference,1), 'vector norm/dot identity at summand scale')
        if a and b:
            self.close(row['cosine'], row['dot'] / (a*b), 'cosine arithmetic')
            self.check(abs(row['cosine']) <= 1 + 2e-12, 'cosine bound')
        else:
            self.equal(row['cosine'], None, 'undefined zero-vector cosine')
        self.equal(row['relative_difference'], None, 'zero-reference ratio') if not a else self.close(
            row['relative_difference'], difference/a, 'relative vector difference')
        layers = row.get('per_layer', {})
        if layers:
            for layer in layers.values():
                self.vectors(layer)
            self.close(row['dot'],math.fsum(v['dot'] for v in layers.values()),'layer dot sum')
            for field in ('reference_norm', 'candidate_norm', 'difference_norm'):
                self.close(row[field], math.sqrt(math.fsum(v[field]**2 for v in layers.values())),
                           'layer aggregation ' + field)
            self.equal(row['coordinates'], sum(v['coordinates'] for v in layers.values()), 'layer coordinate count')
            self.equal(row['differing_coordinates'], sum(v['differing_coordinates'] for v in layers.values()),
                       'layer changed coordinate count')
            for field in ('reference_signed_projection', 'candidate_signed_projection', 'difference_signed_projection'):
                if field in row:
                    self.close(row[field], math.fsum(v[field] for v in layers.values()), 'layer ' + field)
        if 'reference_signed_projection' in row:
            self.close(row['difference_signed_projection'],
                       row['candidate_signed_projection'] - row['reference_signed_projection'],
                       'signed projection subtraction')

    def scores(self, scoring):
        def flatten(value):
            return [leaf for item in value for leaf in (flatten(item) if isinstance(item,list) else [item])]
        self.check(scoring['frozen_before_native_updates'],'score coefficients frozen before updates')
        self.equal(set(scoring['predictions']), {'F-C','F-BETA'}, 'two prespecified score contrasts')
        for contrast, responses in scoring['predictions'].items():
            self.equal(len(responses),2,'two score response reductions')
            for row in responses:
                pools = {key:flatten(value) for key,value in row['per_pool'].items()}
                self.equal(len({len(v) for v in pools.values()}),1,'score per-pool array lengths')
                for key in ('raw','beta_centered'):
                    scalar = row[key]
                    self.close(scalar['total'],math.fsum(pools[key]),'score per-pool sum')
                    self.close(scalar['total'],scalar['positive']+scalar['negative'],'score signed cancellation')
                    self.close(scalar['absolute'],scalar['positive']-scalar['negative'],'score absolute cancellation')
                    self.check(scalar['positive']>=0 and scalar['negative']<=0,'score sign decomposition')
                self.close(row['raw_minus_centered'],row['raw']['total']-row['beta_centered']['total'],'score centering subtraction')
                self.close(row['mass_offset_contribution'], math.fsum(x*y for x,y in zip(pools['mass_residual'],pools['weighted_mean_score'],strict=True)), 'mass offset sum')
                self.close(row['raw_minus_centered'],row['mass_offset_contribution'],'raw-centered mass-offset identity')
                self.close(row['max_absolute_pool_mass_residual'],max(abs(x) for x in pools['mass_residual']),'max mass residual')
                self.check(0<=row['nonempty_pools']<=len(pools['raw']),'pool coverage bound')
            self.score_summaries[contrast] = {
                key:math.fsum(r[field][subfield] for r in responses)
                for key,field,subfield in [('raw','raw','total'),('beta_centered','beta_centered','total'),
                    ('raw_positive','raw','positive'),('raw_negative','raw','negative'),
                    ('centered_positive','beta_centered','positive'),('centered_negative','beta_centered','negative')]}

    def receipts(self, native, result, cases, ledger, work):
        contract = self.read(self.root/'CONTRACT.json')
        bound = self.identity(self.root/'CONTRACT.json')
        self.equal(contract['execution_commit'], EXPECTED_COMMIT, 'pinned execution source')
        study=self.root.parent.parent
        source_manifest=study/'SOURCE.json'
        source_root=study/'implementation_checkout'/'DFlash-Online'
        source=json.loads(source_manifest.read_text())
        self.equal(self.identity(source_manifest),contract['source_identity'],'local source manifest contract binding')
        self.equal(source['commit'],EXPECTED_COMMIT,'local source manifest execution commit')
        for filename,expected in source['files'].items():
            path=source_root/filename
            if expected.get('kind')=='symlink':
                self.check(path.is_symlink(),'source link remains symlink: '+filename)
                self.equal(str(path.readlink()),expected['target'],'source symlink target: '+filename)
            else:
                self.equal(self.identity(path),expected,'source file bytes: '+filename)
        self.identities['source_manifest']=self.identity(source_manifest)
        self.equal(result['execution_commit'], EXPECTED_COMMIT, 'native execution source')
        self.equal(result['contract_identity'],bound,'native contract binding')
        self.equal(ledger['contract_identity'],bound,'ledger contract binding')
        self.equal(contract['cases'],list(CASES),'prespecified cases')
        self.equal(contract['parent_counters'],[128,256,1],'parent counters')
        self.equal(contract['case_counters'],[129,258,1],'case counters')
        self.equal(contract['old_hidden_gate_status'],'failed_retained_diagnostic','historical failure preserved')
        self.equal(contract['parent_binding'],'b120c76b0036b4786eb6b9d11b1594ec1dc6e57e3d20c9ed931924262b1c072a','parent plan binding')
        self.equal(contract['parent_recovery_identity'],dict(bytes=6296473051,sha256='728770a7abf0ba3ef96805ef760997976ad7bedbbee1d8debe165e785a4d98a2'),'parent recovery bound')
        self.equal(contract['parent_export_identity'],dict(bytes=2097259104,sha256='1556409004f7c73c0007c1f01764d4fd1dd189827994596939597be43b2d21b2'),'parent export bound')
        self.equal(contract['original_window_identity'],dict(bytes=25388,sha256='a4245d34875ecc349e04cf34de24ef1d626d9cbd03ee30176f637d189d431d3d'),'qualification window bound')
        for key in ('new_feedback','generations','exports'):
            self.equal(result[key],0,'zero '+key)
        for key in ('full_events','generation_calls','serial_token_oracle_calls'):
            self.equal(contract[key],0,'zero contracted '+key)
        self.equal(result['work'],work,'result work receipt identity')
        for name in CASES:
            self.equal(cases[name]['adam_attempts'],1,'one attempted Adam per case')
            self.equal(cases[name]['adam_completed'],1,'one completed Adam per case')
            for suffix in ('INITIAL_STATE','RESTORATION'):
                state = self.read(native/(name+'_'+suffix+'.json'))
                self.check(state['exact_resume'],'exact case state restoration')
                self.equal(state.get('mismatches',[]),[],'no state mismatches')
                self.equal(state['global_step'],128,'restored parent step')
                self.equal(state['presentations'],256,'restored parent presentations')
        for filename in ('AFTER_STARTUP.json','FINAL_RESTORATION.json'):
            state=self.read(native/filename)
            self.check(state['exact_resume'],'exact startup/final restoration')
            self.equal(state.get('mismatches',[]),[],'no startup/final mismatches')
        self.equal(result['final_restoration'],self.read(native/'FINAL_RESTORATION.json'),'final restoration receipt equality')
        self.equal(ledger['maximum_attempted_calls'],6,'six-call cap')
        self.equal(ledger['maximum_attempts'],2,'one correction retry cap')
        self.check(1<=len(ledger['attempts'])<=2,'bounded launch attempts')
        self.equal([c['number'] for c in ledger['calls']],list(range(1,len(ledger['calls'])+1)),'call reservations contiguous')
        for attempt in ledger['attempts']:
            if attempt['number']==2:
                self.check(bool(attempt['correction_note']),'identified correction note')
            calls=[c for c in ledger['calls'] if c['attempt']==attempt['number']]
            self.equal(len(set(c['case'] for c in calls)),len(calls),'unique cases in each attempt')
            if attempt['status']=='completed':
                self.equal([c['case'] for c in calls],list(CASES),'completed case ordering')
                self.check(all(c['status']=='completed' for c in calls),'all completed calls reserved')
            for call in calls:
                if call['status']=='completed':
                    self.check(call['reserved_epoch']<=call['completed_epoch'],'reservation precedes completion')
        clock=self.read(self.root/'CLOCK.json')
        process=self.read(native.parent/'PROCESS.json')
        admission=self.read(native.parent/'ADMISSION.json')
        launch=self.read(native.parent/'LAUNCH.json')
        self.equal(launch['physical_gpu_uuid'],contract['devices']['B'],'physical B only')
        self.equal(process['status'],'completed','supervisor completion')
        self.equal(process['exit_code'],0,'native child exit code')
        self.equal(clock['contract_identity'],bound,'clock contract identity')
        self.equal(clock['active_attempt'],None,'no active attempt')
        self.close(clock['completed_occupied_card_s'],math.fsum(a['occupied_card_s'] for a in clock['attempts']),'clock occupied sum')
        self.close(process['new_phase_occupied_card_s'],clock['completed_occupied_card_s'],'process/clock card accounting')
        self.close(process['historical_occupied_card_s'],HISTORICAL_CARD_S,'prior cost unchanged')
        self.close(process['historical_plus_new_occupied_card_s'],HISTORICAL_CARD_S+clock['completed_occupied_card_s'],'historical-plus-new accounting')
        self.check(process['new_phase_elapsed_s']<=3600 and process['new_phase_occupied_card_s']<=3600,'hour/card-hour cap')
        self.check(admission['admitted'] and all(admission['checks'].values()),'live admission passed')
        self.check(admission['free_bytes']-admission['remaining_write_bytes']>=400_000_000_000,'forecast reserve')
        self.check(admission['charged_bytes']+admission['remaining_write_bytes']<=2_000_000_000,'forecast disk cap')
        self.check(process['all_locations_charged_bytes']<=2_000_000_000,'final charged disk cap')
        self.check(process['free_bytes']>=400_000_000_000,'final free-space reserve')
        samples_path=native.parent/'RESOURCE_SAMPLES.jsonl'
        samples=[json.loads(line) for line in samples_path.read_text().splitlines() if line]
        self.identities[str(samples_path.relative_to(self.root))]=self.identity(samples_path)
        for sample in samples:
            self.check(sample['all_locations_charged_bytes']<=2_000_000_000,'sampled disk cap')
            self.check(sample['free_bytes']>=400_000_000_000,'sampled free-space reserve')
        self.receipt_summary=dict(attempts=len(ledger['attempts']),adam_reserved=len(ledger['calls']),
            adam_completed=sum(c['status']=='completed' for c in ledger['calls']),
            elapsed_s=process['new_phase_elapsed_s'],occupied_card_s=process['new_phase_occupied_card_s'],
            historical_plus_new_occupied_card_s=process['historical_plus_new_occupied_card_s'],
            final_charged_bytes=process['all_locations_charged_bytes'],final_free_bytes=process['free_bytes'],
            resource_samples=len(samples),witness_status=contract['witness']['status'])
        self.receipt_summary['source_files_rehashed']=len(source['files'])
        postflight=self.read(self.root/'POSTFLIGHT_PRIVATE.json')
        self.equal(postflight['status'],'passed','postflight passed')
        self.equal(postflight['execution_commit'],EXPECTED_COMMIT,'postflight source pin')
        self.equal(postflight['source_manifest_identity'],contract['source_identity'],'postflight source manifest')
        for row in postflight['files']:
            expected={key:row[key] for key in ('bytes','sha256')}
            self.equal(self.identity(self.root/row['path']),expected,'postflight raw receipt file: '+row['path'])
            self.identities[row['path']]=expected
        for row in postflight['checks']:
            self.check(row['passed'],'postflight input check')
            self.equal(row['actual'],row['expected'],'postflight unchanged bound input')
        self.equal(postflight['gpu']['processes'],'','postflight both GPUs idle receipt')
        self.check(postflight['no_checkpoint_or_full_vector_files'],'no prohibited model/vector exports')
        self.equal(postflight['inherited_deletions'],[],'no inherited deletion')
        self.check(postflight['storage']['all_locations_charged_bytes']<=2_000_000_000,'postflight disk cap')
        self.check(postflight['storage']['free_bytes']>=400_000_000_000,'postflight reserve')
        self.receipt_summary.update(postflight_files_rehashed=len(postflight['files']),
            postflight_bytes_rehashed=sum(r['bytes'] for r in postflight['files']),
            postflight_input_checks=len(postflight['checks']),
            postflight_charged_bytes=postflight['storage']['all_locations_charged_bytes'],
            postflight_free_bytes=postflight['storage']['free_bytes'])
        return contract

    def run(self, attempt):
        native = self.root / f'attempt_{attempt:03d}' / 'native'
        result = self.read(native/'RESULT.json')
        self.equal(result['status'], 'completed', 'native completion')
        self.equal(result['planned_cases_completed'], 4, 'four planned cases')
        cases = {name: self.read(native/(name+'.json')) for name in ('BETA', 'C', 'F', 'C_REPEAT')}
        for case in cases.values():
            self.equal(case['status'], 'completed', 'case completion')
            self.equal(case['endpoint']['global_step'], 129, 'disposable step')
            self.equal(case['endpoint']['presentations'], 258, 'disposable presentations')
            self.check(case['restoration']['exact_resume'], 'exact restoration')
        self.check(cases['BETA']['beta_g0_exact'], 'beta g0 exact')
        self.check(cases['C_REPEAT']['exact_C_repeat'], 'C exact repeat receipt')
        self.equal(cases['C']['endpoint'], cases['C_REPEAT']['endpoint'], 'C endpoint repeat')
        self.equal(cases['C']['gradient_identity'], cases['C_REPEAT']['gradient_identity'], 'C gradient repeat')
        ledger = self.read(self.root/'STEP_LEDGER.json')
        self.check(len(ledger['calls']) <= 6, 'bounded attempted calls')
        work = self.read(native/'WORK.json')
        self.equal(work['native_adam_steps'], 4, 'four completed native Adam calls')
        self.equal(work['pending'], [], 'no pending primitive work')
        contract = self.receipts(native,result,cases,ledger,work)
        scoring = self.read(native/'SCORING.json')
        self.scores(scoring)
        self.equal(scoring['reference_gradient'],cases['BETA']['gradient_identity'],'reference/BETA gradient identity')
        identifiers=self.read(self.exported/'IDENTITIES.json')
        self.equal(identifiers['execution_commit'],EXPECTED_COMMIT,'public execution binding')
        for filename, expected in identifiers['raw_input_receipts'].items():
            self.equal(self.identity(self.root/filename),expected,'export input byte binding: '+filename)
        manifest=self.read(self.exported/'PUBLIC_FILES.json')
        for filename,expected in manifest['files'].items():
            self.equal(self.identity(self.exported/filename),expected,'public export byte binding: '+filename)
        comparisons = self.read(self.exported/'READOUT_COMPARISONS.json')
        benefits = {}
        for scope, prefix in (('window', 'window'), ('new_buffer', 'witness')):
            endpoints = ('parent', 'Q_BETA', 'Q_C', 'Q_F', 'N_BETA', 'N_C', 'N_F')
            rows = {endpoint: self.endpoint(native/f'{prefix}_{endpoint}.json')
                    for endpoint in endpoints if (native/f'{prefix}_{endpoint}.json').exists()}
            self.readouts[scope] = rows
            expected_endpoints = set(endpoints) if scope == 'window' else {'parent','N_BETA','N_C','N_F'}
            if scope == 'new_buffer' and contract['witness']['status'] != 'authenticated':
                expected_endpoints=set()
            self.equal(set(rows),expected_endpoints,'complete endpoint coverage')
            benefits[scope] = {}
            for key, published in comparisons[scope].items():
                second, first = key.split('-')
                benefits[scope][key] = self.paired(rows[first], rows[second], published)
            for endpoint,row in rows.items():
                self.equal(len(row['responses']),2 if scope=='window' else 8,'response coverage')
                self.equal([len(r['blocks']) for r in row['responses']],[512,269] if scope=='window' else [len(i['anchors']) for i in contract['witness']['items']],'retained block coverage')
                self.equal(row['optimizer_steps'],0,'readout makes zero Adam calls')
                self.equal(row['backwards'],0,'readout makes zero backward calls')
                self.equal(row['teacher_forwards'],0,'readout makes zero teacher calls')
                self.equal(row['live_counters'],[128,256,1],'functional readout at restored parent boundary')
        if self.readouts['new_buffer']:
            old=self.readouts['window']['parent']['responses']
            new=self.readouts['new_buffer']['parent']['responses']
            self.check(not {r['record_id'] for r in old} & {r['record_id'] for r in new},'window/witness IDs disjoint')
            self.check(not {p['path'] for p in contract['parent_features']} & {p['feature_path'] for p in contract['witness']['items']},'window/witness feature paths disjoint')
            for row,item in zip(new,contract['witness']['items'],strict=True):
                self.equal(row['record_id'],item['record_id'],'retained witness ID binding')
                self.equal(row['anchors'],item['anchors'],'retained witness anchor binding')
        for published in self.read(self.exported/'READOUT_SUMMARY.json'):
            raw=self.readouts[published['scope']][published['endpoint']]
            self.equal(published['responses'],len(raw['responses']),'public response count')
            self.equal(published['block_occurrences'],sum(len(r['blocks']) for r in raw['responses']),'public block count')
            blocks=[b for r in raw['responses'] for b in r['blocks']]
            expected=dict(original_first_errors=sum(b['origin']['e']>=1 for b in blocks),
                original_prefix_occurrences=sum(b['origin']['L'] for b in blocks),
                original_usable_followers=sum(max(0,b['origin']['K']-b['origin']['e']) for b in blocks if b['origin']['e']>=1),
                assessed_occurrences=sum(len(b['positions']) for b in blocks),
                margin_ties=sum(p['margin_tie'] for b in blocks for p in b['positions']),
                wrong_rival_ties=sum(p['wrong_rival_tie'] for b in blocks for p in b['positions']))
            for key,value in expected.items():self.equal(published[key],value,'public opportunities and tie counts: '+key)
            for metric in (*METRICS,'assessment'):
                self.close(published[metric],math.fsum(b['metrics'][metric] for b in blocks),'public endpoint metric total')
            for key in ('mean_response_assessment','mean_response_assessment_fp32','mean_response_assessment_fp64'):
                self.close(published[key],raw[key],'public assessment reduction')
        vectors = self.read(native/'VECTOR_CHAIN.json')
        for key, row in vectors.items():
            if 'reference_norm' in row:
                self.vectors(row)
            else:
                for stage in row.values():
                    self.vectors(stage)
        for contrast in ('F-C','F-BETA','C-BETA'):
            chain=vectors[contrast]
            self.close(chain['linear_to_finite']['candidate_norm'],chain['finite_to_cast']['reference_norm'],'finite stage shared norm')
            self.close(chain['finite_to_cast']['candidate_norm'],chain['cast_to_native']['reference_norm'],'cast stage shared norm')
            self.close(chain['linear_to_finite']['candidate_signed_projection'],chain['finite_to_cast']['reference_signed_projection'],'finite stage shared projection')
            self.close(chain['finite_to_cast']['candidate_signed_projection'],chain['cast_to_native']['reference_signed_projection'],'cast stage shared projection')
            for stage in ('native_gradient','native_point_prediction'):
                self.close(chain[stage]['difference_norm'],0,'identity vector difference')
                self.equal(chain[stage]['differing_coordinates'],0,'identity vector coordinates')
        probe=self.read(native/'NATIVE_POINT_PROBE.json')
        self.close(probe['window_assessment'],math.fsum(probe['response_assessments'])/2,'native probe response reduction')
        errors = self.read(self.exported/'PREDICTION_ERRORS.json')
        for row in errors:
            prediction, benefit = row['prediction'], row['finite_native_benefit']
            expected = prediction-benefit
            self.close(row['signed_prediction_error'], expected, 'prediction signed error')
            self.close(row['absolute_prediction_error'], abs(expected), 'prediction absolute error')
            self.close(benefit,benefits['window']['N_'+row['contrast'].replace('-','-N_')],'prediction finite benefit source')
            key='native_gradient' if row['evaluation_point']=='Q_BETA' else 'native_point_prediction'
            self.close(prediction,vectors[row['contrast']][key]['reference_signed_projection'],'prediction vector projection source')
            sign=lambda x:'positive' if x>0 else 'negative' if x<0 else 'zero'
            self.equal(row['prediction_sign'],sign(prediction),'prediction sign')
            self.equal(row['finite_native_benefit_sign'],sign(benefit),'finite sign')
            self.equal(row['signs_match'],sign(prediction)==sign(benefit),'prediction sign agreement')
            if benefit:
                self.close(row['relative_absolute_prediction_error'],abs(expected)/abs(benefit),'named prediction error ratio')
            else:
                self.equal(row['relative_absolute_prediction_error'],None,'zero benefit ratio undefined')
        with (self.exported/'CONTRASTS.csv').open() as stream:
            table = list(csv.DictReader(stream))
        for row in table:
            candidate, reference = row['contrast'].split('-')
            if row['stage']=='score_prediction':
                expected=(self.score_summaries[row['contrast']]['raw'] if row['contrast'] in self.score_summaries else
                    self.score_summaries['F-BETA']['raw']-self.score_summaries['F-C']['raw'])
                self.close(float(row['value']),expected,'score table value')
            elif row['stage']=='jvp_vjp_reconciliation':
                chain=vectors[row['contrast']]
                self.close(float(row['value']),chain['linear_to_finite']['reference_signed_projection']-chain['native_gradient']['reference_signed_projection'],'JVP/VJP reported residual')
            if row['stage'] in ('cast_shadow_functional_benefit', 'native_functional_benefit'):
                family = 'Q' if row['stage'].startswith('cast') else 'N'
                self.close(float(row['value']), benefits['window'][f'{family}_{candidate}-{family}_{reference}'],
                           'contrast table functional benefit')
        # Public compact outputs must never contain content-bearing field names.
        forbidden={'record_id','source_id','anchors','proposals','labels','canonical','rival','prediction','feature_path'}
        # prediction is an explicitly permitted scalar in prediction-error tables.
        forbidden.remove('prediction')
        def public_keys(value):
            if isinstance(value,dict):
                self.check(not (set(value)&forbidden),'public content exclusion')
                for nested in value.values(): public_keys(nested)
            elif isinstance(value,list):
                for nested in value: public_keys(nested)
        for path in self.exported.glob('*.json'):
            public_keys(json.loads(path.read_text()))
        fallacies = [
            dict(number=1,name="Simpson's paradox",status='CHECKED_DESCRIPTIVE',detail='Matched response means, every response benefit, and block gains/losses are retained. Mixed subgroup signs are reported; no aggregate-to-subgroup inference.'),
            dict(number=2,name='Ecological fallacy',status='CHECKED_NO_INFERENCE',detail='Fixed response/block results do not imply individual requests, populations, or serving gains. Anchors overlap and are not independent observations.'),
            dict(number=3,name="Berkson's paradox",status='NOT_APPLICABLE',detail='No estimated association or selected-sample correlation. Fixed TRAIN selection limits external validity and is disclosed.'),
            dict(number=4,name='Collider bias',status='NOT_APPLICABLE',detail='No regression adjustment or conditioning on post-treatment covariates; roles are frozen at the common parent.'),
            dict(number=5,name='Base rate neglect',status='CHECKED_DESCRIPTIVE',detail='Raw repair/follower counts accompany original opportunity totals; no diagnostic accuracy or population predictive probability is claimed.'),
            dict(number=6,name='Regression to the mean',status='CHECKED_CONTROLLED_LOCAL',detail='F and native C are matched one-step interventions from the same fixed warm parent. Selection is fixed; no selected-extreme cohort efficacy inference.'),
            dict(number=7,name='Survivorship bias',status='CHECKED_COMPLETE_RECEIPTS',detail='All four planned cases, all selected endpoint readouts, every reserved call and attempt are audited. C_REPEAT is an integrity check, not an independent replicate.'),
            dict(number=8,name='Look-elsewhere effect',status='CHECKED_NO_INFERENCE',detail='F-C is primary, F-BETA secondary, C-BETA derived context. All fixed contrasts and native-point probes are retained; no p-values or selective significance claims.'),
            dict(number=9,name='Garden of forking paths',status='CHECKED_BOUNDED_DIAGNOSTIC',detail='The fixed handoff, contract, source commit and ledger constrain this local diagnostic. Historical development selection prevents a confirmatory population interpretation.'),
            dict(number=10,name='Correlation versus causation',status='CHECKED_SCOPE_LIMITED',detail='Controlled local endpoint differences support only this exact intervention/window. A discrepancy does not uniquely explain the preceding 128-update serving result.'),
            dict(number=11,name='Reverse causality',status='NOT_APPLICABLE',detail='Frozen coefficients precede independently restored one-step endpoints; no cross-sectional directional association is estimated.'),
        ]
        return dict(status='passed', verification_status='ANALYZED', model_rerun=False,
                    assertions=self.checks, authenticated_files=len(self.identities),
                    scalar_tolerance=dict(atol=2e-12, rtol=2e-12),
                    benefits=benefits, score_cancellation=self.score_summaries,
                    paired_summaries=self.pair_summaries, receipts=self.receipt_summary,
                    fallacy_scan=dict(coverage='11/11',checks=fallacies),
                    raw_identities=self.identities,
                    audit_code_identity=self.identity(Path(__file__)),
                    limits=['Full model vectors were not saved, so tensor contractions cannot be independently reconstructed.',
                            'FP32 softplus/head outputs are retained native measurements, not independently recomputed logits.',
                            'Source-input and parent model identities are compared to pinned receipts; this offline audit does not rehash remote model/feature files.',
                            'Score per-pool sums and cancellation are checked; unsaved per-token coefficient/score products cannot be independently reconstructed.',
                            'Resource receipts are sampled and do not prove the absence of a transient peak between samples.',
                            'Few fixed overlapping TRAIN blocks support no population or serving claims.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--exported', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report')
    args = parser.parse_args()
    result = Audit(args.root, args.exported).run(args.attempt)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    if args.report:
        lines=['## Material Passport','', '- Origin Skill: experiment-agent', '- Origin Mode: validate',
            '- Origin Date: '+datetime.date.today().isoformat(),'- Verification Status: ANALYZED',
            '- Version Label: point1_independent_offline_v1','', '## Validation Report','',
            'The independent standard-library audit passed '+str(result['assertions'])+' assertions across '+str(result['authenticated_files'])+' retained files. It imported no production reducer and ran no model. This verifies retained scalar arithmetic, token-based readout reconstruction and receipt consistency; it does not independently reproduce unsaved tensor operations.',
            '', 'All four separately restored one-step cases, full endpoint coverage, raw/centered score cancellation, per-layer vector aggregate identities, paired response/block summaries, source/export bindings and recorded time/storage budgets were checked. No confidence intervals, p-values or independent-anchor inference were computed. Overall interpretation confidence: CAUTION, because this is a fixed TRAIN mechanism diagnostic.',
            '', '## Fallacy Scan','', '**Coverage: 11/11 checked.**','', '| Check | Status | Evidence and limit |','|---|---|---|']
        lines.extend('| '+r['name']+' | '+r['status']+' | '+r['detail']+' |' for r in result['fallacy_scan']['checks'])
        lines.extend(['','## Reproducibility and limits','','Model reproducibility verdict: CANNOT_VERIFY. Offline scalar/token reconstruction: passed. C_REPEAT is an exact execution repeat receipt, not a statistical replicate.',''])
        lines.extend('- '+value for value in result['limits'])
        Path(args.report).write_text('\n'.join(lines)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'assertions', 'authenticated_files')}))
