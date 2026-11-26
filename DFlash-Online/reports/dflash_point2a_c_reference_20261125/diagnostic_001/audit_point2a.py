"""Independent CPU/stdlib audit of Point2A receipts; no experiment imports.

Reconstructs retained-array/scalar arithmetic and proposal-derived readouts.
It cannot reconstruct the unsaved gradient, direction, or endpoint vectors.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import tarfile
import traceback

CASES = ('C', 'F_BETA', 'F_C')
CONTRASTS = (('F_C', 'F_BETA'), ('F_C', 'C'), ('F_BETA', 'C'))
METRICS = ('raw_repair', 'useful_repair', 'damaged_prefix_positions',
           'damaged_prefix_blocks', 'surviving_followers', 'reachable_followers',
           'matching_prefix_length', 'fixed_K_shortfall')
LOWER = {'damaged_prefix_positions', 'damaged_prefix_blocks', 'fixed_K_shortfall'}
PRIOR_CARD_S = 5510.8580226639515


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def shape(value):
    return [len(value)] + shape(value[0]) if isinstance(value, list) and value else []


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def vector_identity(arrays):
    digest, tensors = hashlib.sha256(), []
    for index, value in enumerate(arrays):
        metadata = dict(name='response_' + str(index), shape=shape(value), dtype='torch.float32')
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        for x in flatten(value):
            digest.update(struct.pack('<f', x))
        tensors.append(metadata)
    return dict(sha256=digest.hexdigest(), tensors=tensors)


def centered(scores, beta):
    offsets = [x - min(scores) for x in scores]
    mean = math.fsum(b*x for b, x in zip(beta, offsets)) / math.fsum(beta)
    return [x - mean for x in offsets]


def allocate64(scores, beta):
    c = centered(scores, beta)
    maximum = max(map(abs, c), default=0)
    return list(beta) if maximum == 0 else [b*(1 + 0.25*x/maximum) for b, x in zip(beta, c)]


def signed(values):
    return dict(total=math.fsum(values), positive=math.fsum(x for x in values if x > 0),
                negative=math.fsum(x for x in values if x < 0), absolute=math.fsum(map(abs, values)))


class Audit:
    def __init__(self, root, source, manifest, execution_commit, exported=None):
        self.root, self.source, self.manifest = Path(root), Path(source), Path(manifest)
        self.commit = execution_commit
        self.exported = None if exported is None else Path(exported)
        self.checks, self.identities, self.results = 0, {}, {}

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            raise AssertionError(label)

    def equal(self, actual, expected, label):
        self.check(actual == expected, f'{label}: {actual!r} != {expected!r}')

    def close(self, actual, expected, label, atol=2e-12, rtol=2e-12):
        self.check(math.isfinite(actual) and math.isfinite(expected)
                   and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
                   f'{label}: {actual!r} != {expected!r}')

    def read(self, path):
        path = Path(path)
        self.identities[str(path)] = identity(path)
        return json.loads(path.read_text())

    def bind(self, path, expected):
        actual = identity(path)
        self.identities[str(path)] = actual
        self.equal(actual, expected, 'file identity ' + str(path))

    def sources(self, contract):
        self.bind(self.manifest, contract['source_identity'])
        manifest = self.read(self.manifest)
        self.equal(manifest['commit'], self.commit, 'source commit')
        prefix = subprocess.check_output(['git', '-C', str(self.source), 'rev-parse', '--show-prefix'], text=True).strip().rstrip('/')
        repository = subprocess.check_output(['git', '-C', str(self.source), 'rev-parse', '--show-toplevel'], text=True).strip()
        tree = self.commit + (':' + prefix if prefix else '')
        archive = subprocess.check_output(['git', '-C', repository, 'archive', tree])
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            members = {m.name: m for m in stream.getmembers() if not m.isdir()}
            self.equal(set(members), set(manifest['files']), 'complete committed source membership')
            for relative, expected in manifest['files'].items():
                self.check(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'safe source path')
                member = members[relative]
                if expected.get('kind') == 'symlink':
                    self.check(member.issym(), 'source symlink kind')
                    self.equal(member.linkname, expected['target'], 'source symlink target')
                else:
                    data = stream.extractfile(member).read()
                    self.equal(dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),
                               expected, 'committed source ' + relative)
        self.results['source_files_verified'] = len(members)

    def postflight(self, contract):
        delivery = self.read(self.root.parent/'DELIVERY.json')
        self.equal(delivery['status'], 'passed', 'delivery status')
        self.equal(delivery['source_identity'], self.commit, 'delivery execution source')
        delivered = set()
        for row in delivery['files']:
            relative = Path(row['path'])
            self.check(not relative.is_absolute() and '..' not in relative.parts, 'safe delivery path')
            self.check(str(relative) not in delivered, 'unique delivery row')
            delivered.add(str(relative))
            self.bind(self.root/relative, {key: row[key] for key in ('bytes', 'sha256')})
        self.equal(delivered, {str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file()},
                   'complete raw delivery membership')
        self.equal(delivery['total_bytes'], sum(r['bytes'] for r in delivery['files']), 'delivered bytes')
        self.results['delivery_files_rehashed'] = len(delivered)
        self.results['delivery_bytes_rehashed'] = delivery['total_bytes']
        post = self.read(self.root/'POSTFLIGHT_PRIVATE.json')
        self.equal(post['status'], 'passed', 'postflight status')
        self.equal(post['execution_commit'], self.commit, 'postflight source')
        self.equal(post['source_manifest_identity'], contract['source_identity'], 'postflight manifest')
        self.check(post['no_checkpoint_or_full_vector_files'], 'no tensor files')
        self.equal(post['inherited_deletions'], [], 'no inherited deletion')
        paths = set()
        for row in post['files']:
            relative = Path(row['path'])
            self.check(not relative.is_absolute() and '..' not in relative.parts, 'safe raw path')
            self.check(str(relative) not in paths, 'unique raw identity row')
            paths.add(str(relative))
            self.bind(self.root/relative, {key: row[key] for key in ('bytes', 'sha256')})
        for row in post['checks']:
            self.check(row['passed'], 'protected artifact retained')
            self.equal(row['actual'], row['expected'], 'protected artifact identity')
        self.equal(delivery['manifest_entries_verified'], len(post['files']), 'postflight manifest coverage')
        self.check(post['storage']['free_bytes'] >= 400_000_000_000, 'postflight reserve')
        self.check(post['storage']['all_locations_charged_bytes'] <= 2_000_000_000, 'postflight disk cap')
        self.results['raw_files_rehashed'] = len(paths)
        self.results['raw_bytes_rehashed'] = sum(row['bytes'] for row in post['files'])
        self.results['protected_artifact_receipts'] = len(post['checks'])

    def endpoint(self, path):
        row = self.read(path)
        self.equal(row['status'], 'passed', 'endpoint status')
        self.check(row['state_preserved'], 'endpoint state preservation')
        self.equal(row['live_counters'], [128, 256, 1], 'readouts at restored parent')
        self.equal((row['backwards'], row['optimizer_steps'], row['teacher_forwards']), (0, 0, 0), 'readout work')
        for response in row['responses']:
            blocks = response['blocks']
            nonempty = sum(block['origin']['m'] > 0 for block in blocks)
            self.equal(nonempty, response['nonempty_blocks'], 'assessment response denominator')
            self.equal(len(response['anchors']), len(blocks), 'anchors and blocks')
            for block in blocks:
                origin, proposed = block['origin'], block['proposals']
                labels, valid = origin['labels'], origin['valid']
                self.equal(len(proposed), 16, 'native block size')
                self.equal(valid, [0 < i <= origin['m'] for i in range(16)], 'consecutive valid positions')
                self.equal(origin['prefix'], [0 < i <= origin['L'] for i in range(16)], 'original prefix mask')
                expected_error = origin['L']+1 if origin['L'] < origin['m'] else -1
                self.equal(origin['e'], expected_error, 'original first error')
                self.equal(origin['repair'], [i == expected_error for i in range(16)], 'original repair mask')
                self.equal(origin['H'], origin['K']-origin['L'] if expected_error > 0 else 0, 'original continuation horizon')
                length = 0
                for i in range(1, 16):
                    if not valid[i] or proposed[i] != labels[i]:
                        break
                    length += 1
                error = origin['e']
                damage = sum(flag and proposed[i] != labels[i] for i, flag in enumerate(origin['prefix']))
                repaired = error > 0 and proposed[error] == labels[error]
                followers = range(error + 1, origin['K'] + 1) if error > 0 else ()
                expected = dict(raw_repair=int(repaired), useful_repair=int(repaired and not damage),
                    damaged_prefix_positions=damage, damaged_prefix_blocks=int(damage > 0),
                    surviving_followers=sum(proposed[i] == labels[i] for i in followers),
                    reachable_followers=sum(length >= i for i in followers),
                    matching_prefix_length=length, fixed_K_shortfall=max(origin['K'] - length, 0))
                for metric, value in expected.items():
                    self.equal(block['metrics'][metric], value, 'proposal metric ' + metric)
                self.equal(block['current']['L'], length, 'current prefix length')
                if row['endpoint'] == 'parent':
                    self.equal(origin['L'], length, 'parent-origin consecutive prefix')
                    later = [i for i in range(expected_error+1,origin['m']+1)
                             if expected_error > 0 and proposed[i] != labels[i]]
                    endpoint = min(later)-1 if later else origin['m']
                    self.equal(origin['K'], endpoint, 'parent-origin reachable endpoint')
                self.equal([p['position'] for p in block['positions']], list(range(1, origin['K'] + 1)), 'assessment support')
                terms = []
                for position in block['positions']:
                    i, margin = position['position'], position['margin']
                    self.equal(position['canonical'], labels[i], 'canonical binding')
                    self.equal(position['prediction'], proposed[i], 'prediction binding')
                    self.equal(position['correct'], proposed[i] == labels[i], 'correctness')
                    self.equal(position['margin_tie'], margin == 0, 'tie flag')
                    self.equal(position['rival_changed'], position['rival'] != position['origin_rival'], 'rival change')
                    self.check(position['rival'] != labels[i], 'wrong rival differs from label')
                    if margin > 0:
                        self.equal(proposed[i], labels[i], 'positive margin')
                    elif margin < 0:
                        self.check(proposed[i] != labels[i], 'negative margin')
                    else:
                        self.equal(proposed[i], min(labels[i], position['rival']), 'lowest-ID tie')
                    value = position['assessment_contribution_fp32']
                    self.check(math.isfinite(value) and value >= 0, 'finite assessment term')
                    # Independent real-arithmetic softplus check, not a claim to
                    # reproduce the native GPU transcendental implementation.
                    softplus = max(-margin, 0) + math.log1p(math.exp(-abs(margin)))
                    self.close(value, (origin['K'] - i + 1)*softplus/math.log(2),
                               'native FP32 softplus within rounding', atol=2e-6, rtol=3e-7)
                    terms.append(value)
                self.close(block['assessment_sum_fp64'], math.fsum(terms), 'block FP64 sum')
            total = math.fsum(block['assessment_sum_fp64'] for block in blocks)
            self.close(response['response_assessment_fp64'], total/nonempty if nonempty else 0, 'response FP64 mean')
        self.close(row['mean_response_assessment_fp64'],
                   math.fsum(r['response_assessment_fp64'] for r in row['responses'])/len(row['responses']), 'endpoint FP64 mean')
        return row

    def paired(self, first, second, published):
        count = len(first['responses'])
        self.equal(count, len(second['responses']), 'paired response count')
        benefits, terms, deltas = [], [], {key: [] for key in METRICS}
        response_deltas = {key: [] for key in METRICS}
        for ordinal, (a, b) in enumerate(zip(first['responses'], second['responses'], strict=True)):
            self.equal((a['record_id'], a['anchors'], a['nonempty_blocks']),
                       (b['record_id'], b['anchors'], b['nonempty_blocks']), 'paired bindings')
            denominator, local = a['nonempty_blocks'], []
            own = {key: [] for key in METRICS}
            for x, y in zip(a['blocks'], b['blocks'], strict=True):
                self.equal(x['origin'], y['origin'], 'frozen readout origins')
                for left, right in zip(x['positions'], y['positions'], strict=True):
                    self.equal((left['position'], left['canonical']), (right['position'], right['canonical']), 'paired position')
                    value = left['assessment_contribution_fp32'] - right['assessment_contribution_fp32']
                    local.append(value)
                    if denominator:
                        terms.append(value/denominator/count)
                for metric in METRICS:
                    delta = y['metrics'][metric] - x['metrics'][metric]
                    deltas[metric].append(delta)
                    own[metric].append(delta)
            benefit = math.fsum(local)/denominator if denominator else 0
            benefits.append(benefit)
            self.close(published['responses'][ordinal]['assessment_benefit_fp64'], benefit, 'per-response benefit')
            for metric in METRICS:
                response_deltas[metric].append(sum(own[metric]))
        result = math.fsum(benefits)/count
        self.close(published['assessment_benefit_fp64'], result, 'paired FP64 benefit')
        self.close(published['assessment_benefit_original_fp32'], first['mean_response_assessment_fp32'] - second['mean_response_assessment_fp32'], 'original FP32 benefit')
        for side in ('positive', 'negative'):
            chosen = [x for x in terms if x > 0] if side == 'positive' else [x for x in terms if x < 0]
            self.close(published[side + '_contribution_sum_fp64'], math.fsum(chosen), 'matched ' + side)
            self.equal(published[side + '_contributions'], len(chosen), 'matched count ' + side)
        self.equal(published['zero_contributions'], sum(x == 0 for x in terms), 'matched zero count')
        self.equal(published['assessed_occurrences'], len(terms), 'assessed occurrence count')
        for family, values in (('block_metrics', deltas), ('response_metrics', response_deltas)):
            for metric, changes in values.items():
                orientation = -1 if metric in LOWER else 1
                expected = dict(delta=sum(changes), gains=sum(orientation*x > 0 for x in changes),
                                losses=sum(orientation*x < 0 for x in changes), ties=sum(x == 0 for x in changes))
                self.equal(published[family][metric], expected, family + ' ' + metric)
        return dict(assessment_benefit=result, matching_prefix_delta=sum(deltas['matching_prefix_length']),
                    useful_repair_delta=sum(deltas['useful_repair']), response_benefits=benefits)

    def qualification(self, native, parent, scoring_c, scoring_beta):
        selected = self.read(native/'AUDIT_PLAN.json')
        qualification = self.read(native/'QUALIFICATION.json')
        progress = self.read(native/'REFERENCE_PROGRESS.json')
        self.equal(qualification['status'], 'passed', 'new-direction qualification status')
        self.check(qualification['passed'], 'new-direction numerical admission')
        self.equal(qualification['selection'], selected, 'immutable selected pools')
        self.equal(qualification['direction_metadata']['primary'], scoring_c['direction_identity'], 'oracle C direction')
        self.equal(qualification['direction_metadata']['effective'], scoring_c['effective_direction_identity'], 'oracle effective direction')
        self.check(all(type(v) is bool for v in qualification['repeat_exact'].values()), 'repeatability exactness recorded')
        self.equal(progress['status'], 'completed', 'oracle completion')
        self.equal(len(progress['rows']), 28, 'fixed 28-token oracle')
        expected = []
        for i, response in enumerate(parent['responses']):
            eligible = [b for b, block in enumerate(response['blocks']) if block['origin']['L'] >= 2]
            block = eligible[0]
            positions = list(range(1, response['blocks'][block]['origin']['L'] + 1))
            self.equal(len(positions), (13, 15)[i], 'fixed original pool size')
            expected.append(dict(response_index=i, block_index=block, positions=positions))
        self.equal(selected['pools'], expected, 'lowest eligible complete pools')
        self.check(selected['selected_before_scores'], 'prospective selection receipt')
        labels = []
        for row in progress['rows']:
            self.equal(row['accumulation_factor'], 0.5, 'oracle accumulation factor')
            self.close(row['reference_score_fp64'], -row['primary_dot_fp64']/(2*row['denominator']), 'primary oracle normalization')
            scale = qualification['direction_metadata']['direction_scale']
            self.close(row['effective_reference_score_fp64'], -row['effective_dot_fp64']/(2*row['denominator']*scale), 'effective normalization')
            self.equal(row['reference_score_fp32'], f32(row['reference_score_fp64']), 'reference FP32 representation')
            labels.append({key: row[key] for key in ('response_index', 'block_index', 'position')})
        self.equal(labels, [dict(response_index=p['response_index'], block_index=p['block_index'], position=i)
                            for p in expected for i in p['positions']], 'complete oracle order')
        for pool, wanted in zip(qualification['selected_pool_receipts'], expected, strict=True):
            self.equal({key: pool[key] for key in wanted}, wanted, 'qualification pool identity')
            beta, ref = pool['beta'], pool['reference_scores_fp32']
            r, b = wanted['response_index'], wanted['block_index']
            matching = [row for row in progress['rows'] if row['response_index']==r and row['block_index']==b]
            self.equal(pool['reference_scores_fp64'], [row['reference_score_fp64'] for row in matching], 'oracle pool scalar binding')
            self.equal(ref, [row['reference_score_fp32'] for row in matching], 'oracle pool FP32 binding')
            self.equal(beta, [scoring_beta['coefficient_arrays']['BETA'][r][0][b][i] for i in wanted['positions']], 'oracle frozen beta binding')
            self.equal(pool['fast_scores_fp32'], [scoring_c['score_arrays'][r][0][b][i] for i in wanted['positions']], 'oracle full fast score binding')
            diagnostic = pool['direction_diagnostic']
            effective = [row['effective_reference_score_fp64'] for row in matching]
            for i, row in enumerate(matching):
                self.close(diagnostic['direction_cast_score_difference'][i], effective[i]-row['reference_score_fp64'], 'direction cast diagnostic')
                self.close(diagnostic['derivative_route_score_difference'][i], pool['fast_scores_fp32'][i]-effective[i], 'route diagnostic')
            center = centered(ref, beta)
            independent = allocate64(ref, beta)
            stored = pool['reference_coefficients_fp32']
            for actual, oracle, b in zip(stored, independent, beta):
                self.close(actual, oracle, 'reference allocator independent FP64', atol=2e-7+2e-6*abs(b), rtol=0)
            g_ref = math.fsum((a-b)*c for a,b,c in zip(stored,beta,center))
            allowance = math.fsum((2e-7+2e-6*abs(b))*abs(c) for b,c in zip(beta,center))
            self.close(pool['gate_c']['reference_alignment'], g_ref, 'reference centered gain')
            self.close(pool['gate_c']['tolerance'], allowance, 'alignment allowance')
            constant = len(set(ref)) == 1
            fidelity, alignment = [], []
            for repeat, alloc in enumerate(pool['fast_coefficients_fp32']):
                expected_fast = allocate64(pool['fast_scores_fp32'] if repeat == 0 else pool['repeated_fast_scores_fp32'], beta)
                for actual, oracle, b in zip(alloc, expected_fast, beta):
                    self.close(actual, oracle, 'fast allocator independent FP64', atol=2e-7+2e-6*abs(b), rtol=0)
                fidelity.append(all(abs(a-r) <= 0.025*b+2e-7+2e-6*abs(b) for a,r,b in zip(alloc,stored,beta)))
                gain = math.fsum((a-b)*c for a,b,c in zip(alloc,beta,center))
                self.close(pool['gate_c']['fast_alignment'][repeat], gain, 'fast centered gain')
                alignment.append(all(abs(a-b) <= 2e-7+2e-6*abs(b) for a,b in zip(alloc,beta))
                                 if constant else gain >= 0.9*g_ref-allowance)
            self.equal(pool['gate_b']['fast_vs_reference'], fidelity, 'coefficient fidelity decisions')
            self.equal(pool['gate_c']['fast_passed'], alignment, 'alignment decisions')
            self.check(all(fidelity) and all(alignment), 'inherited gates pass independently')
            a, b = pool['fast_coefficients_fp32']
            mutual = all(abs(x-y) <= 0.025*z+2e-7+2e-6*abs(z) for x,y,z in zip(a,b,beta))
            self.equal(pool['gate_b']['fast_repeat_vs_first'], mutual, 'repeatability criterion')
            self.check(mutual, 'repeatability passed independently')
            if qualification['repeat_exact']['allocations']:
                self.equal(a, b, 'claimed exact selected repeat')
            if qualification['repeat_exact']['scores']:
                self.equal(pool['fast_scores_fp32'], pool['repeated_fast_scores_fp32'], 'claimed exact selected scores')
        for repeat in qualification['invariants']:
            self.equal(len(repeat), 2, 'both invariant response receipts')
            for row in repeat:
                self.check(row['passed'] and all(row['checks'].values()), 'all support/mass/repair invariants')
        return labels

    def scores(self, native, parent):
        beta_receipt, c_receipt = [self.read(native/('SCORING_' + x + '.json')) for x in ('BETA', 'C')]
        arrays = dict(beta_receipt['coefficient_arrays'], **c_receipt['coefficient_arrays'])
        beta = arrays['BETA']
        for receipt in (beta_receipt, c_receipt):
            self.equal(receipt['accumulation_factor'], 0.5, 'fixed score accumulation')
            self.equal(receipt['relative_cap'], 0.25, 'fixed beta cap')
            self.check(receipt['coefficients_sealed_before_own_native_update'], 'prospective coefficients')
            self.equal(vector_identity(receipt['score_arrays']), receipt['scores'], 'score array hash')
            for name, values in receipt['coefficient_arrays'].items():
                self.equal(vector_identity(values), receipt['coefficients'][name], 'coefficient array hash ' + name)
        self.equal(c_receipt['reference'], 'C', 'changed reference name')
        self.equal(beta_receipt['reference'], 'BETA', 'old reference name')
        for r, response in enumerate(parent['responses']):
            movement_rows = {row['block_index']:row for row in c_receipt['score_changes_from_beta'][r]['per_pool']}
            for k, block in enumerate(response['blocks']):
                positions = [i for i, flag in enumerate(block['origin']['prefix']) if flag]
                base = beta[r][0][k]
                for name in CASES:
                    allocated = arrays[name][r][0][k]
                    for i in range(16):
                        self.check(math.isfinite(allocated[i]) and math.isfinite(base[i]), 'finite coefficient')
                        if i not in positions:
                            self.equal(allocated[i], base[i], 'unchanged nonprefix coefficient')
                        else:
                            tolerance = 2e-7+2e-6*abs(base[i])
                            self.check(0.75*base[i]-tolerance <= allocated[i] <= 1.25*base[i]+tolerance, 'unchanged beta cap')
                            self.check(allocated[i] > 0, 'positive coefficient')
                    self.close(math.fsum(allocated[i] for i in positions), math.fsum(base[i] for i in positions),
                               'preserved beta pool mass', atol=2e-7, rtol=2e-6)
                    if positions and name in ('F_BETA', 'F_C'):
                        receipt = beta_receipt if name == 'F_BETA' else c_receipt
                        scores = [receipt['score_arrays'][r][0][k][i] for i in positions]
                        reference = allocate64(scores, [base[i] for i in positions])
                        for i, expected in zip(positions, reference):
                            self.close(allocated[i], expected, 'full-window unchanged allocation formula',
                                       atol=2e-7+2e-6*abs(base[i]), rtol=0)
                if positions:
                    new = [c_receipt['score_arrays'][r][0][k][i] for i in positions]
                    old = [beta_receipt['score_arrays'][r][0][k][i] for i in positions]
                    weights = [base[i] for i in positions]
                    ca, cb = centered(new,weights), centered(old,weights)
                    row = movement_rows[k]
                    sign = lambda x:(x > 0)-(x < 0)
                    pairs = [(i,j) for i in range(len(positions)) for j in range(i+1,len(positions))]
                    oa = [sign(ca[i]-ca[j]) for i,j in pairs]
                    ob = [sign(cb[i]-cb[j]) for i,j in pairs]
                    expected = dict(raw_scores_equal=new==old,
                        centered_sign_changes=sum(sign(a)!=sign(b) for a,b in zip(ca,cb)),
                        centered_strict_sign_reversals=sum(a*b<0 for a,b in zip(ca,cb)),
                        ordered_pairs=len(pairs),ordering_changes=sum(a!=b for a,b in zip(oa,ob)),
                        ordering_strict_reversals=sum(a*b<0 for a,b in zip(oa,ob)),
                        ties_created=sum(a==0 and b!=0 for a,b in zip(oa,ob)),
                        ties_removed=sum(a!=0 and b==0 for a,b in zip(oa,ob)))
                    for key,value in expected.items():
                        self.equal(row[key],value,'centered score movement '+key)
                    self.close(row['raw_maximum_absolute_difference'],max(abs(a-b) for a,b in zip(new,old)),'raw score movement')
        contrasts = self.read(native/'CONTRASTS.json')
        for x, y in CONTRASTS:
            receipt = contrasts[x+'-'+y]
            all_terms = {'raw': [], 'beta_centered': []}
            changed_total = 0
            for r, response in enumerate(parent['responses']):
                saved = receipt['per_response'][r]
                response_terms = {'raw': [], 'beta_centered': []}
                movement = {row['block_index']: row for row in saved['coefficient_movement']['per_pool']}
                for k, block in enumerate(response['blocks']):
                    positions = [i for i, flag in enumerate(block['origin']['prefix']) if flag]
                    if not positions:
                        continue
                    weights = [beta[r][0][k][i] for i in positions]
                    scores = [c_receipt['score_arrays'][r][0][k][i] for i in positions]
                    center = centered(scores, weights)
                    delta = [f32(arrays[x][r][0][k][i]-arrays[y][r][0][k][i]) for i in positions]
                    changed = sum(v != 0 for v in delta)
                    changed_total += changed
                    move = movement[k]
                    self.equal(move['changed_positions'], changed, 'pool changed coefficients')
                    self.equal(move['coefficients_equal'], not changed, 'pool coefficient equality')
                    self.close(move['beta_budget'], math.fsum(weights), 'pool budget')
                    self.close(move['maximum_relative_movement'], max(abs(d)/b for d,b in zip(delta,weights)), 'maximum relative movement')
                    self.close(move['half_l1_mass_moved_over_beta_budget'], math.fsum(map(abs,delta))/(2*math.fsum(weights)), 'half-L1 movement')
                    for kind, values in (('raw', scores), ('beta_centered', center)):
                        terms = [d*s for d,s in zip(delta,values)]
                        response_terms[kind].extend(terms)
                        all_terms[kind].extend(terms)
                        self.close(saved['score_contrast']['per_pool'][kind][0][k], math.fsum(terms), 'per-pool score contraction')
                for kind in all_terms:
                    for key, expected in signed(response_terms[kind]).items():
                        self.close(saved['score_contrast'][kind][key], expected, 'per-response signed score')
            self.equal(receipt['changed_positions'], changed_total, 'contrast changed coefficients')
            self.equal(receipt['coefficient_equality'], arrays[x] == arrays[y], 'contrast coefficient equality')
            for kind, terms in all_terms.items():
                values = signed(terms)
                for key, expected in values.items():
                    self.close(receipt['prediction'][kind][key], expected, 'all-response signed score')
                self.signed_consistency(receipt['prediction'][kind], 'predicted cancellation')
            self.signed_consistency(receipt['actual_gradient_contraction'], 'reported gradient contraction scalar consistency')
        return beta_receipt, c_receipt, contrasts

    def signed_consistency(self, row, label):
        self.close(row['total'], row['positive']+row['negative'], label+' sum')
        self.close(row['absolute'], row['positive']-row['negative'], label+' absolute')
        self.equal(row['sign'], (row['total'] > 0)-(row['total'] < 0), label+' sign')
        expected = None if row['absolute'] == 0 else 1-abs(row['total'])/row['absolute']
        if expected is None:
            self.equal(row['cancellation_fraction'], None, label+' zero denominator')
        else:
            self.close(row['cancellation_fraction'], expected, label+' ratio')

    def public_tables(self, readouts, contrasts, native):
        def table(name):
            path = self.exported/name
            self.identities[str(path)] = identity(path)
            with path.open() as stream:
                return list(csv.DictReader(stream))
        response_rows, block_rows = table('ENDPOINT_READOUTS.csv'), table('ENDPOINT_BLOCKS.csv')
        expected_responses, expected_blocks = 0, 0
        for row in response_rows:
            source = readouts[row['scope']][row['endpoint']]['responses'][int(row['response_ordinal'])]
            blocks = source['blocks']
            expected = dict(block_occurrences=len(blocks),nonempty_blocks=source['nonempty_blocks'],
                assessment_fp32=source['response_assessment_fp32'],assessment_fp64=source['response_assessment_fp64'],
                existing_block_reduced_assessment=source['existing_block_reduced_assessment'],
                original_first_errors=sum(b['origin']['e']>0 for b in blocks),
                original_prefix_occurrences=sum(b['origin']['L'] for b in blocks),
                original_usable_followers=sum(b['origin']['K']-b['origin']['e'] for b in blocks if b['origin']['e']>0),
                assessed_occurrences=sum(len(b['positions']) for b in blocks),
                margin_ties=sum(p['margin_tie'] for b in blocks for p in b['positions']),
                wrong_rival_ties=sum(p['wrong_rival_tie'] for b in blocks for p in b['positions']))
            expected.update({key:math.fsum(b['metrics'][key] for b in blocks) for key in (*METRICS,'assessment')})
            for key,value in expected.items():
                self.close(float(row[key]),value,'published response field '+key)
        for row in block_rows:
            source = readouts[row['scope']][row['endpoint']]['responses'][int(row['response_ordinal'])]['blocks'][int(row['block_ordinal'])]
            expected = {key:source['origin'][key] for key in ('L','e','H','K','m')}
            expected.update({key:source['metrics'][key] for key in (*METRICS,'assessment')})
            expected['assessment_sum_fp64'] = source['assessment_sum_fp64']
            for key,value in expected.items():
                self.close(float(row[key]),value,'published block field '+key)
        for scope,rows in readouts.items():
            expected_responses += sum(len(row['responses']) for row in rows.values())
            expected_blocks += sum(len(response['blocks']) for row in rows.values() for response in row['responses'])
        self.equal(len(response_rows),expected_responses,'complete published response rows')
        self.equal(len(block_rows),expected_blocks,'complete published block rows')
        self.equal(len({(r['scope'],r['endpoint'],r['response_ordinal']) for r in response_rows}),len(response_rows),'unique response table rows')
        self.equal(len({(r['scope'],r['endpoint'],r['response_ordinal'],r['block_ordinal']) for r in block_rows}),len(block_rows),'unique block table rows')
        table_rows = table('CONTRASTS.csv')
        self.equal(len(table_rows),12,'three contrasts by four table readouts')
        seen = set()
        for row in table_rows:
            name,scope,stage = row['contrast'],row['scope'],row['stage']
            self.check((name,scope,stage) not in seen,'unique contrast table row')
            seen.add((name,scope,stage))
            raw = contrasts[name]
            if stage=='C_score_prediction':
                expected = raw['prediction']['raw']
                self.close(float(row['value']),expected['total'],'table C score value')
                self.close(float(row['beta_centered_value']),raw['prediction']['beta_centered']['total'],'table centered score')
            elif stage=='C_native_gradient_contraction':
                expected = raw['actual_gradient_contraction']
                self.close(float(row['value']),expected['total'],'table native-gradient scalar')
            else:
                self.equal(stage,'native_assessment_benefit','declared contrast table stage')
                expected = raw['native_readouts'][scope]
                self.close(float(row['value']),expected['assessment_benefit_fp64'],'table native benefit')
                self.close(float(row['original_fp32_value']),expected['assessment_benefit_original_fp32'],'table original FP32 benefit')
                for side in ('positive','negative'):
                    self.close(float(row[side]),expected[side+'_contribution_sum_fp64'],'table matched '+side)
                    self.equal(int(row[side+'_occurrences']),expected[side+'_contributions'],'table matched count '+side)
                continue
            for key in ('total','positive','negative','absolute','sign'):
                self.close(float(row[key]),expected[key],'table signed '+key)
        reference_rows = table('REFERENCE_TOKENS.csv')
        oracle = self.read(native/'REFERENCE_PROGRESS.json')['rows']
        self.equal(len(reference_rows),len(oracle),'complete published token reference rows')
        for row,raw in zip(reference_rows,oracle,strict=True):
            for key,value in row.items():
                self.equal(value,raw[key],'table token dtype') if key=='native_fkl_dtype' else self.close(float(value),raw[key],'table token scalar '+key)
        manifest = self.read(self.exported/'PUBLIC_FILES.json')
        for name,expected in manifest['files'].items():
            self.bind(self.exported/name,expected)
        # Check actual retained private identifiers and disallowed structured
        # fields, in addition to detecting absolute operational paths.
        private_ids = {r['record_id'] for rows in readouts.values() for endpoint in rows.values() for r in endpoint['responses']}
        forbidden = {'record_id','record_ids','response_ids','anchors','labels','canonical','rival','proposals',
                     'feature_path','prompt','input_ids','teacher_hidden','score_arrays','coefficient_arrays',
                     'tokens','traceback','error'}
        def visit(value):
            if isinstance(value,dict):
                self.check(not forbidden & value.keys(),'no private public JSON fields')
                for child in value.values(): visit(child)
            elif isinstance(value,list):
                for child in value: visit(child)
            elif isinstance(value,str):
                self.check(not any(marker in value for marker in ('/home/minhao','/data/sources/','/tmp/ssh-')),'no private absolute paths')
                self.check(value not in private_ids,'no retained record identifiers')
        scanned = 0
        for path in sorted(self.exported.iterdir()):
            if path.suffix=='.json' and path.name not in ('INDEPENDENT_POINT2A_AUDIT.json',):
                visit(json.loads(path.read_text())); scanned += 1
            elif path.suffix=='.csv':
                with path.open() as stream:
                    reader = csv.DictReader(stream)
                    self.check(not forbidden & set(reader.fieldnames),'no private CSV columns')
                    for row in reader: visit(row)
                scanned += 1
        self.results['public_table_audit'] = dict(status='passed',response_rows=len(response_rows),
            block_rows=len(block_rows),contrast_rows=len(table_rows),token_rows=len(reference_rows),
            source_manifest_files=len(manifest['files']),privacy_files_scanned=scanned)

    def run(self, attempt):
        contract = self.read(self.root/'CONTRACT.json')
        self.equal(contract['schema'], 'point2a_c_referenced_scoring_v1', 'contract schema')
        self.equal(contract['status'], 'admitted', 'prospective admission')
        self.equal(contract['execution_commit'], self.commit, 'execution pin')
        self.equal(contract['cases'], list(CASES), 'fixed cases')
        self.equal(contract['primary_contrast'], 'F_C-F_BETA', 'primary contrast')
        self.equal(contract['prior_occupied_card_s'], PRIOR_CARD_S, 'historical costs retained')
        self.equal(contract['old_hidden_gate_status'], 'failed_retained_diagnostic', 'historical failure retained')
        self.equal(contract['limits'], dict(elapsed_s=3600,occupied_card_s=3600,allocated_bytes=2_000_000_000,
            attempted_adam_calls=6,planned_adam_calls=3,correction_retries=1,reserve_bytes=400_000_000_000,
            oracle_backwards_per_attempt=30,oracle_backwards_total=60), 'unchanged prospective bounds')
        criteria = contract['numerical_criteria']
        for key,value in dict(relative_cap=0.25,mass_atol=2e-7,mass_rtol=2e-6,
            gate_b_relative_beta=0.025,gate_c_alignment_fraction=0.9,maximum_selected_pools=2,
            maximum_reference_backwards_per_audit=30,maximum_reference_backwards_including_one_correction=60,
            expected_selected_pool_sizes=[13,15]).items():
            self.equal(criteria[key],value,'prospective numerical criterion '+key)
        self.sources(contract)
        self.postflight(contract)
        native = self.root/f'attempt_{attempt:03d}'/'native'
        result = self.read(native/'RESULT.json')
        self.equal(result['status'], 'completed', 'native completion')
        self.equal(result['execution_commit'], self.commit, 'native execution pin')
        self.equal(result['contract_identity'], identity(self.root/'CONTRACT.json'), 'native prospective contract')
        for key in ('new_feedback', 'generations', 'exports', 'full_vectors_written'):
            self.equal(result[key], 0, 'no new ' + key)
        cases = {name: self.read(native/(name+'.json')) for name in CASES}
        for name, case in cases.items():
            self.check(case['status'] in ('completed', 'reused'), 'completed native case')
            self.equal((case['endpoint']['global_step'], case['endpoint']['presentations'], case['endpoint']['micro']), (129,258,258), 'native endpoint counters')
            self.check(case['endpoint']['gradients_cleared'], 'clean native endpoint')
            if case['status'] == 'completed':
                self.equal((case['adam_attempts'],case['adam_completed']), (1,1), 'single native Adam')
            for suffix in ('INITIAL_STATE', 'RESTORATION'):
                state = self.read(native/(name+'_'+suffix+'.json'))
                self.check(state['status']=='passed' and state['exact_resume'] and state['runtime_exact'] and state['ancillary_exact'], 'full branch restoration')
        for name in ('C', 'F_BETA'):
            self.check(cases[name]['historical_reproduction']['complete_recorded_scientific_identity'], 'complete historical control identity')
            previous = 'C' if name == 'C' else 'F'
            historical_path = self.root.parents[2]/'dflash_point1_update_fidelity_20261125'/'diagnostic_001'/'raw'/'attempt_001'/'native'/(previous+'.json')
            self.bind(historical_path, contract['historical_controls'][name]['identity'])
            historical = self.read(historical_path)
            def science(case):
                row = {key:case[key] for key in ('gradient_identity','endpoint','native_preclip_norm','adam_completed')}
                row['rows'] = json.loads(json.dumps(case['rows']))
                for response in row['rows']:
                    response.pop('training_s',None)
                    response.get('response_observation',{}).pop('event_id',None)
                return row
            self.equal(science(cases[name]),science(historical),'independent complete historical control equality '+name)
        for name in ('AFTER_STARTUP', 'AFTER_QUALIFICATION', 'FINAL_RESTORATION'):
            state = self.read(native/(name+'.json'))
            self.check(state['status']=='passed' and state['exact_resume'], 'phase restoration')
        parent_identity = self.read(native/'PARENT_STATE_IDENTITY.json')
        preserved = self.read(native/'SCORING_STATE_PRESERVATION.json')
        self.check(preserved['checked_before_restoration'] and preserved['status']=='passed', 'scoring pre-restoration check')
        self.equal(preserved['complete_endpoint_identity'], parent_identity, 'scoring parent identity')
        readouts, benefits = {}, {}
        for scope, prefix in (('window','window'), ('new_buffer','witness')):
            rows = {name: self.endpoint(native/(prefix+'_'+name+'.json')) for name in ('parent','N_C','N_F_BETA','N_F_C')}
            self.equal(len(rows['parent']['responses']), 2 if scope=='window' else 8, 'frozen set response count')
            readouts[scope], benefits[scope] = rows, {}
        self.check(not {r['record_id'] for r in readouts['window']['parent']['responses']} &
                   {r['record_id'] for r in readouts['new_buffer']['parent']['responses']}, 'disjoint TRAIN record IDs')
        beta_scoring, c_scoring, contrasts = self.scores(native, readouts['window']['parent'])
        self.equal(c_scoring['reference_gradient'], cases['C']['gradient_identity'], 'actual C preclip reference')
        self.equal(c_scoring['reference_binding']['parent_identity'], parent_identity, 'shadow parent binding')
        self.check(c_scoring['reference_binding']['parent_restored_before_shadow'], 'restore before C shadow')
        oracle_labels = self.qualification(native, readouts['window']['parent'], c_scoring, beta_scoring)
        for scope, rows in readouts.items():
            for x,y in CONTRASTS:
                published = contrasts[x+'-'+y]['native_readouts'][scope]
                benefits[scope][x+'-'+y] = self.paired(rows['N_'+y], rows['N_'+x], published)
        ledger, clock = self.read(self.root/'STEP_LEDGER.json'), self.read(self.root/'CLOCK.json')
        self.equal(ledger['contract_identity'], identity(self.root/'CONTRACT.json'), 'ledger contract')
        self.check(len(ledger['calls']) <= 6 and len(ledger['attempts']) <= 2 and len(ledger['token_backwards']) <= 60, 'global work caps')
        tokens = [r for r in ledger['token_backwards'] if r['attempt']==attempt]
        self.equal([r['label'] for r in tokens], oracle_labels, 'token reservation order')
        self.check(all(r['status']=='completed' for r in tokens), 'completed reserved oracle work')
        calls = [r for r in ledger['calls'] if r['attempt']==attempt]
        self.equal([r['case'] for r in calls], list(CASES), 'C-first native schedule')
        self.check(all(r['status']=='completed' for r in calls), 'completed reserved native work')
        for a in ledger['attempts']:
            self.check(sum(r['attempt']==a['number'] for r in ledger['token_backwards']) <= 30, 'per-attempt oracle cap')
            if a['number'] > 1:
                self.check(bool(a['correction_note']), 'identified correction retry')
        work = self.read(native/'WORK.json')
        self.equal(work['pending'], [], 'no pending work')
        self.equal(work['native_adam_steps'], 3, 'native Adam accounting')
        process = self.read(self.root/f'attempt_{attempt:03d}'/'PROCESS.json')
        self.equal(process['status'], 'completed', 'supervised completion')
        self.check(process['new_phase_elapsed_s'] <= 3600 and clock['completed_occupied_card_s'] <= 3600, 'phase time/card bounds')
        self.equal(clock['historical_occupied_card_s'], PRIOR_CARD_S, 'clock historical costs')
        self.close(process['historical_plus_new_occupied_card_s'], PRIOR_CARD_S+clock['completed_occupied_card_s'], 'cumulative card costs')
        self.check(process['all_locations_charged_bytes'] <= 2_000_000_000 and process['free_bytes'] >= 400_000_000_000, 'supervised storage limits')
        if self.exported is not None:
            public = self.read(self.exported/'READOUT_COMPARISONS.json')
            for scope, rows in readouts.items():
                for x,y in CONTRASTS:
                    self.paired(rows['N_'+y], rows['N_'+x], public[scope][x+'-'+y])
            for relative, expected in self.read(self.exported/'IDENTITIES.json')['raw_input_receipts'].items():
                self.bind(self.root/relative, expected)
            self.public_tables(readouts,contrasts,native)
        self.results.update(status='passed', verification_status='ANALYZED', assertions=self.checks,
            execution_commit=self.commit, source_files_verified=self.results['source_files_verified'],
            authenticated_files=len(self.identities), raw_identities=self.identities, benefits=benefits,
            fallacy_checks=fallacies(), fallacy_checks_covered=11,
            arithmetic_tolerances=dict(scalar_atol=2e-12, scalar_rtol=2e-12,
                independent_allocator='Inherited MASS_ATOL+MASS_RTOL*abs(beta); no changed admission gate',
                softplus_atol=2e-6, softplus_rtol=3e-7),
            limitations=[
                'No full vectors retained: scalar consistency does not independently reconstruct gradient/direction/endpoint tensor contractions.',
                'Native logits are unavailable. Margin/argmax consistency and FP32 contribution rounding are checked from retained measurements.',
                'Protected remote input identities are authenticated receipts; original multi-GB inputs are not reread by this local CPU audit.',
                'Repeatedly inspected overlapping TRAIN occurrences are development diagnostics, not independent observations or serving evidence.',
                'No mechanism adoption, further run, or research-stage advancement follows this audit.'])
        return self.results


def fallacies():
    return [dict(name=name, status=status, assessment=explanation) for name,status,explanation in (
        ("Simpson's Paradox", 'checked', 'Keep per-response gains/losses/ties and local versus disjoint outcomes separate; aggregates cannot erase conflicts.'),
        ('Ecological Fallacy', 'checked', 'No population or individual-generalization claim from aggregated TRAIN occurrences.'),
        ("Berkson's Paradox", 'limitation', 'Fixed development selection is deliberate; it does not represent a random serving population.'),
        ('Collider Bias', 'limitation', 'Roles are frozen from the parent before endpoint outcomes; conditioning on these roles limits interpretation.'),
        ('Base Rate Neglect', 'checked', 'Reports retain original prefix/error/follower counts and response/nonempty-block denominators.'),
        ('Regression to the Mean', 'limitation', 'One fixed window and one-step controls cannot establish robustness; no extra window is selected after results.'),
        ('Survivorship Bias', 'checked', 'Reserved failed work remains charged, complete selected pools are required, and all planned cases are retained.'),
        ('Look-Elsewhere Effect', 'checked', 'Three prospective contrasts are descriptive; no significance tests over overlapping anchors.'),
        ('Garden of Forking Paths', 'checked', 'Single C-reference change, sealed pools/coefficients and fixed criteria; no result-dependent tuning.'),
        ('Correlation != Causation', 'limitation', 'Restored controlled branches identify this local intervention only, not a general cause of prior serving outcomes.'),
        ('Reverse Causality', 'checked', 'Direction and coefficient seals precede candidate updates; endpoint readouts cannot select the scoring rule.'),
    )]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--execution-commit', required=True)
    parser.add_argument('--attempt', default=1, type=int)
    parser.add_argument('--exported')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit = Audit(args.root,args.source,args.source_manifest,args.execution_commit,args.exported)
    try:
        result = audit.run(args.attempt)
    except BaseException as error:
        result = dict(status='failed', assertions=audit.checks, error=repr(error),
                      traceback=traceback.format_exc(), raw_identities=audit.identities)
        Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
        raise
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('status','assertions','authenticated_files')}))


if __name__ == '__main__':
    main()
