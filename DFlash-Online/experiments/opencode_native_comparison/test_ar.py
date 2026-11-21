"""CPU fixtures for AR contracts; these do not qualify native GPU equivalence."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from online_feedback.native import validate_generation
from online_feedback.sglang_backend import SGLangDFlashBackend
from . import ar_backend, ar_worker
from .ar_backend import SGLangARBackend, engine_settings
from .validity import validate_ar_output, validate_canonical, validate_measurement


def metadata(tokens, *, cap, prompt=2):
    eos = tokens[-1] == 99
    reason = dict(type='length', length=cap) if len(tokens) == cap else dict(type='stop', matched=99)
    return dict(prompt_tokens=prompt, completion_tokens=len(tokens), cached_tokens=0,
                finish_reason=reason)


def validate(tokens, cap):
    return validate_ar_output([1, 2], tokens, metadata(tokens, cap=cap),
                              max_new_tokens=cap, stop_token_ids=[99])


def test_real_ar_config_has_no_draft_or_block_size_emulation():
    settings = engine_settings('/target')
    assert settings['speculative_algorithm'] is None
    assert 'speculative_draft_model_path' not in settings
    assert 'speculative_num_draft_tokens' not in settings
    assert settings['context_length'] == 4112
    assert settings['dtype'] == 'bfloat16'
    assert settings['attention_backend'] == 'fa3'
    assert settings['disable_cuda_graph'] is False
    # Keep ordinary decode graphs; match native DFlash's automatically disabled
    # experimental prefill graph path when speculation is enabled.
    assert settings['disable_piecewise_cuda_graph'] is True
    assert settings['chunked_prefill_size'] == 8192
    assert settings['cuda_graph_max_bs'] == 8
    assert settings['skip_server_warmup'] is True
    assert settings['max_running_requests'] == 1
    with pytest.raises(ValueError, match='4096'):
        engine_settings('/target', 2048)


@pytest.mark.parametrize('tokens,cap,finish', [
    ([3], 1, 'length'), ([99], 1, 'eos'), ([3, 99], 17, 'eos'),
    ([3] * 17, 17, 'length'), ([3] * 2048, 2048, 'length')])
def test_ar_caps_and_eos(tokens, cap, finish):
    assert validate(tokens, cap) == (tokens, finish)


@pytest.mark.parametrize('tokens,cap', [([], 1), ([3, 4], 1), ([99, 4], 17),
                                     ([3], 17), ([True], 1), ([-1], 1)])
def test_invalid_native_tokens_fail(tokens, cap):
    meta = metadata(tokens or [3], cap=cap)
    with pytest.raises(ValueError):
        validate_ar_output([1, 2], tokens, meta, max_new_tokens=cap, stop_token_ids=[99])


@pytest.mark.parametrize('change', [
    {'completion_tokens': 2}, {'prompt_tokens': 3}, {'cached_tokens': 1},
    {'finish_reason': {'type': 'abort', 'message': 'failed'}},
    {'finish_reason': {'type': 'stop', 'matched': 99}},
    {'finish_reason': {'type': 'length', 'length': 17}}])
def test_raw_metadata_contradictions_fail(change):
    meta = metadata([3], cap=1)
    meta.update(change)
    with pytest.raises(ValueError):
        validate_ar_output([1, 2], [3], meta, max_new_tokens=1, stop_token_ids=[99])


def test_logical_context_is_not_physical_headroom():
    with pytest.raises(ValueError, match='context'):
        validate_ar_output([1] * 2049, [3] * 2048,
            metadata([3] * 2048, cap=2048, prompt=2049), max_new_tokens=2048,
            stop_token_ids=[99])


def test_native_zero_verification_is_valid_for_short_cap():
    assert validate_generation([1], [1, 3], [], max_new_tokens=1,
                               stop_tokens=[99]) == ([3], 'length')
    row = dict(response_ids=[3], metadata=dict(request_latency_s=0.2,
        native_generate_s=0.1, verification_calls=0, accepted_draft_tokens=0,
        committed_progress=0, trace=[]))
    validate_measurement(row, arm='F')
    row['response_ids'].append(4)
    row['metadata']['committed_progress'] = 1
    with pytest.raises(ValueError, match='Zero-call'):
        validate_measurement(row, arm='F')


def test_stop_clipped_acceptance_cannot_exceed_committed_output():
    row = dict(response_ids=[3, 99], metadata=dict(request_latency_s=0.2,
        native_generate_s=0.1, verification_calls=1, accepted_draft_tokens=8,
        committed_progress=1, trace=[dict(accepted_draft_tokens=8)]))
    with pytest.raises(ValueError, match='Stop-clipped'):
        validate_measurement(row, arm='F')


def fake_backend(tmp_path, monkeypatch, *, outputs=None):
    backend = SGLangARBackend.__new__(SGLangARBackend)
    backend.root = tmp_path
    backend.context_limit = 4096
    backend.serial = 0
    backend.stop_tokens = [99]
    backend.checkpoint_id = '/target'
    backend.failed = False
    tokens = [3] if outputs is None else outputs
    native = dict(output_ids=tokens, meta_info=metadata(tokens, cap=1),
                  native_generate_s=0.0, cache_maintenance_s=0.0)
    backend._call = lambda *args, **kwargs: copy.deepcopy(native)
    monkeypatch.setattr(ar_backend, 'preserve_rng', nullcontext)
    return backend, native


def test_backend_records_na_and_complete_worker_timer(tmp_path, monkeypatch):
    backend, _ = fake_backend(tmp_path, monkeypatch)
    row = backend.generate([1, 2], max_new_tokens=1, request_id='r',
        session_id='s', request_index=0)
    assert row['validation']['status'] == 'passed'
    assert row['metadata']['verification_calls'] is None
    assert row['metadata']['accepted_draft_tokens'] is None
    assert row['metadata']['trace'] is None
    assert row['metadata']['feedback_capture_in_timed_request'] is False
    assert row['metadata']['request_latency_s'] >= row['metadata']['native_generate_s']
    validate_measurement(row, arm='AR')
    row['metadata']['verification_calls'] = 0
    with pytest.raises(ValueError, match='N/A'):
        validate_measurement(row, arm='AR')


def test_validation_failure_is_saved_before_raising(tmp_path, monkeypatch):
    backend, native = fake_backend(tmp_path, monkeypatch)
    native['meta_info']['completion_tokens'] = 400
    path = tmp_path / 'score.json'
    with pytest.raises(ValueError, match='completion count'):
        backend.generate([1, 2], max_new_tokens=1, request_id='r', session_id='s',
                         request_index=0, save_to=path)
    saved = json.loads(path.read_text())
    assert saved['response_ids'] == [3]
    assert saved['validation']['status'] == 'failed'
    assert saved['validation']['errors']
    assert backend.failed


def test_ar_cannot_collect_feedback_or_exceed_cap(tmp_path, monkeypatch):
    backend, _ = fake_backend(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='never learning'):
        backend.generate([1], max_new_tokens=1, request_id='r', session_id='s',
                         request_index=0, purpose='serving')
    with pytest.raises(ValueError, match='budget'):
        backend.generate([1], max_new_tokens=2049, request_id='r', session_id='s', request_index=0)
    assert not hasattr(backend, 'publish')
    assert not hasattr(backend, 'prepare_feedback')
    assert SGLangARBackend._call is SGLangDFlashBackend._call
    assert SGLangARBackend.close is SGLangDFlashBackend.close


def test_exact_canonical_checks(tmp_path, monkeypatch):
    backend, _ = fake_backend(tmp_path, monkeypatch)
    row = backend.generate([1, 2], max_new_tokens=1, request_id='r', session_id='s', request_index=0)
    canonical = copy.deepcopy(row)
    canonical['record_id'] = 'different_canonical_execution'
    validate_canonical(row, canonical)
    validate_canonical(row, canonical, expected_record_id='r')
    with pytest.raises(ValueError, match='execution record_id'):
        validate_canonical(row, canonical, expected_record_id='wrong')
    canonical['response_ids'] = [4]
    with pytest.raises(ValueError, match='response_ids'):
        validate_canonical(row, canonical)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.idle = True
        self.cached = 0
        self.flush_success = True

    def flush_cache(self):
        self.calls.append('flush')
        return SimpleNamespace(success=self.flush_success)

    def collective_rpc(self, method, *, outputfile):
        self.calls.append(method)
        Path(outputfile).write_text(json.dumps(dict(idle=self.idle,
            cached_tokens=self.cached, speculation_disabled=True,
            draft_worker_absent=True, overlap_enabled=True)))

    def generate(self, *, input_ids, sampling_params):
        self.calls.append(('generate', input_ids, sampling_params))
        return dict(output_ids=[3], meta_info=metadata([3], cap=1))


def worker_fixture(tmp_path):
    worker = ar_worker.ARWorker.__new__(ar_worker.ARWorker)
    worker.root = tmp_path
    worker.engine = FakeEngine()
    return worker


def test_worker_flushes_and_inspects_before_native_generate(tmp_path):
    worker = worker_fixture(tmp_path)
    request = dict(prompt_ids=[1, 2], max_new_tokens=1, stop_token_ids=[99],
                   trace_path=str(tmp_path / 'raw.json'))
    row = worker.generate(request)
    assert worker.engine.calls[:2] == ['flush', 'code_ar_cache_state']
    assert worker.engine.calls[2] == ('generate', [1, 2], dict(temperature=0, top_p=1,
        top_k=1, max_new_tokens=1, ignore_eos=False, stop_token_ids=[99]))
    assert worker.engine.calls[3] == 'code_ar_cache_state'
    assert row['speculative_observer'] is False
    assert row['cache_before']['cached_tokens'] == 0
    assert json.loads((tmp_path / 'raw.json').read_text())['output_ids'] == [3]


@pytest.mark.parametrize('kind', ['flush', 'populated', 'busy'])
def test_cache_failure_does_not_generate(tmp_path, kind):
    worker = worker_fixture(tmp_path)
    if kind == 'flush':
        worker.engine.flush_success = False
    elif kind == 'populated':
        worker.engine.cached = 10
    else:
        worker.engine.idle = False
    with pytest.raises(RuntimeError):
        worker.generate(dict(prompt_ids=[1], max_new_tokens=1, stop_token_ids=[99],
                             trace_path=str(tmp_path / 'raw.json')))
    assert not any(isinstance(call, tuple) for call in worker.engine.calls)


def test_raw_native_output_survives_postgeneration_idle_failure(tmp_path):
    worker = worker_fixture(tmp_path)
    original = worker.engine.generate

    def generate(**kwargs):
        result = original(**kwargs)
        worker.engine.idle = False
        return result

    worker.engine.generate = generate
    with pytest.raises(RuntimeError, match='idle'):
        worker.generate(dict(prompt_ids=[1, 2], max_new_tokens=1, stop_token_ids=[99],
                             trace_path=str(tmp_path / 'raw.json')))
    assert json.loads((tmp_path / 'raw.json').read_text())['output_ids'] == [3]


def test_cache_inspection_allows_native_ar_overlap():
    scheduler = SimpleNamespace(_is_no_request=lambda: True, waiting_queue=[],
        chunked_req=None, grammar_manager=SimpleNamespace(grammar_queue=[]),
        tree_cache=SimpleNamespace(total_size=lambda: 0),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        draft_worker=None, enable_overlap=True)
    state = ar_worker.cache_state(scheduler)
    assert state['idle'] and state['speculation_disabled'] and state['overlap_enabled']


def test_startup_counter_only_observes_existing_forwards(monkeypatch):
    monkeypatch.setattr(ar_worker, '_INITIALIZATION', [])
    graph = SimpleNamespace(model_runner=SimpleNamespace(is_draft_worker=False), num_tokens_per_bs=1)
    calls, receipts = [], []

    def original(graph, bs, forward):
        return [forward(i) for i in range(3)]

    wrapped = ar_worker.observe_capture(original,
        lambda: receipts.append(copy.deepcopy(ar_worker._INITIALIZATION)))
    assert wrapped(graph, 1, lambda i: calls.append(i) or i) == [0, 1, 2]
    assert calls == [0, 1, 2]
    row = receipts[-1][0]
    assert row['status'] == 'completed'
    assert row['model_forward_calls'] == row['model_forward_completions'] == 3
    assert row['model_forward_failures'] == 0


def test_startup_failure_keeps_attempted_work(monkeypatch):
    monkeypatch.setattr(ar_worker, '_INITIALIZATION', [])
    graph = SimpleNamespace(model_runner=SimpleNamespace(is_draft_worker=False), num_tokens_per_bs=1)
    receipts = []

    def failing_forward():
        raise RuntimeError('fixture failure')

    wrapped = ar_worker.observe_capture(lambda graph, bs, forward: forward(),
        lambda: receipts.append(copy.deepcopy(ar_worker._INITIALIZATION)))
    with pytest.raises(RuntimeError, match='fixture failure'):
        wrapped(graph, 1, failing_forward)
    row = receipts[-1][0]
    assert row['status'] == 'failed'
    assert row['model_forward_calls'] == row['model_forward_failures'] == 1
    assert row['model_forward_completions'] == 0


def gate_inputs():
    def row(index, role):
        return dict(request_id=f'{role}_{index}', request_index=index,
                    prompt_ids=[index + 100] * (index + 1), stop_token_ids=[99],
                    max_new_tokens=2048)
    return dict(qualification_requests=[row(i, 'q') for i in range(16)],
                warmup_requests=[row(i + 20, 'w') for i in range(2)])


class FakeLedger:
    def __init__(self):
        self.calls = []

    def call(self, counts, label, function, *args, **kwargs):
        self.calls.append((counts, label))
        return function(*args, **kwargs)


class GateBackend:
    def __init__(self, method, *, wrong=False):
        self.method, self.wrong = method, wrong
        self.checkpoint_id = '/target' if method == 'AR' else '/draft'

    def generate(self, prompt_ids, **kwargs):
        cap = kwargs['max_new_tokens']
        tokens = [3] if cap == 1 else [3, 99]
        if self.wrong:
            tokens[0] = 4
        trace = [] if cap == 1 else [dict(output_start=0, anchor_token=tokens[0],
            committed_progress=1, committed_tokens=[99], accepted_draft_tokens=0)]
        ar = self.method == 'AR'
        return dict(record_id=kwargs['request_id'], prompt_ids=prompt_ids,
            response_ids=tokens, finish_reason='length' if cap == 1 else 'eos',
            checkpoint_id=self.checkpoint_id, stop_token_ids=[99], max_new_tokens=cap,
            request_index=kwargs['request_index'], purpose=kwargs['purpose'],
            metadata=dict(request_latency_s=0.2, native_generate_s=0.1,
                native_meta_info=metadata(tokens, cap=cap, prompt=len(prompt_ids)),
                verification_calls=None if ar else len(trace),
                accepted_draft_tokens=None if ar else 0,
                committed_progress=None if ar else len(tokens)-1,
                committed_progress_per_call=None,
                calls_per_1000_output_tokens=None if ar else 1., trace=None if ar else trace))

    def prepare_feedback(self, record, path):
        return dict(status='prepared', feature_path=str(path))


def test_gate_exact_slots_cover_repeats_caps_capture_and_cards():
    from collections import Counter
    from .serving_gate import qualification_slots
    slots = qualification_slots(gate_inputs())
    assert len(slots) == 60
    assert Counter(s['card'] for s in slots) == {'A': 56, 'B': 4}
    assert Counter(s['method'] for s in slots) == {'AR': 30, 'F': 30}
    assert sum(s['collect_features'] for s in slots) == 16
    assert Counter(s['max_new_tokens'] for s in slots) == {2048: 44, 1: 8, 17: 8}


def test_full_cpu_gate_checks_native_shapes_and_accounting(tmp_path):
    from .serving_gate import qualify_card, merge_qualification
    plan = dict(target='/target', checkpoints={'F': '/draft'}, devices={'A': 'GPUa', 'B': 'GPUb'},
                execution_commit='fixture', canonical_provenance={'fixture': True}, session='qual')
    inputs, ledger, lifetimes = gate_inputs(), FakeLedger(), []

    def factory(method, gpu, root):
        lifetimes.append((method, gpu))
        return GateBackend(method)

    roots = {card: tmp_path / card for card in ('A', 'B')}
    for card in roots:
        qualify_card(plan, inputs, card, roots[card], ledger=ledger, backend_factory=factory,
                     close_backend=lambda *args: None, boundary=lambda: None)
    result = merge_qualification(plan, inputs, roots, tmp_path / 'QUALIFIED.json')
    assert result['status'] == 'passed_initial_serving'
    assert result['executions'] == 60
    assert result['publication_generations_remaining'] == 4
    assert len(result['warmup_canonicals']) == 2
    assert all(set(bundle) == {'AR','F'}
        and bundle['AR']['reference_kind'] == 'target_only_AR'
        and bundle['F']['reference_kind'] == 'frozen_DFlash'
        for bundle in result['warmup_canonicals'].values())
    assert sum(call[0].get('generation', 0) for call in ledger.calls) == 60
    assert sum(call[0].get('teacher_preparation', 0) for call in ledger.calls) == 16
    assert len(lifetimes) == 4


@pytest.mark.parametrize('wrong_method,valid,unattempted',[('AR',18,37),('F',46,9)])
def test_gate_preserves_first_invalid_and_all_unattempted_slots(tmp_path,wrong_method,valid,unattempted):
    from .serving_gate import qualify_card
    plan = dict(target='/target', checkpoints={'F': '/draft'}, devices={'A': 'GPUa', 'B': 'GPUb'},
                execution_commit='fixture', canonical_provenance={}, session='qual')
    class WrongRepeat(GateBackend):
        def generate(self, prompt_ids, **kwargs):
            self.wrong = self.method == wrong_method and '_normal_repeat_' in kwargs['request_id']
            return super().generate(prompt_ids, **kwargs)

    with pytest.raises(ValueError, match='Serving gate failed'):
        qualify_card(plan, gate_inputs(), 'A', tmp_path / 'failed', ledger=FakeLedger(),
            backend_factory=lambda method, gpu, root: WrongRepeat(method),
            close_backend=lambda *args: None, boundary=lambda: None)
    work = json.loads((tmp_path / 'failed' / 'WORK.json').read_text())
    assert sum(row['status'] == 'valid' for row in work.values()) == valid
    assert sum(row['status'] == 'invalid' for row in work.values()) == 1
    assert sum(row['status'] == 'unattempted' for row in work.values()) == unattempted
    failed = next(row for row in work.values() if row['status'] == 'invalid')
    assert json.loads(Path(failed['record_path']).read_text())['response_ids'] == [4, 99]


def test_precheck_refuses_partial_panel_without_modelwork(tmp_path):
    from .serving_gate import panel_precheck
    ledger = FakeLedger()
    with pytest.raises(ValueError, match='1,024'):
        panel_precheck({}, {'evaluation_requests': []}, 'A', tmp_path / 'partial',
            ledger=ledger, backend_factory=None, close_backend=None, boundary=lambda: None)
    assert ledger.calls == []


def test_gate_retains_cross_path_difference_with_exact_within_path_repeats(tmp_path):
    from .serving_gate import qualify_card, merge_qualification
    plan = dict(target='/target', checkpoints={'F':'/draft'}, devices={'A':'GPUa','B':'GPUb'},
                execution_commit='fixture', canonical_provenance={}, session='qual')
    inputs, ledger = gate_inputs(), FakeLedger()
    roots = {card:tmp_path/card for card in ('A','B')}
    for card in roots:
        qualify_card(plan, inputs, card, roots[card], ledger=ledger,
            backend_factory=lambda method,gpu,root:GateBackend(method,wrong=method=='F'),
            close_backend=lambda *args:None,boundary=lambda:None)
    result=merge_qualification(plan, inputs, roots, tmp_path/'QUALIFIED.json')
    assert result['status']=='passed_initial_serving'
    assert len(ledger.calls)==60+16
    for bundle in result['warmup_canonicals'].values():
        assert bundle['AR']['response_ids']==[3,99]
        assert bundle['F']['response_ids']==[4,99]
    work=json.loads((roots['A']/'WORK.json').read_text())
    comparisons=[comparison for row in work.values() for comparison in row['AR_speculative_comparisons']]
    assert comparisons and all(row['output_equal'] is False for row in comparisons)
    assert all(row['status']=='valid' for row in work.values())


@pytest.mark.parametrize('card',['A','B'])
def test_first_reused_ar_slot_preserves_bytes_without_model_or_ledger_call(tmp_path,card):
    from .serving_gate import _run_slots, qualification_slots
    from experiments.ok_coding.storage import identity
    inputs=gate_inputs()
    slot=next(row for row in qualification_slots(inputs) if row['card']==card and row['method']=='AR')
    request=next(row for row in inputs['qualification_requests'] if row['request_id']==slot['request_id'])
    record=GateBackend('AR').generate(request['prompt_ids'],max_new_tokens=slot['max_new_tokens'],
        request_id=slot['slot_id'],request_index=request['request_index'],purpose='evaluation')
    retained=tmp_path/'retained.json'
    retained.write_text(json.dumps(record,indent=3)+'\n')
    ledger=FakeLedger()
    def unexpected(*args,**kwargs):
        raise AssertionError('Reused raw records must not start a model or invoke close')
    plan=dict(target='/target',checkpoints={'F':'/draft'},devices={'A':'GPUa','B':'GPUb'},
              execution_commit='fixture',canonical_provenance={},session='qual')
    records,_=_run_slots(plan,{request['request_id']:request},[slot],tmp_path/'resumed',
        ledger=ledger,backend_factory=unexpected,close_backend=unexpected,boundary=lambda:None,
        category={'generation':1},reused={slot['slot_id']:dict(record_path=str(retained),identity=identity(retained))})
    destination=tmp_path/'resumed'/'records'/(slot['slot_id']+'.json')
    assert destination.read_bytes()==retained.read_bytes()
    assert records[slot['slot_id']]==record and not ledger.calls
    work=json.loads((tmp_path/'resumed'/'WORK.json').read_text())
    assert work[slot['slot_id']]['status']=='valid'
    assert work[slot['slot_id']]['reused_origin']['record_path']==str(retained)
