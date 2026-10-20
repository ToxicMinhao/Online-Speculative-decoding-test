"""Qualification must gate a completely fresh campaign; metadata never launches."""
from pathlib import Path
import pytest
from . import run
from .test_experiment import Backend, references, stream
from .experiment import EXPECTED_STREAMS


def test_clean_qualification_then_full_bounded_campaign(tmp_path):
    streams = [stream(name) for name in EXPECTED_STREAMS]
    refs = {s['stream_id']: references(s) for s in streams}
    created = []
    def factory(config):
        value = Backend(config['agreement_arm'])
        created.append((dict(config), value))
        return value
    result = run.run_qualified_campaign(streams, refs, factory, tmp_path, config={}, fixtures={'status': 'passed'})
    assert result['status'] == 'complete'
    assert len(created) == 28 and len({id(x[1]) for x in created}) == 28
    assert all(x[0]['agreement_qualification'] for x in created[:4])
    assert all(not x[0]['agreement_qualification'] for x in created[4:])
    assert len({x[0]['output_root'] for x in created}) == 28
    assert all(x[1].training_state == 2 for x in created)
    assert len(list((tmp_path / 'campaign').glob('*/RESULT.json'))) == 6


def test_failed_qualification_does_not_start_campaign(tmp_path):
    source = stream(); created = []
    def factory(config):
        created.append(config)
        return Backend(config['agreement_arm'], qualifier=False)
    result = run.run_qualified_campaign([source], {source['stream_id']: references(source)}, factory,
        tmp_path, config={}, fixtures={'status': 'passed'})
    assert result['status'] == 'qualification_failed' and result['campaign'] is None
    assert not (tmp_path / 'campaign').exists()
    assert [x['agreement_arm'] for x in created] == ['H', 'L', 'G']


def test_failed_fixture_cannot_construct_backend(tmp_path):
    with pytest.raises(ValueError):
        run.run_qualified_campaign([], {}, lambda config: pytest.fail('unexpected worker'), tmp_path,
                                   config={}, fixtures={'status': 'failed'})
