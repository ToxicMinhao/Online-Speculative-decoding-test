"""Agreement-filter integration with the unchanged qualified native runner.

Import and planning are CPU-only. The reviewed CLI is the only GPU entry point.
All arms share the existing model admission, output observer, cold caches and
whole-publication timing. Independent worker lifetimes isolate arm states.
"""
from __future__ import annotations

import importlib.util
import copy
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
# Child workers execute this file directly, including under multiprocessing.
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from experiments.dflash_osd_agreement_filter import training

_spec = importlib.util.spec_from_file_location('_agreement_per_turn_native', HERE.parent / 'dflash_per_turn/native.py')
per_turn = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = per_turn
_spec.loader.exec_module(per_turn)
base, followup = per_turn.base, per_turn.followup
require, sha, write_json = base.require, base.sha, base.write_json
ARMS = ('H', 'L', 'G', 'N')


def effective_training_receipt(inherited, trainer, config, compiler_runtime):
    """Replace inherited eight-response metadata with the actual arm recipe."""
    effective = copy.deepcopy(inherited)
    effective.setdefault('args', {})['learning_rate'] = trainer.learning_rate
    grouped = config['agreement_arm'] in ('G', 'N')
    effective.update(
        recipe={**training.RECIPE, 'learning_rate': trainer.learning_rate},
        agreement_arm=config['agreement_arm'],
        agreement_source_sha256=config['agreement_source_sha256'],
        schedule='Newest actual response only; two passes and two committed optimizer steps',
        compiler_runtime=copy.deepcopy(compiler_runtime),
        qualification=config['agreement_qualification'],
        numerical_protocol=(training.GROUP_PROTOCOL['schema'] if grouped else
                            'qualified_ordinary_author_full_loss_gradient'),
        agreement_numerics=copy.deepcopy(training.core.NUMERICS),
        agreement_numerics_applies_to=('G', 'N'),
        ordinary_proposal_gradient='direct full-loss native autograd gradient',
        gradient_backward_calls_per_optimizer_step=3 if grouped else 1,
        gradient_backward_calls_per_event=6 if grouped else 2,
        gradient_sum_role='diagnostic only; unchanged historical tolerance is not an admission gate'
                          if grouped else 'not computed',
        sum_tolerance_is_admission_gate=False,
    )
    return effective


def direct_full_qualification_checks(result):
    """Validate v2 construction while retaining a failed historical sum check."""
    from experiments.dflash_osd_agreement_filter.gradient_diagnostics import SUM_ERROR
    presentations = [p for batch in result['accounting']['batches']
                     for p in batch['presentations']]
    require(len(presentations) == 2, 'Two direct-full response passes are required')
    diagnostics = []
    for presentation in presentations:
        require(presentation.get('ordinary_proposal_gradient') ==
                'direct full-loss native autograd gradient' and
                presentation.get('gradient_backward_calls') == 3 and
                presentation.get('numerical_protocol') == training.GROUP_PROTOCOL['schema'] and
                presentation.get('sum_tolerance_is_admission_gate') is False,
                'Direct full-gradient qualification receipt missing or changed')
        check = presentation.get('gradient_sum_diagnostic')
        require(isinstance(check, dict) and
                check.get('schema') == 'agreement_gradient_sum_diagnostic_v1' and
                check.get('status') in ('passed', 'failed') and
                check.get('nonfinite_input_coordinates') == 0,
                'Finite raw-gradient diagnostic missing or invalid')
        counts = check.get('nonfinite_counts', {})
        require(set(counts) == {'a', 'b', 'sum', 'reference', 'error', 'tolerance',
                               'error_over_tolerance'} and
                all(type(value) is int and value == 0 for value in counts.values()),
                'Nonfinite raw-gradient diagnostic')
        require(check['status'] != 'failed' or
                (check.get('first_failure') or {}).get('error') == SUM_ERROR,
                'Unexpected failed raw-gradient diagnostic')
        diagnostics.append(check)
    return dict(status='passed', numerical_protocol=training.GROUP_PROTOCOL['schema'],
                ordinary_proposal_gradient='direct full-loss native autograd gradient',
                gradient_backward_calls_per_optimizer_step=3,
                gradient_backward_calls_per_event=6,
                gradient_sum_diagnostics=diagnostics, sum_tolerance_is_admission_gate=False,
                scope='Direct-full construction checks; historical sum failures remain diagnostic')


def source_pins():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}


def check_sources(config):
    require(config.get('agreement_source_sha256') == source_pins(), 'Reviewed agreement-filter source changed')
    require(config.get('agreement_arm') in ARMS, 'Unknown independent learning arm')
    require(type(config.get('agreement_qualification')) is bool, 'Explicit qualification flag required')
    per_turn.check_sources(config)


def worker_command(config):
    command = per_turn._original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Native command interface changed')
    command[-1] = str(HERE / 'native.py')
    return command


base.worker_command = worker_command


class NativeBackend(per_turn.NativeBackend):
    def start_episode(self, episode_id):
        require(self.config.get('agreement_arm') in ARMS, 'Arm must be explicit')
        require(type(self.config.get('agreement_qualification')) is bool, 'Qualification scope must be explicit')
        require(self.config.get('remaining_diagnostic_cases', 0) == 0, 'No extra learning-rate diagnostic campaign')
        pins = source_pins()
        require('agreement_source_sha256' not in self.config or self.config['agreement_source_sha256'] == pins,
                'Configured agreement source differs')
        self.config.update(agreement_source_sha256=pins, remaining_diagnostic_cases=0)
        return super().start_episode(episode_id)

    def train_event(self, event, responses):
        require(event in (1, 2), 'Only two update events per arm')
        return super().train_event(event, responses)

    def diagnostic_event(self, *args, **kwargs):
        raise RuntimeError('Unplanned learning-rate branches are outside this pilot')

    def finalize_retention(self, *, result):
        """After scores are durable, retire only this completed arm's rolling bulk."""
        require(self.worker is None and self.trainer is None and self.episode_id is None,
                'Finish workers before final retention')
        require(result.get('status') == 'complete' and result.get('arm') == self.config['agreement_arm'] and
                result.get('cleanup', {}).get('status') == 'passed', 'Incomplete arm must retain recovery')
        require([x.get('turn') for x in result.get('canonical', [])] == [1, 2, 3] and
                [x.get('event') for x in result.get('updates', [])] == [1, 2] and
                all(x.get('mask_and_schedule_check') for x in result['updates']), 'Incomplete causal evidence')
        scores = result.get('scores', [])
        expected = 4 if result['arm'] == 'H' else 2
        require(len(scores) == expected and all(x.get('same_generation_valid') for x in scores),
                'Invalid paired scores must retain restart state')
        path = self.out / 'ARM_RESULT.json'
        require(not path.exists(), 'Preserve existing completion receipt')
        write_json(path, result)
        receipt = base.binding(path)
        require(json.loads(base.checked_binding(receipt).read_text()) == result, 'Completion evidence differs')
        retired = self.retire({'F'}, set(), _final=True)
        retired['completion_evidence'] = receipt
        retired['lost_capability'] = 'Exact completed-arm optimizer continuation and checkpoint replay; compact native traces, tokens, exposure and results retained.'
        write_json(self.out / 'FINAL_RETENTION.json', retired)
        return retired


class ServingService(per_turn.ServingService):
    def __init__(self, config):
        check_sources(config)
        super().__init__(config)


class TrainerService(per_turn.TrainerService):
    def __init__(self, config):
        check_sources(config)
        # Pin saved-buffer lifetime before inherited setup can compile a model.
        # Apply to ordinary H/L and grouped G/N, including the reference trial.
        compiler_runtime = training.configure_compiler_runtime()
        write_json(Path(config['worker_out']) / 'COMPILER_RUNTIME.json', compiler_runtime)
        # Build the same teacher, BF16 draft/head and source-pinned author setup.
        # Avoid the parent per-turn constructor's initial optimizer-state file.
        followup.TrainerService.__init__(self, config)
        self.trainer = training.AgreementTrainer(self.draft, self.components.embed_tokens,
            self.components.lm_head, self.teacher_features, episode_id=config['episode_id'], target_id='T0',
            target_identity_sha256=config['target']['identity_sha256'],
            released_checkpoint_sha256=config['released']['checkpoint_identity_sha256'], mask_token_id=128002,
            attention_backend='flex_attention', check_frozen=self.check_frozen,
            arm=config['agreement_arm'], qualification=config['agreement_qualification'])
        self.trainer.gradient_diagnostics_dir = self.out / 'gradient_checks'
        effective = effective_training_receipt(
            json.loads((self.out / 'AUTHOR_RUNTIME.json').read_text()), self.trainer, config, compiler_runtime)
        write_json(self.out / 'AUTHOR_RUNTIME.json', effective)
        self.states = self.out / 'states'; self.states.mkdir()
        self.updates = {}
        initial = self.trainer.save_state(self.states / 'initial.pt')
        self.current_state = {'state': initial, 'checkpoint': config['released']}
        self.qualification_receipt = None

    def train(self, body):
        require('operation' not in body and body.get('event') in (1, 2), 'Only declared chronological updates allowed')
        self._check_responses(body['responses'])
        if self.config['agreement_qualification'] and self.config['agreement_arm'] == 'G' and body['event'] == 1:
            require(self.qualification_receipt is None, 'Qualification must run once')
            started = time.perf_counter()
            try:
                self.qualification_receipt = self.trainer.qualify_disabled_filter(body['responses'])
            except BaseException:
                failed_receipt = getattr(self.trainer, 'last_qualification_receipt', None)
                if failed_receipt is not None:
                    self.qualification_receipt = copy.deepcopy(failed_receipt)
                    self.qualification_receipt['observed_wall_s'] = time.perf_counter() - started
                    write_json(self.out / 'DISABLED_FILTER_QUALIFICATION.json', self.qualification_receipt)
                raise
            self.qualification_receipt['observed_wall_s'] = time.perf_counter() - started
            write_json(self.out / 'DISABLED_FILTER_QUALIFICATION.json', self.qualification_receipt)
            require(self.qualification_receipt.get('status') == 'passed', 'Disabled-filter equivalence failed')
        result = super().train(body)
        result['agreement_arm'] = self.config['agreement_arm']
        if self.config['agreement_qualification'] and self.config['agreement_arm'] == 'G':
            require(self.qualification_receipt is not None and self.qualification_receipt['status'] == 'passed',
                    'Disabled-filter qualification receipt missing')
        if self.config['agreement_qualification'] and self.config['agreement_arm'] in ('G', 'N'):
            result['qualification_checks'] = direct_full_qualification_checks(result)
            if self.config['agreement_arm'] == 'G':
                result['qualification_checks']['disabled_filter_status'] = self.qualification_receipt['status']
        if self.qualification_receipt is not None and body['event'] == 1:
            result['disabled_filter_qualification'] = self.qualification_receipt
        write_json(self.out / ('UPDATE_%d.json' % body['event']), result)
        return result

    def diagnostic(self, body):
        raise RuntimeError('No additional training branches are authorized')


def worker_main():
    check_sources(base._worker_config())
    base._ServingService = ServingService
    base._TrainerService = TrainerService
    base.worker_main()


if __name__ == '__main__':
    require(sys.argv[1:] == ['--worker'], 'Use the reviewed agreement-filter CLI')
    worker_main()
