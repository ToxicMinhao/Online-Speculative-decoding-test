import unittest
from copy import deepcopy
from audit_work import analyze_ledger, compare_summary, verify_full_postflight, LIMITS, COMMIT


def start():
    return [dict(kind='contract',limits=LIMITS.copy(),gpu_seconds=14400),
            dict(kind='gpu_start',occupied_gpus=1)]


def attempt(ticket,kind='draft_block',at=1,**metadata):
    return dict(kind=kind,status='attempted',ticket=ticket,gpu_seconds=at,metadata=metadata)


def finish(ticket,kind='draft_block',at=2,status='completed'):
    return dict(kind=kind,status=status,ticket=ticket,gpu_seconds=at,measurements={})


class WorkLedgerTests(unittest.TestCase):
    def test_failed_unknown_and_startup_work_remain_charged(self):
        rows=start()+[
            attempt(1,at=1,operation='startup_draft',startup=True),finish(1,at=2),
            attempt(2,at=3),finish(2,at=4,status='failed'),attempt(3,at=5),
            dict(kind='gpu_end',gpu_seconds=6)]
        result=analyze_ledger(rows)
        self.assertEqual(result['counts']['attempted'],{'draft_block':3})
        self.assertEqual(result['counts']['completed'],{'draft_block':1})
        self.assertEqual(result['counts']['failed'],{'draft_block':1})
        self.assertEqual(result['counts']['unknown'],{'draft_block':1})
        self.assertEqual(result['remaining']['draft_block'],32765)
        self.assertEqual(sum(r.get('completed',0) for r in result['operations'] if r['startup']),1)

    def test_context_tokens_and_nested_time_do_not_become_distinct_work(self):
        rows=start()+[
            attempt(1,'original_completion',1),
            attempt(2,'context_forward',2,operation='native_target_prefill',tokens=12),
            attempt(3,'target_call',3,operation='native_target_prefill'),
            finish(3,'target_call',4),finish(2,'context_forward',5),
            finish(1,'original_completion',6),dict(kind='gpu_end',gpu_seconds=7)]
        result=analyze_ledger(rows)
        self.assertEqual(result['context_tokens'],{'native_target_prefill':{'completed':12}})
        self.assertEqual(result['gpu_seconds'],7)
        self.assertEqual(len(result['duration_groups']),3)
        self.assertNotIn('total_model_seconds',result)

    def test_duplicate_or_wrong_kind_finish_fails(self):
        cases=[start()+[attempt(1),finish(1),finish(1,at=3)],
               start()+[attempt(1),finish(1,'target_call')]]
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaisesRegex(ValueError,'finish'):
                analyze_ledger(rows)

    def test_cap_forbidden_and_clock_fail(self):
        overrun=start()+[attempt(i,'original_completion',i) for i in range(1,258)]
        bad_clock=start()+[attempt(1,at=2),finish(1,at=1)]
        forbidden=start()+[attempt(1,'optimizer_step')]
        for rows in (overrun,bad_clock,forbidden):
            with self.subTest(last=rows[-1]),self.assertRaises(ValueError):
                analyze_ledger(rows)

    def test_reported_work_cannot_hide_a_failed_attempt(self):
        result=analyze_ledger(start()+[attempt(1),finish(1,status='failed'),
                                      dict(kind='gpu_end',gpu_seconds=3)])
        summary={**result['counts'],'remaining':result['remaining'],
                 'gpu_seconds':3,'gpu_seconds_limit':14400}
        compare_summary(result,summary,final=True)
        summary={**summary,'failed':{}}
        with self.assertRaisesRegex(ValueError,'failed'):
            compare_summary(result,summary,final=True)

    def test_empty_context_metadata_is_not_silently_zero(self):
        rows=start()+[attempt(1,'context_forward'),finish(1,'context_forward'),
                      dict(kind='gpu_end',gpu_seconds=3)]
        with self.assertRaisesRegex(ValueError,'context-token'):
            analyze_ledger(rows)

    def test_full_postflight_authenticates_actual_hashes_not_boolean_labels(self):
        preflight = dict(code_root='/source', artifact_checks=[dict(path='/weights',sha256='a')],
                         source_files={'code.py':dict(bytes=10,sha256='b')})
        receipt = dict(status='passed',actual_commit=COMMIT,expected_execution_commit=COMMIT,
                       working_tree='',new_model_work=0,wall_s=1,
                       checks=[dict(path='/weights',matches=True,actual=dict(bytes=99,sha256='a')),
                               dict(path='/source/code.py',matches=True,actual=dict(bytes=10,sha256='b'))])
        self.assertEqual(verify_full_postflight(preflight,receipt)['total_checks'],2)
        changed = deepcopy(receipt)
        changed['checks'][0]['actual']['sha256'] = 'different'
        missing = deepcopy(receipt)
        missing['checks'].pop()
        duplicate = deepcopy(receipt)
        duplicate['checks'].append(deepcopy(duplicate['checks'][0]))
        for invalid in (changed,missing,duplicate):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                verify_full_postflight(preflight,invalid)


if __name__=='__main__':
    unittest.main()
