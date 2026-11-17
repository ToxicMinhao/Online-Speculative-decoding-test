"""Orchestration tests: CPU/native gates precede the single bounded chain."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation import launch


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.plan=dict(root=str(self.root),source_root=str(self.root/'source'),
            execution_commit='a'*40,training_python='/python',compiler_root='/cache',
            devices={'A':'physical_A','B':'physical_B'},estimated_seconds={
                'qualification_training':10.,'qualification_serving':10.,'C7':10.,'C_collect':10.,'C8':10.,'final':10.})
        self.path=self.root/'PLAN.json';write_json(self.path,self.plan)

    def execute(self, *, parallel=True, failure=None, pending=False):
        calls=[]
        def cpu_check(*args,**kwargs):
            write_json(self.root/'CPU_PREFLIGHT.json',dict(status='passed',parallel_qualification_admitted=parallel))
        def children(path,plan,jobs):
            names=[job[0] for job in jobs];calls.append(names)
            if failure in names:raise RuntimeError('retained worker failure')
            for name in names:
                if name in launch.TASKS:
                    write_json(self.root/'tasks'/name/'STATUS.json',dict(status='completed'))
            if pending and 'final' in names:
                work=json.loads((self.root/'WORK.json').read_text());work['pending']={'1':{'counts':{'adam':1}}}
                write_json(self.root/'WORK.json',work)
        with ExitStack() as stack:
            for target in ['check_compilers','child_environment']:
                stack.enter_context(patch.object(launch,target,return_value={}))
            stack.enter_context(patch.object(launch.subprocess,'check_output',return_value='100 /cache\n'))
            stack.enter_context(patch.object(launch.subprocess,'run',side_effect=cpu_check))
            stack.enter_context(patch.object(launch,'run_children',side_effect=children))
            stack.enter_context(patch.object(launch,'require_time_for',return_value={'status':'passed'}))
            stack.enter_context(patch.object(launch,'boundary',return_value={'status':'passed'}))
            stack.enter_context(patch.object(launch,'qualify_resources',return_value={'status':'passed','timing':{}}))
            try:result=launch.run(self.path)
            except RuntimeError:return calls,json.loads((self.root/'LAUNCH.json').read_text())
        return calls,result

    def test_native_gates_then_complete_order_with_parallel_qualification(self):
        calls,result=self.execute()
        self.assertEqual(calls,[['training_A','serving_B'],['serving_A'],['C7'],['C_collect'],['C8'],['final']])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['completed'],['cpu_preflight','qualification','C7','C_collect','C8','final'])

    def test_sequential_host_policy_is_obeyed(self):
        calls,result=self.execute(parallel=False)
        self.assertEqual(calls[:3],[['training_A'],['serving_B'],['serving_A']])
        self.assertEqual(result['status'],'completed')

    def test_failed_qualification_stops_without_production_or_retry(self):
        calls,result=self.execute(failure='training_A')
        self.assertEqual(calls,[['training_A','serving_B']])
        self.assertEqual(result['status'],'stopped')
        self.assertEqual(result['completed'],['cpu_preflight'])
        with self.assertRaises(FileExistsError):launch.run(self.path)
        self.assertFalse((self.root/'tasks').exists())

    def test_failed_production_stops_remaining_chain(self):
        calls,result=self.execute(failure='C_collect')
        self.assertEqual(calls[-1],['C_collect'])
        self.assertNotIn(['C8'],calls)
        self.assertNotIn(['final'],calls)
        self.assertEqual(result['status'],'stopped')
        self.assertIn('C7',result['completed'])

    def test_unknown_end_work_is_not_completed(self):
        calls,result=self.execute(pending=True)
        self.assertEqual(calls[-1],['final'])
        self.assertEqual(result['status'],'stopped')
        self.assertIn('unresolved work',result['error'])

    def test_failed_child_stops_owned_peers_and_preserves_receipts(self):
        (self.root/'logs').mkdir()
        failed=Mock(pid=111,returncode=1);failed.poll.return_value=1
        peer=Mock(pid=222,returncode=-15);peer.poll.return_value=None
        with patch.object(launch.subprocess,'Popen',side_effect=[failed,peer]) as popen, \
             patch.object(launch,'child_environment',return_value={}), \
             patch.object(launch,'clock_status',return_value={'occupied_s':1,'elapsed_s':1}), \
             patch.object(launch,'stop_owned') as stop:
            with self.assertRaises(RuntimeError):
                launch.run_children(self.path,self.plan,[launch.qualification_job('training','A'),launch.qualification_job('serving','B')])
        self.assertEqual(popen.call_count,2)
        self.assertEqual(stop.call_count,2)
        self.assertEqual(json.loads((self.root/'processes/training_A.json').read_text())['exit_code'],1)
        self.assertEqual(json.loads((self.root/'processes/serving_B.json').read_text())['exit_code'],-15)


if __name__=='__main__':unittest.main()
