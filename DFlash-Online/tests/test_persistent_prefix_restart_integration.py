"""The restarted supervisor must execute only the remaining scientific tasks."""
from pathlib import Path
import io
import unittest
from unittest.mock import patch
import test_persistent_prefix_production_restart as fixture_module
from experiments.persistent_prefix_withholding import production_restart, campaign, report, resources
from online_feedback.storage import write_json


class RestartIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ProductionRestartTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.plan
        self.root = self.fixture.root
        self.proof = production_restart.import_restart(self.plan)
        self.plan.update(costs={k:1. for k in campaign.REQUIRED_COSTS},
            task_memory_bytes={k:1 for k in ('training','collection','reference','final')})
        self.path = self.root/'sealed/PLAN.json'
        write_json(self.path,self.plan)

    def test_only_remaining_tasks_run_and_old_unknown_forward_stays_charged(self):
        started=[]
        per_task = {
            'S8':dict(adam=128,response_forward=258,response_backward=258,training_initialization=1),
            'references':dict(reference_generation=24),
            'final':dict(warmup_generation=72,scored_generation=1152)}
        class Process:
            pid=0
            def poll(self):return 0
        def start(path,plan,task,uuid):
            started.append((task.name,uuid))
            work=campaign.read(self.root/'WORK.json')
            for key,amount in per_task[task.name].items():
                work['attempted'][key]+=amount;work['completed'][key]+=amount
            write_json(self.root/'WORK.json',work)
            write_json(self.root/'tasks'/task.name/'STATUS.json',dict(status='completed'))
            return dict(process=Process(),log=io.StringIO(),task=task,uuid=uuid,started_epoch=0)
        clock=dict(occupied_s=2849.2045271396637,elapsed_s=11000,active={})
        with patch.object(campaign,'mandatory_gates'), patch.object(campaign,'clock_status',return_value=clock), \
             patch.object(campaign,'current_memory',return_value=200*2**30), \
             patch.object(campaign,'process_tree_rss',return_value=0), \
             patch.object(campaign,'storage_boundary'), \
             patch.object(campaign,'memory_admission',return_value={'status':'passed'}), \
             patch.object(campaign,'gpu_admission',return_value={'status':'passed'}), \
             patch.object(campaign,'start_task',side_effect=start), patch.object(campaign.time,'sleep'):
            result=campaign.run_campaign(self.path)
        self.assertEqual({x[0] for x in started[:2]},{'S8','references'})
        self.assertEqual(started[2][0],'final')
        self.assertEqual(started[-1][1],self.plan['devices']['A'])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['work']['attempted'],self.proof['expected_attempted'])
        self.assertEqual(result['work']['completed'],self.proof['expected_completed'])
        self.assertEqual(result['work']['inherited_pending'],self.proof['failed_pending'])
        accounting=report.work_accounting(self.root,self.plan)
        self.assertTrue(accounting['accounting_complete'])
        self.assertEqual(next(x for x in accounting['rows'] if x['kind']=='response_forward')['unknown'],1)

    def test_new_pending_work_still_blocks_completion(self):
        work=campaign.read(self.root/'WORK.json')
        work.update(attempted=self.proof['expected_attempted'],completed=self.proof['expected_completed'],pending={'new':{'counts':{'response_forward':1}}})
        write_json(self.root/'WORK.json',work)
        self.assertFalse(report.work_accounting(self.root,self.plan)['accounting_complete'])

    def test_elapsed_extension_does_not_change_card_limit_or_original_epoch(self):
        self.assertEqual(resources.effective_elapsed_limit({}),14400)
        self.assertEqual(resources.effective_elapsed_limit(self.plan),21600)
        self.assertEqual(campaign.read(self.root/'CLOCK.json')['first_model_load_epoch'],1794802603.911516)
        result=resources.time_admission(occupied_spent_s=14000,remaining_components={'work':1000},
            elapsed_spent_s=14000,remaining_schedule_s=1000,elapsed_cap_s=21600)
        self.assertEqual(result['status'],'blocked')
        self.assertEqual(result['card_cap_s'],14400)
        self.plan['elapsed_time_amendment']['occupied_card_cap_s']=21600
        with self.assertRaises(ValueError):resources.effective_elapsed_limit(self.plan)


if __name__=='__main__':unittest.main()
