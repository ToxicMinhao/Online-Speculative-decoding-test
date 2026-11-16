"""Admission is pure accounting over sealed metadata and native qualification."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments.persistent_prefix_withholding import admission
from experiments.persistent_prefix_withholding.campaign import REQUIRED_COSTS, task_graph
from experiments.persistent_prefix_withholding.resources import (
    ALLOCATION_BYTES, CARD_SECONDS, HOST_HEADROOM_BYTES, storage_plan,
)

GIB = 2**30


def fixtures():
    rates = dict(training_event=100., training_startup=2., training_load=3., publication=4.,
        feedback_response=.01, teacher_response=.02, reference_response=.03, serving_load=5.,
        scored_request=.04, warmup_request=.05, activation=.1, resident_wait=1.)
    basis = dict(rates=rates,
        data_scale=dict(training=1., feedback=1., reference=1., scored=1., warmup=1.),
        current_panel_metadata=dict(
            selected_historical_frozen=dict(output_tokens=dict(n=104,total=20800,p95=300.)),
            missing_fixed_K6_reference_count=24,
            event7_retained=dict(output_tokens=dict(mean=250.))))
    def train(pair, load, startup, host, gpu):
        return dict(status='passed', loading_s=load, startup_s=startup,
            sampled_process_tree_peak_rss_bytes=host, peak_gpu_reserved_bytes=gpu,
            qualification=dict(cases=[dict(pair_wall_excluding_preclip_observer_s=pair,
                                           pair_wall_s=pair+10.)]))
    training = dict(A=train(1.5,9.,6.,10*GIB,20*GIB), B=train(2.,10.,8.,11*GIB,21*GIB))
    def serve(load, latency, outputs, host, endpoint, peak):
        return dict(status='passed', loading_s=load,
            sampled_process_tree_peak_rss_bytes=host,
            gpu_memory_after_loading=dict(used_bytes=endpoint-1*GIB),
            gpu_memory_after_generations=dict(used_bytes=endpoint),
            graph_and_physical_work=dict(peak_gpu_reserved_bytes=peak,
                                         peak_gpu_allocated_bytes=peak-GIB),
            qualification=dict(rows=[dict(request_latency_s=latency,output_tokens=outputs)]*4))
    serving = dict(A=serve(15.,.8,100,8*GIB,30*GIB,32*GIB),
                   B=serve(20.,1.,100,9*GIB,31*GIB,33*GIB))
    plan = dict(historical_cost_basis='sealed-cost-basis.json',root='/unused/cpu-test',
                execution_commit='exact-reviewed-commit',cpu_gate='cpu-prefight.json')
    return plan,basis,training,serving


class AdmissionTests(unittest.TestCase):
    def run_admission(self, *, basis=None, training=None, serving=None, available=200*GIB,
                      occupied=50., elapsed=25.):
        plan, initial, initial_training, initial_serving = fixtures()
        basis = initial if basis is None else basis
        training = initial_training if training is None else training
        serving = initial_serving if serving is None else serving
        with patch.object(admission,'read',return_value=basis) as read, \
             patch.object(admission,'clock_status',return_value=dict(occupied_s=occupied,elapsed_s=elapsed)), \
             patch.object(admission,'current_memory',return_value=available):
            result=admission.qualify_resources(plan,training,serving)
        read.assert_called_once_with('sealed-cost-basis.json')
        return result

    def test_all_categories_and_worst_native_rates_are_priced(self):
        result=self.run_admission()
        self.assertEqual(set(result['costs']),REQUIRED_COSTS)
        costs=result['costs']
        self.assertEqual(costs['training_event'],256.)
        self.assertEqual(costs['training_startup'],8.)
        self.assertEqual(costs['training_load'],10.)
        self.assertEqual(costs['serving_load'],20.)
        self.assertEqual(costs['feedback_response'],2.5)
        self.assertEqual(costs['reference_response'],3.)
        self.assertEqual(costs['scored_request'],2.1875)
        self.assertEqual(costs['warmup_request'],1.)
        self.assertEqual(result['measured_worst_pair_compute_s'],2.)
        self.assertEqual(result['projected_panel_output_mean'],218.75)
        self.assertFalse(result['candidate_outcomes_used_for_pricing'])
        self.assertFalse(result['scientific_settings_changed'])
        self.assertEqual(result['status'],'passed')

    def test_prior_historical_maximum_is_not_replaced_by_faster_qualification(self):
        _,basis,_,_=fixtures()
        basis['rates'].update(training_event=700.,training_startup=90.,training_load=80.,
                              serving_load=70.,warmup_request=2.)
        result=self.run_admission(basis=basis)
        for key in ('training_event','training_startup','training_load','serving_load','warmup_request'):
            self.assertEqual(result['costs'][key],basis['rates'][key])

    def test_sealed_length_scales_apply_without_outcome_reselection(self):
        _,basis,_,_=fixtures()
        basis['rates'].update(feedback_response=5.,teacher_response=2.,reference_response=5.,
                              scored_request=5.,warmup_request=5.)
        basis['data_scale']=dict(training=2.,feedback=3.,reference=4.,scored=5.,warmup=6.)
        result=self.run_admission(basis=basis)
        self.assertEqual(result['costs']['training_event'],512.)
        self.assertEqual(result['costs']['feedback_response'],15.)
        self.assertEqual(result['costs']['teacher_response'],6.)
        self.assertEqual(result['costs']['reference_response'],20.)
        self.assertEqual(result['costs']['scored_request'],25.)
        self.assertEqual(result['costs']['warmup_request'],30.)

    def test_missing_or_nonfinite_component_cannot_be_admitted(self):
        for name in REQUIRED_COSTS:
            _,basis,_,_=fixtures()
            del basis['rates'][name]
            with self.subTest(missing=name),self.assertRaises((ValueError,KeyError)):
                self.run_admission(basis=basis)
        for invalid in (float('nan'),float('inf'),-1.):
            _,basis,_,_=fixtures()
            basis['rates']['publication']=invalid
            with self.subTest(value=invalid),self.assertRaises(ValueError):
                self.run_admission(basis=basis)

    def test_all_planned_tasks_enter_card_time_with_actual_spent_work(self):
        result=self.run_admission(occupied=123.,elapsed=45.)
        tasks=task_graph(result['costs'],24)
        remaining=sum(task.estimated_card_s for task in tasks)
        self.assertEqual(result['time']['raw_remaining_card_s'],remaining)
        self.assertEqual(result['time']['conservative_total_card_s'],123.+1.5*remaining+1800)
        self.assertEqual(result['time']['card_cap_s'],CARD_SECONDS)
        self.assertEqual(result['time']['elapsed_cap_s'],14400)
        self.assertEqual(len(tasks),10)

    def test_card_time_and_elapsed_caps_each_block(self):
        self.assertEqual(self.run_admission(occupied=14400.)['status'],'blocked')
        self.assertEqual(self.run_admission(elapsed=14400.)['status'],'blocked')

    def test_host_bounds_include_margins_and_global_reserve(self):
        result=self.run_admission()
        self.assertEqual(result['task_memory_bytes']['training'],23*GIB)
        self.assertEqual(result['task_memory_bytes']['collection'],13*GIB)
        self.assertEqual(result['task_vram_bytes']['training'],23*GIB)
        insufficient=self.run_admission(available=23*GIB+HOST_HEADROOM_BYTES-1)
        self.assertEqual(insufficient['status'],'blocked')

    def test_insufficient_parallel_ram_prices_serial_elapsed(self):
        result=self.run_admission(available=60*GIB)
        self.assertFalse(result['two_trainers_admitted'])
        self.assertFalse(result['mixed_training_serving_admitted'])
        remaining=sum(t.estimated_card_s for t in task_graph(result['costs'],24))
        self.assertEqual(result['time']['conservative_total_elapsed_s'],25.+1.5*remaining+1800)

    def test_two_native_workers_also_must_fit_parallel_schedule(self):
        _,_,training,serving=fixtures()
        for row in training.values():
            row['sampled_process_tree_peak_rss_bytes']=28*GIB  # +12 =>40
        for row in serving.values():
            row['sampled_process_tree_peak_rss_bytes']=76*GIB  # +4 =>80
        result=self.run_admission(training=training,serving=serving,available=155*GIB)
        self.assertTrue(result['two_trainers_admitted'])
        self.assertTrue(result['mixed_training_serving_admitted'])
        remaining=sum(t.estimated_card_s for t in task_graph(result['costs'],24))
        self.assertEqual(result['time']['conservative_total_elapsed_s'],25.+1.5*remaining+1800)

    def test_native_vram_bound_uses_peak_not_only_final_sample(self):
        result=self.run_admission()
        self.assertEqual(result['task_vram_bytes']['collection'],37*GIB)
        self.assertEqual(result['task_vram_bytes']['reference'],37*GIB)
        self.assertEqual(result['task_vram_bytes']['final'],37*GIB)

    def test_independent_storage_guard_keeps_110gb_cap_and_400gb_reserve(self):
        args=dict(weight_bytes=1_000_000,recovery_bytes=3_000_000,hidden_size=128,captured_layers=3)
        initial=storage_plan([100]*128,**args)
        fit=storage_plan([100]*128,**args,
            prior_allocation_bytes=ALLOCATION_BYTES-initial['peak_new_bytes'])
        over=storage_plan([100]*128,**args,
            prior_allocation_bytes=ALLOCATION_BYTES-initial['peak_new_bytes']+1)
        self.assertEqual(fit['status'],'passed')
        self.assertEqual(fit['peak_new_bytes'],110_000_000_000)
        self.assertEqual(fit['required_initial_free_bytes'],510_000_000_000)
        self.assertEqual(over['status'],'blocked')


if __name__=='__main__':
    unittest.main()
