"""Offline Point1 publication contains scalar evidence, never response content."""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.frontier_preservation_pilot.point1_export import export, write_json


def readout(endpoint, value=2.0, record='PRIVATE_RECORD_SENTINEL'):
    block = dict(origin=dict(L=1,e=2,H=1,K=2,m=3,labels=[987654321],valid=[True],prefix=[True]),
                 current=dict(L=1), proposals=[987654321], positions=[
                     dict(position=1,canonical=987654321,assessment_contribution_fp32=value,
                          margin_tie=False,wrong_rival_tie=False),
                     dict(position=2,canonical=987654322,assessment_contribution_fp32=value/2,
                          margin_tie=False,wrong_rival_tie=False)],
                 metrics=dict(raw_repair=0,useful_repair=0,damaged_prefix_positions=0,
                              damaged_prefix_blocks=0,surviving_followers=0,reachable_followers=0,
                              matching_prefix_length=1,fixed_K_shortfall=1,assessment=value*1.5),
                 assessment_sum_fp64=value*1.5)
    row = dict(record_id=record,anchors=[7654321],blocks=[block],nonempty_blocks=1,
               response_assessment_fp32=value*1.5,response_assessment_fp64=value*1.5,
               existing_block_reduced_assessment=value*1.5)
    return dict(status='passed',endpoint=endpoint,responses=[row],state_preserved=True,
                draft_forwards=1,head_calls=1,mean_response_assessment_fp32=value*1.5,
                mean_response_assessment=value*1.5,mean_response_assessment_fp64=value*1.5)


def vector(reference='linear', candidate='finite', a=1.0, b=2.0, projection=0.5):
    row = dict(reference=reference,candidate=candidate,reference_norm=a,candidate_norm=b,
               difference_norm=abs(a-b),dot=a*b,cosine=1.0,relative_difference=abs(a-b)/a,
               relative_difference_denominator='norm('+reference+')',zero_status='both_nonzero',
               differing_coordinates=1,coordinates=2,reference_signed_projection=projection,
               candidate_signed_projection=projection*2,difference_signed_projection=projection,
               projection_definition='-dot_FP64(cotangent, vector)',products_and_reductions='FP64')
    row['per_layer']={'layers.0':dict(row)}
    return row


def source(root):
    native=root/'attempt_001/native'
    native.mkdir(parents=True)
    plan=dict(schema='point1_update_fidelity_v1',execution_commit='a'*40,status='admitted',
              witness=dict(status='authenticated',set_name='new_buffer',responses=8,
                           items=[dict(record_id='PRIVATE_RECORD_SENTINEL',anchors=[7654321],
                                       feature_path='/PRIVATE_PATH_SENTINEL',feature_identity=dict(bytes=7,sha256='b'*64))]),
              parent_features=[dict(record_id='PRIVATE_RECORD_SENTINEL',path='/PRIVATE_PATH_SENTINEL',
                                   identity=dict(bytes=6,sha256='c'*64))],
              parent_recovery_identity=dict(bytes=8,sha256='d'*64),limits={'attempted_adam_calls':6})
    write_json(root/'CONTRACT.json',plan)
    write_json(native/'RESULT.json',dict(status='completed',execution_commit='a'*40,
        final_restoration=dict(status='passed',exact_resume=True),wall_s=3.0,
        startup=dict(status='completed',optimizer_steps=0,wall_s=1.0,record_id='PRIVATE_RECORD_SENTINEL')))
    write_json(root/'attempt_001/PROCESS.json',dict(status='completed',elapsed_s=4.0))
    write_json(root/'attempt_001/ADMISSION.json',dict(admitted=True))
    write_json(root/'CLOCK.json',dict(completed_occupied_card_s=4.0))
    write_json(root/'STEP_LEDGER.json',dict(schema='point1_native_step_ledger_v1',calls=[],
        attempts=[dict(number=1,status='completed',correction_note=None)],maximum_attempted_calls=6))
    write_json(native/'WORK.json',dict(attempted={},completed={},pending=[],native_adam_steps=4))
    def score(value):
        return dict(raw=dict(total=value,positive=value+1,negative=-1,absolute=value+2),
                    beta_centered=dict(total=value,positive=value+1,negative=-1,absolute=value+2),
                    raw_minus_centered=0,mass_offset_contribution=0,max_absolute_pool_mass_residual=0,
                    prefix_occurrences=2,nonempty_pools=1)
    write_json(native/'SCORING.json',dict(predictions={'F-C':[score(0.2),score(0.3)],
                                                       'F-BETA':[score(0.5),score(0.6)]}))
    chains={}
    for name in ('F-C','F-BETA','C-BETA'):
        chains[name]={key:vector() for key in ('linear_to_finite','finite_to_cast','cast_to_native','native_gradient','native_point_prediction')}
    write_json(native/'VECTOR_CHAIN.json',chains)
    write_json(native/'NATIVE_POINT_PROBE.json',dict(window_assessment=1.0))
    for name in ('BETA','C','F','C_REPEAT'):
        write_json(native/f'{name}.json',dict(status='completed',adam_attempts=1,adam_completed=1,
            beta_g0_exact=True,exact_C_repeat=True,rows=[dict(record_id='PRIVATE_RECORD_SENTINEL')]))
        for suffix in ('INITIAL_STATE','RESTORATION'):
            write_json(native/f'{name}_{suffix}.json',dict(status='passed',exact_resume=True,mismatches=[]))
    for name in ('AFTER_STARTUP','FINAL_RESTORATION'):
        write_json(native/f'{name}.json',dict(status='passed',exact_resume=True,mismatches=[]))
    for endpoint,value in [('parent',2.0),('Q_BETA',1.8),('Q_C',1.7),('Q_F',1.6),('N_BETA',1.9),('N_C',1.5),('N_F',1.0)]:
        write_json(native/f'window_{endpoint}.json',readout(endpoint,value))
        if not endpoint.startswith('Q_'):
            write_json(native/f'witness_{endpoint}.json',readout(endpoint,value,record='PRIVATE_WITNESS_SENTINEL'))
    return native


class Point1ExportTests(unittest.TestCase):
    def test_complete_export_arithmetic_and_private_content(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)/'raw'
            source(root)
            output=Path(folder)/'public'
            result=export(root,output)
            self.assertTrue(result['complete'])
            text='\n'.join(path.read_text() for path in output.iterdir())
            for private in ('PRIVATE_RECORD_SENTINEL','PRIVATE_WITNESS_SENTINEL','PRIVATE_PATH_SENTINEL','987654321','7654321'):
                self.assertNotIn(private,text)
            with (output/'CONTRASTS.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            score=next(row for row in rows if row['contrast']=='F-C' and row['stage']=='score_prediction')
            self.assertEqual(float(score['value']),0.5)
            self.assertEqual(score['role'],'primary')
            derived=next(row for row in rows if row['contrast']=='C-BETA' and row['stage']=='score_prediction')
            self.assertAlmostEqual(float(derived['value']),0.6)
            functional=next(row for row in rows if row['contrast']=='F-C' and row['stage']=='native_functional_benefit')
            self.assertEqual(float(functional['value']),0.75)
            self.assertEqual(float(functional['positive'])+float(functional['negative']),0.75)
            predictions=json.loads((output/'PREDICTION_ERRORS.json').read_text())
            self.assertEqual(predictions[0]['absolute_prediction_error'],0.25)
            self.assertEqual(predictions[0]['evaluation_point'],'Q_BETA')
            manifest=json.loads((output/'PUBLIC_FILES.json').read_text())
            for name,binding in manifest['files'].items():
                data=(output/name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(),binding['sha256'])
                self.assertEqual(len(data),binding['bytes'])
            audit=json.loads((output/'EXPORT_AUDIT.json').read_text())
            self.assertEqual(audit['readout_count'],11)
            self.assertFalse(audit['missing_files'])

    def test_incomplete_work_stays_partial(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)/'raw'
            native=source(root)
            (native/'VECTOR_CHAIN.json').unlink()
            (native/'window_N_F.json').unlink()
            write_json(native/'RESULT.json',dict(status='failed',execution_commit='a'*40))
            write_json(root/'attempt_001/PROCESS.json',dict(status='failed',elapsed_s=2.0))
            output=Path(folder)/'public'
            result=export(root,output)
            self.assertFalse(result['complete'])
            audit=json.loads((output/'EXPORT_AUDIT.json').read_text())
            self.assertEqual(audit['status'],'partial_receipts_exported')
            self.assertIn('attempt_001/native/window_N_F.json',audit['missing_files'])
            self.assertEqual(audit['readout_count'],10)

    def test_wrong_label_cannot_be_published(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)/'raw'
            native=source(root)
            write_json(native/'window_N_F.json',readout('N_C'))
            with self.assertRaisesRegex(ValueError,'label'):
                export(root,Path(folder)/'public')

    def test_never_overwrite_publication(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)/'raw'
            source(root)
            output=Path(folder)/'public'
            output.mkdir()
            with self.assertRaises(FileExistsError):
                export(root,output)


if __name__=='__main__':
    unittest.main()
