"""Flatten retained production geometry and exposure statistics; no model imports."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct

from analyze_production import distribution, write_csv, write_json


def export(roots, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    windows=[];responses=[];presentations=[];exposure=[];reconciliations=[];inputs={}
    for dataset,root in roots.items():
        for event in (5,6):
            for arm in ('KPLUS','FSG','SCALAR'):
                relative=f'campaign_001/{dataset}/event_{event:03d}/{arm}/RESULT.json'
                data=(Path(root)/relative).read_bytes();result=json.loads(data)
                inputs[dataset+':'+relative]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                assert len(result['records'])==256
                prefix=dict(dataset=dataset,event=event,arm=arm)
                totals=result['selection_totals']
                for key,value in totals.items():
                    values=[1. if key=='response_presentations' else r['response_observation']['sum_metrics'][key]
                            for r in result['records']]
                    # TrainerCore adds detached scalar tensors at each two-response
                    # boundary, then Python sums boundary metrics over the event.
                    pair_totals=[struct.unpack('f',struct.pack('f',values[i]+values[i+1]))[0]
                                 for i in range(0,256,2)]
                    assert sum(pair_totals)==value,(dataset,event,arm,key)
                    reconciliations.append(dict(**prefix,metric=key,retained_boundary_total=value,
                        reproduced_fp32_pair_total=sum(pair_totals),separately_summed_response_total=math.fsum(values),
                        response_minus_boundary=math.fsum(values)-value))
                exposure.append(dict(**prefix,**totals,
                    selected_occurrence_fraction=totals['selected_position_occurrences']/totals['available_position_occurrences'],
                    selected_weight_fraction=totals['selected_position_weight']/totals['available_position_weight']))
                for index,r in enumerate(result['records']):
                    observation=r['response_observation']
                    assert observation['scope']=='one_original_response_forward'
                    assert observation['presentation_ordinal']==index
                    presentations.append(dict(**prefix,presentation=index,pass_index=r['pass'],
                        global_step_after=r['optimizer_step_after'],optimizer_stepped=r['optimizer_stepped'],
                        active_anchors=r['active_anchors'],prompt_tokens=r['prompt_tokens'],response_tokens=r['response_tokens'],
                        observed_loss=observation['loss'],ratio_metrics=observation['ratio_metrics'],
                        **observation['sum_metrics']))
                if arm=='KPLUS':continue
                for row in result['objective_work']['functional_geometry']['steps']:
                    identity=dict(**prefix,global_step=row['global_step'],window=row['global_step']-(event-1)*128)
                    sketch=row.get('sketch',{})
                    flat=dict(**identity,active_tensors=row['active_tensors'],active_coordinates=row['active_coordinates'],
                        preclip_gradient_norm=row['preclip_gradient_norm'],clip_threshold=1.0,
                        preclip_norm_exceeds_threshold=row['preclip_gradient_norm']>1.0,
                        **{k:v for k,v in sketch.items() if k!='responses'})
                    for name in ('native','raw_fsg','feasible_fsg','candidate','committed'):
                        for key,value in row.get(name,{}).items():
                            flat[name+'_'+key]=value
                    for name in ('native_risk','candidate_risk'):
                        for key in ('value','raw','eta','absolute_sum','negative_roundoff'):
                            flat[name+'_'+key]=row.get(name,{}).get(key)
                    flat['decision']=row['decision'];flat['realization']=row.get('realization')
                    flat['scalar']=row.get('scalar')
                    windows.append(flat)
                    for index,r in enumerate(row.get('responses',[])):
                        gradient=next((v for v in sketch.get('responses',[]) if v['response']==index),None)
                        responses.append(dict(**identity,response=index,**r,
                            auxiliary_c=None if gradient is None else gradient['c'],
                            auxiliary_gradient_norm=None if gradient is None else gradient['gradient_norm'],
                            auxiliary_gradient_dot_native=None if gradient is None else gradient['gradient_dot_native'],
                            auxiliary_gradient_dot_native_over_c=None if gradient is None else gradient['gradient_dot_native_over_c']))
    summaries=[]
    quantities=('preclip_gradient_norm','sigma','condition_number','solve_residual','native_projected_energy',
                'cross_response_cosine','raw_fsg_changed_zero_native_coordinates','feasible_fsg_changed_zero_native_coordinates',
                'committed_changed_zero_native_coordinates','committed_different_native_coordinates')
    for dataset in roots:
        for arm in ('FSG','SCALAR'):
            for event in (5,6,'combined'):
                subset=[r for r in windows if r['dataset']==dataset and r['arm']==arm and (event=='combined' or r['event']==event)]
                for key in quantities:
                    summaries.append(dict(dataset=dataset,arm=arm,event=event,variable=key,**distribution([r.get(key) for r in subset],len(subset))))
    write_csv(output/'GEOMETRY_TRACE.csv',windows);write_csv(output/'RESPONSE_GEOMETRY_TRACE.csv',responses)
    write_csv(output/'ORDINARY_PRESENTATIONS.csv',presentations);write_csv(output/'EXPOSURE_BY_EVENT.csv',exposure)
    write_csv(output/'GEOMETRY_DISTRIBUTIONS.csv',summaries);write_json(output/'INPUTS.json',inputs)
    write_csv(output/'EXPOSURE_RECONCILIATION.csv',reconciliations)
    receipt=dict(status='completed',filtered_windows=len(windows),filtered_response_rows=len(responses),
        ordinary_presentations=len(presentations),event_receipts=len(exposure),new_model_work=0,
        absent_quantities=['ordinary_loss_gradient_dot_candidate','ordinary_loss_gradient_dot_native'],
        interpretation='Auxiliary gradient products are not ordinary training-gradient products; occurrence counts are not unique tokens.',
        clip_threshold_provenance='sealed/*/config_*.yaml training.max_grad_norm=1.0; norm value recorded before clipping.')
    write_json(output/'RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gsm8k',required=True);p.add_argument('--mathqa',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();export(dict(gsm8k=a.gsm8k,mathqa=a.mathqa),a.output)
