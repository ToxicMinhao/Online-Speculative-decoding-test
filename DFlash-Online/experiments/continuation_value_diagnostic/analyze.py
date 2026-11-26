"""Recompute the sealed fixed-block analysis and publish token-free tables."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from online_feedback.storage import write_json
from .metrics import paired_block, legacy_pairs, coverage_table
from .statistics import analyze


def csv_table(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def sanitized(value, identifiers):
    if isinstance(value,dict):
        return {('response_ordinal' if k=='record_id' else k):sanitized(v,identifiers) for k,v in value.items()}
    if isinstance(value,list):return [sanitized(v,identifiers) for v in value]
    if isinstance(value,str) and value in identifiers:return identifiers[value]
    return value


def collect_rows(contract, root):
    rows=[]
    for binding in contract['features']:
        directory=root/'primary'/f'gpu{binding["physical_shard"]}'
        receipts=[json.loads((directory/endpoint/f'{binding["ordinal"]:03d}.json').read_text())
                  for endpoint in ('parent','child')]
        for receipt in receipts:
            if receipt['record_id']!=binding['record_id'] or receipt['anchors']!=binding['anchors']:
                raise ValueError('Native response readout differs from sealed identity/anchors')
            if len(receipt['blocks'])!=len(binding['anchors']):
                raise ValueError('Native readout lost a selected block')
        for anchor,parent,child in zip(binding['anchors'],receipts[0]['blocks'],receipts[1]['blocks']):
            rows.append(paired_block(parent,child,record_id=binding['record_id'],anchor=anchor))
    return rows


def run(root, public):
    started=time.perf_counter()
    contract=json.loads((root/'CONTRACT.json').read_text())
    status=json.loads((root/'STATUS.json').read_text())
    if status['status']!='completed':raise ValueError('Do not fit an incomplete primary cohort')
    rows=collect_rows(contract,root)
    identifiers={row['record_id']:row['ordinal'] for row in contract['features']}
    write_json(root/'BLOCK_READOUTS.json',rows)
    results=analyze(rows,correctness_passed=True)
    write_json(root/'ANALYSIS_PRIVATE.json',results)
    parent=json.loads((root/'legacy/parent.json').read_text())
    child=json.loads((root/'legacy/child.json').read_text())
    legacy={name:legacy_pairs(parent,child,name) for name in ('new_buffer','parent_training')}
    write_json(root/'LEGACY_BLOCKS.json',legacy)
    public.mkdir(parents=True,exist_ok=True)
    exposure=coverage_table(rows,list(identifiers))
    for row,binding in zip(exposure,contract['features']):
        row.update(available_first_pass_anchors=binding['available_first_pass_anchors'],
                   selected_anchors=len(binding['anchors']))
    csv_table(public/'EXPOSURE.csv',sanitized(exposure,identifiers))
    csv_table(public/'PER_RESPONSE.csv',sanitized(results['per_response'],identifiers))
    csv_table(public/'INFLUENCE.csv',sanitized(results['influence'],identifiers))
    write_json(public/'BOOTSTRAP.json',sanitized(results['bootstrap'],identifiers))
    write_json(public/'MODEL_RESULTS.json',sanitized({k:v for k,v in results.items()
        if k not in ('per_response','influence','bootstrap')},identifiers))
    decomposition=[]
    outcome_keys=('raw_repair','useful_repair','prefix_damaged','positive_progress','lost_prefix',
                  'reachable_origin_followers','damaged_prefix_positions','surviving_origin_followers',
                  'uncapped_progress','Y')
    for cohort,current in [('primary_all',rows),('primary_HIGH',[r for r in rows if r['high']]),
                           *[('legacy_'+name,data) for name,data in legacy.items()]]:
        decomposition.append(dict(cohort=cohort,responses=len({r['record_id'] for r in current}),
                                  blocks=len(current),**{key:sum(r[key] for r in current) for key in outcome_keys}))
    csv_table(public/'OUTCOME_DECOMPOSITION.csv',decomposition)
    public_contract={k:contract[k] for k in ('schema','base_commit','execution_commit','selection','statistics',
        'limits','seed','block_size','projection_chunk_blocks','mode','model_precision','margin_arithmetic',
        'offline_reduction','primary_cohort','qualification','budget')}
    public_contract.update(sealed_private_contract_identity=dict(bytes=(root/'CONTRACT.json').stat().st_size,
        sha256=hashlib.sha256((root/'CONTRACT.json').read_bytes()).hexdigest()),
        endpoints={k:dict(identity=v['identity'],counters=v['counters']) for k,v in contract['endpoints'].items()},
        features=[dict(response_ordinal=r['ordinal'],feature_identity=r['feature_identity'],
                       anchors_sha256=hashlib.sha256(json.dumps(r['anchors'],separators=(',',':')).encode()).hexdigest(),
                       anchors=len(r['anchors'])) for r in contract['features']])
    write_json(public/'CONTRACT.json',public_contract)
    write_json(public/'IDENTITIES.json',dict(execution_commit=contract['execution_commit'],
        source_identity=contract['source_identity'],sealed_private_inputs_identity=dict(
            bytes=(root/'IDENTITIES.json').stat().st_size,
            sha256=hashlib.sha256((root/'IDENTITIES.json').read_bytes()).hexdigest()),
        endpoints=public_contract['endpoints'],features=public_contract['features'],
        private_data_route='Authenticated PR60 shared event-2 packets and archived first-pass RESULT rows; exact paths in private CONTRACT'))
    write_json(root/'ANALYSIS_TIMING.json',dict(cpu_analysis_wall_s=time.perf_counter()-started))
    print(json.dumps(dict(decision=results['decision'],primary=results['primary']),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    args=p.parse_args();run(args.root,args.public)


if __name__=='__main__':main()
