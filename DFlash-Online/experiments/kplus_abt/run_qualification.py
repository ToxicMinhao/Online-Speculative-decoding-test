"""Run precisely twelve disposable cases, two resource fixtures and eight replays."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.ok_coding.storage import identity
from experiments.ok_coding.campaign import require_complete_work
from online_feedback.storage import write_json
from .contract import ARMS, DATASETS, QUALIFICATION_CASES, QUALIFICATION_BUDGET
from .data import read
from .preflight import verify
from .storage import retire_exact


def retire_case(root, dataset, case, *, baseline=False):
    """Only successful disposable artifacts; failures retain all tensors."""
    output=root/'qualification'/dataset/case
    row=read(output/'RESULT.json')
    if row['status']!='completed':
        raise ValueError('Do not retire a failed qualification case')
    if case in ARMS and read(output/'SERVING_CHECK.json')['status']!='passed':
        raise ValueError('Disposable export must complete its declared native check first')
    candidates=[]
    if case!='KPLUS' or baseline:
        candidates.append((output/'ENDPOINT.pt',row['endpoint_identity'],'Completed disposable state/gradient comparison'))
    if case in ARMS and not baseline:
        for relative,expected in row['export_check']['files'].items():
            candidates.append((Path(row['export'])/relative,expected,'Disposable serving check completed'))
    return retire_exact(root,candidates,output/('BASELINE_RETIREMENT.json' if baseline else 'RETIREMENT.json'),
        retained=dict(result=str(output/'RESULT.json'),comparisons=str(output),raw_serving=str(output/'SERVING_RAW.json')),
        lost='Disposable optimizer/parameter/gradient replay is retired, not archived; all scientific checks, '
             'small block/anchor/RNG snapshots and raw serving evidence remain. Main training resumes original KPLUS4.')


def run(root):
    root=Path(root);plan=read(root/'sealed/PLAN.json')
    output=root/'qualification';output.mkdir(exist_ok=False)
    receipt=dict(status='running',execution_commit=plan['execution_source']['commit'],binding=plan['binding'],
        completed=[],attempted=None,budgets=QUALIFICATION_BUDGET,automatic_retry=False)
    started=time.perf_counter()
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['training_gpu']))
    def launch(command, log):
        with Path(log).open('w') as stream:
            completed=subprocess.run(command,cwd=plan['execution_source']['source_path'],env=env,
                stdout=stream,stderr=subprocess.STDOUT,check=False)
        if completed.returncode:
            raise RuntimeError('Qualification worker failed: '+str(log))
    try:
        verify(root,scope='qualification')
        # These four new responses belong to the declared 512 main feedback total.
        receipt['attempted']=dict(kind='prepare_shared_feedback')
        write_json(output/'STATUS.json',receipt)
        launch([plan['training_python'],'-u','-m','experiments.kplus_abt.prepare_feedback',str(root)],output/'feedback.log')
        for dataset in DATASETS:
            for case in QUALIFICATION_CASES:
                receipt['attempted']=dict(dataset=dataset,case=case,kind='disposable_update')
                write_json(output/'STATUS.json',receipt)
                command=[plan['training_python'],'-u','-m','experiments.kplus_abt.qualification',
                         '--root',str(root),'--dataset',dataset,'--case',case]
                launch(command,output/f'{dataset}_{case}.log')
                if case in ARMS:
                    receipt['attempted']=dict(dataset=dataset,case=case,kind='serving')
                    write_json(output/'STATUS.json',receipt)
                    launch(command+['--serve'],output/f'{dataset}_{case}_serving.log')
                retire_case(root,dataset,case)
                receipt['completed'].append(dict(dataset=dataset,case=case))
                write_json(output/'STATUS.json',receipt)
            retire_case(root,dataset,'KPLUS',baseline=True)
        admission=summarize(root)
        write_json(output/'ADMISSION.json',admission)
        receipt.update(status='completed',attempted=None)
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt['wall_s']=time.perf_counter()-started
        write_json(output/'STATUS.json',receipt)
    return receipt


def summarize(root):
    root=Path(root);plan=read(root/'sealed/PLAN.json')
    counts=dict(optimizer_steps=0,response_forwards=0,response_backwards=0,serving_generations=0)
    cases=[]
    for dataset in DATASETS:
        for case in QUALIFICATION_CASES:
            output=root/'qualification'/dataset/case;row=read(output/'RESULT.json')
            views=4 if case in ('CLEAN2','RANDOM','ADV') else 2
            require_complete_work(row['work'],dict(response_forwards=views,response_backwards=views,optimizer_steps=1))
            require_complete_work(row['startup']['work'],dict(response_forwards=2,response_backwards=2,optimizer_steps=0))
            if row['status']!='completed' or row['complete_state']['status']!='passed':
                raise ValueError('Incomplete qualification state')
            if case in ('DISABLED','ADV_ZERO') and not row['neutral_exact']['exact_equal']:
                raise ValueError('Native neutral qualification failed')
            if case in ('CLEAN2','RANDOM','ADV') and (row['perturbation']['status']!='passed' or
                    row['sampling_state_exact']['exact_equal'] is not True or len(row['native_input_pairs'])!=2):
                raise ValueError('Perturbation qualification failed')
            for name in ('optimizer_steps','response_forwards','response_backwards'):
                counts[name]+=row['work']['completed'][name]
            for name in ('response_forwards','response_backwards'):
                counts[name]+=row['startup']['work']['completed'][name]
            if case=='ADV':
                resource=row['resource']
                require_complete_work(resource['work'],dict(response_forwards=4,response_backwards=4,optimizer_steps=0))
                if resource['status']!='passed':
                    raise ValueError('Production-sized resource qualification failed')
                for name in ('response_forwards','response_backwards'):
                    counts[name]+=resource['work']['completed'][name]
            if case in ARMS:
                serving=read(output/'SERVING_CHECK.json')
                require_complete_work(serving['work'],dict(serving_generations=1))
                if serving['status']!='passed':
                    raise ValueError('Serving correctness failed')
                counts['serving_generations']+=serving['work']['completed']['serving_generations']
            cases.append(dict(dataset=dataset,case=case,result=str(output/'RESULT.json')))
    expected={key:QUALIFICATION_BUDGET[key] for key in counts}
    if counts!=expected:
        raise ValueError('Qualification work differs from sealed budget')
    return dict(status='passed',execution_commit=plan['execution_source']['commit'],binding=plan['binding'],
        counts=counts,cases=cases,checks={key:True for key in ('native_neutral_exact','two_view_execution',
            'mask_scope','perturbation_gradient','bf16_realization','same_clean_selection','one_adam_per_window',
            'serving','resource','complete_state')},
        scope='Native implementation/resource qualification, not a performance result; CLEAN2 differences retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root');args=parser.parse_args()
    run(args.root)
