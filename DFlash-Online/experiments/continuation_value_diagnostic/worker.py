"""One physical GPU: read parent and child, without constructing a trainer."""

import argparse
import json
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from .native import ReadOnlyEndpoint, qualify_endpoint


def run(contract, phase, shard):
    import torch
    root = Path(contract['root'])/phase/f'gpu{shard}'
    root.mkdir(parents=True, exist_ok=False)
    work = dict(status='running',phase=phase,shard=shard,physical_gpu=contract['devices'][shard],
                execution_commit=contract['execution_commit'],endpoints={},start_epoch=time.time())
    write_json(root/'WORK.json', work)
    try:
        for endpoint, pin in contract['endpoints'].items():
            before=time.perf_counter()
            adapter=ReadOnlyEndpoint.load(contract['original_config'], pin['export'], pin['identity'], endpoint=endpoint)
            endpoint_root=root/endpoint
            endpoint_root.mkdir()
            work['endpoints'][endpoint]=dict(load_s=time.perf_counter()-before,status='loaded')
            write_json(root/'WORK.json',work)
            if phase=='qualification':
                legacy=json.loads((Path(contract['root'])/'legacy'/f'{endpoint}.json').read_text())
                result=qualify_endpoint(adapter,contract['legacy_binding']['sets']['new_buffer'][:2],
                                        legacy['sets']['new_buffer'][:2],contract['features'][0],endpoint_root)
                work['endpoints'][endpoint]['qualification']=result
            else:
                for row in contract['features']:
                    if row['physical_shard']==shard:
                        adapter.read_response(row,endpoint_root/f'{row["ordinal"]:03d}.json')
            work['endpoints'][endpoint]['final_state']=adapter.close()
            work['endpoints'][endpoint]['status']='completed'
            write_json(root/'WORK.json',work)
            del adapter
            torch.cuda.empty_cache()
        work['status']='passed'
    except BaseException as error:
        work.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        work['end_epoch']=time.time()
        work['elapsed_s']=work['end_epoch']-work['start_epoch']
        work['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        work['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        write_json(root/'WORK.json',work)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True)
    parser.add_argument('--phase',choices=['qualification','primary'],required=True)
    parser.add_argument('--shard',type=int,required=True)
    args=parser.parse_args()
    run(json.loads(Path(args.contract).read_text()),args.phase,args.shard)


if __name__=='__main__':
    main()
