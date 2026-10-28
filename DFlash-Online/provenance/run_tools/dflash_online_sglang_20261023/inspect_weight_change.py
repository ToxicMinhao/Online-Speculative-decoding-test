"""Read-only qualification check: exported draft really changed from release."""
from pathlib import Path
import json
import math
import sys
import torch
from safetensors import safe_open

torch.set_num_threads(4)
released, updated, output = map(Path, sys.argv[1:])

def locations(root):
    result = {}
    for path in root.glob('*.safetensors'):
        with safe_open(path, framework='pt', device='cpu') as source:
            for name in source.keys(): result[name] = path
    return result

old, new = locations(released), locations(updated)
assert old.keys() == new.keys()
rows = []
for name in sorted(old):
    with safe_open(old[name],framework='pt',device='cpu') as a, safe_open(new[name],framework='pt',device='cpu') as b:
        x,y=a.get_tensor(name),b.get_tensor(name)
        assert x.shape == y.shape and x.dtype == y.dtype == torch.bfloat16
        changed=0; squared=0.
        for xx,yy in zip(x.reshape(-1).split(1048576),y.reshape(-1).split(1048576)):
            changed += int(torch.count_nonzero(xx != yy))
            delta=xx.float()-yy.float()
            squared += float((delta.double().square()).sum())
        rows.append({'parameter':name,'elements':x.numel(),'changed_elements':changed,'delta_l2':math.sqrt(squared)})
value={'released':str(released),'updated':str(updated),'dtype':'bfloat16',
       'tensor_count':len(rows),'changed_tensor_count':sum(x['changed_elements']>0 for x in rows),
       'changed_elements':sum(x['changed_elements'] for x in rows),'parameters':rows}
assert value['changed_elements'] > 0
output.write_text(json.dumps(value,indent=2)+'\n')
print({k:v for k,v in value.items() if k!='parameters'})
