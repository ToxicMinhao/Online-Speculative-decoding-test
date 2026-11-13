"""CPU-only, streamed final BF16 parameter comparison; never modifies exports."""
import math
from pathlib import Path
import torch
from safetensors import safe_open


def compare_weights(reference, candidate):
    rows=[]
    with safe_open(str(Path(reference)/'model.safetensors'),framework='pt',device='cpu') as left, \
         safe_open(str(Path(candidate)/'model.safetensors'),framework='pt',device='cpu') as right:
        if set(left.keys())!=set(right.keys()):raise ValueError('Final export parameter names differ')
        for name in left.keys():
            a,b=left.get_tensor(name),right.get_tensor(name)
            if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Final tensor shape/dtype differs: '+name)
            aa,bb=a.reshape(-1),b.reshape(-1)
            count,square,maximum=0,0.,0.
            for start in range(0,aa.numel(),1_000_000):
                delta=bb[start:start+1_000_000].double()-aa[start:start+1_000_000].double()
                if not bool(torch.isfinite(delta).all()):raise ValueError('Nonfinite endpoint difference')
                count+=int(torch.count_nonzero(delta));square+=float(delta.square().sum())
                maximum=max(maximum,float(delta.abs().max()))
            rows.append(dict(parameter=name,shape=list(a.shape),dtype=str(a.dtype),coordinates=a.numel(),
                different=count,difference_squared_norm=square,max_absolute_difference=maximum))
    return dict(status='passed',different_coordinates=sum(r['different'] for r in rows),
                coordinates=sum(r['coordinates'] for r in rows),
                difference_norm=math.sqrt(math.fsum(r['difference_squared_norm'] for r in rows)),
                parameters=rows,model_forwards=0,optimizer_steps=0)
