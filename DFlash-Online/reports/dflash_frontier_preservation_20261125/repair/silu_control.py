"""CPU-only controlled comparison; no model or GPU initialization."""
import json
from pathlib import Path
import torch
x=torch.linspace(-8,8,8192,dtype=torch.bfloat16).requires_grad_()
u=torch.randn(x.shape,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(1701))
native=torch.autograd.grad(torch.nn.functional.silu(x),x,u,create_graph=False)[0]
higher=torch.autograd.grad(torch.nn.functional.silu(x),x,u,create_graph=True)[0].detach()
explicit=torch.ops.aten.silu_backward.default(u,x).detach()
result=dict(torch_version=torch.__version__,coordinates=x.numel(),different=int((native!=higher).sum()),
            max_absolute_error=float((native.float()-higher.float()).abs().max()),
            relative_l2=float((native.float()-higher.float()).norm()/native.float().norm()),
            explicit_native_backward_exact=torch.equal(native,explicit),cuda_initialized=torch.cuda.is_initialized(),
            interpretation='Controlled CPU BF16 higher-order backward difference; native model attribution requires the unchanged first-VJP check')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
