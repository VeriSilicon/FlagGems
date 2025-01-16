## Introduction
Verisilicon support FlagGems as a backend, based on our ZenCompiler, VPEX and OpenCL driver.

## Usage
```python
import torch
import vpex
import flag_gems

A = torch.rand((64, 64), device='vsi', dtype=torch.float32)
B = torch.rand((64, 64), device='vsi', dtype=torch.float32)
with flag_gems.use_gems():
    result = torch.add(A, B)
print(result.to('cpu'))
```
## Support op list
[abs, add, arange, clamp, cos, cumsum, div:true_divide, div:trunc_divide, div:remainder, erf, exp, fill, fill_tensor,
full, full_like, mean, mul, neg, ones, ones_like, reciprocal, repeat_interleave_self_int, rms_norm:forward, silu:forward,
sin, stack, sub, sum, tanh:forward, zeros, zeros_like, diag:2d_to_1d]
