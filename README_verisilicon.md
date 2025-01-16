## Introduction
Verisilicon support FlagGems as a backend, based on our ZenCompiler, VPEX, triton-vsi-backend and OpenCL driver.

## Preparation
1.Install pytorch==2.5.1, and it will also install its dependencies triton==3.1.0

2.Download ZenCompiler from https://github.com/VeriSilicon/ZenCompiler, it includes ZenCompiler binary, TcTools and OpenCL.

3.Download triton-vsi-backend form https://github.com/VeriSilicon/triton-vsi-backend, and create a Symbolic Link.
```shell
cd <your_env/lib/python3.x/site-packages/triton/backends>
ln -s <this_repo_root/backend> vsi
```

4.Download VPEX wheel from https://github.com/VeriSilicon/VPEX, and pip install it.

### Environment variable setting
```shell
    export VSI_DRIVER_PATH=your/path/to/ZenCompiler/lib
    export VSI_INCLUDE_PATH=your/path/to/ZenCompiler/include
    export TC_TOOLKITS_PATH=your/path/to/ZenCompiler/lib
    export ZEN_COMPILER_PATH=your/path/to/ZenCompiler/bin
    export CC=clang++
    export LD_LIBRARY_PATH=your/path/to/ZenCompiler/lib:your_env/lib/python3.x/site-packages/vpex/lib:$LD_LIBRARY_PATH
```

## Usage
```python
import torch
import vpex
import flag_gems
from triton.backends.vsi.driver import VSIDriver
triton.runtime.driver.set_active(VSIDriver())

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
