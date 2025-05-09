## Introduction
VeriSilicon support FlagGems as a backend, based on our VPEX, triton-vsi-backend, OpenCL driver, and TensorCore toolkits.

## Preparation
1.Install pytorch>=2.6.0, and it will also install its dependencies triton>=3.2.0

2.Download triton-vsi-backend wheel form https://github.com/VeriSilicon/triton-vsi-backend and install with pip.

3.Download VSI SDK from https://github.com/VeriSilicon/triton-vsi-backend, it includes VSI OpenCL driver and TensorCore toolkits.

4.Download VPEX wheel from https://github.com/VeriSilicon/VPEX, and pip install it.

### Environment variable setting
```shell
    export VSI_SDK_DIR=your/path/to/vsi_sdk
    export LD_LIBRARY_PATH=your/path/to/vsi_sdk/drivers
    export GEMS_VENDOR=VeriSilicon
```

## Usage

```python
import torch
import vpex
import flag_gems

A = torch.randn((64, 64), device='vsi', dtype=torch.float32)
B = torch.randn((64, 64), device='vsi', dtype=torch.float32)
with flag_gems.use_gems():
    result = torch.add(A, B)
print(result.to('cpu'))
```
## Support op list
[abs, add, addmm, all, amax, any, arange, bitwise_and, bitwise_or, bitwise_or, clamp, cos, cumsum, div, eq, erf, exp, fill, fill_tensor, full, full_like, ge, gt, index_select, le, lt, masked_fill, maximum, mean, minimum, mv, mul, ne, neg, ones, ones_like, prod, reciprocal, repeat_interleave, rms_norm:forward, silu:forward, sin, sub, sum, tanh:forward, zeros, zeros_like, logical_and, logical_or, logical_not, logical_xor, rsub, dot, vdot, elu, log]

Example:
```shell
    pytest -m abs --ref cpu --mode quick
```