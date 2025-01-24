import vpex
import triton
from triton.backends.vsi.driver import VSIDriver
triton.runtime.driver.set_active(VSIDriver())

from backend_utils import VendorInfoBase  # noqa: E402

from .heuristics_config_utils import HEURISTICS_CONFIGS
from torchgen.model import DispatchKey
global specific_ops, unused_ops
specific_ops = None
unused_ops = None
vendor_info = VendorInfoBase(
    vendor_name="VeriSilicon", device_name="vsi", device_query_cmd=""
)


def OpLoader():
    global specific_ops, unused_ops
    if specific_ops is None:
        from . import ops  # noqa: F403

        specific_ops = ops.get_specific_ops()
        unused_ops = ops.get_unused_ops()


__all__ = ["HEURISTICS_CONFIGS", "vendor_info", "OpLoader"]
