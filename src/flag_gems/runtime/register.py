from . import backend, commom_utils, error
from .backend.device import DeviceDetector


class Register:
    def __init__(
        self,
        config,
        user_unused_ops_list=None,
        lib=None,
    ):
        # lib is a instance of torch.library.Library
        self.device = DeviceDetector()
        self.lib = lib
        # reg_key like 'CUDA', reg_bac_key like AutogradCUDA
        self.reg_key = self.device.name.upper() if self.device.name in ["cuda", "amd", "cpu"] else "PrivateUse1"
        self.reg_bac_key = "Autograd" + self.reg_key
        self.all_ops = []
        self.vendor_extend_configs = self.get_vendor_extend_op()
        self.vendor_unused_ops_list = self.get_vendor_unused_op()
        self.unused_ops = user_unused_ops_list + self.vendor_unused_ops_list
        self.config = config + self.vendor_extend_configs
        self.config_filter()
        self.for_each()

    def config_filter(self):
        self.config = [
            item for item in self.config if item[1].__name__ not in self.unused_ops
        ]

    def get_vendor_extend_op(self):
        if self.device.vendor != commom_utils.vendors.NVIDIA:
            return backend.get_curent_device_extend_op(self.device.vendor_name)
        return ()

    def get_vendor_unused_op(self):
        if self.device.vendor != commom_utils.vendors.NVIDIA:
            return backend.get_curent_device_unused_op(self.device.vendor_name)
        return []

    def register_impl(self, key, fn, has_backward):
        if self.device.vendor != commom_utils.vendors.NVIDIA:
            if key in self.extend_configs_dict:
                single_item = self.extend_configs_dict[key]
                _, fn, has_backward = single_item
        if has_backward is commom_utils.Autograd.enable:
            device_key = self.reg_bac_key
        else:
            device_key = self.reg_key
        self.all_ops.append(key)
        self.lib.impl(key, fn, device_key)

    def for_each(self):
        self.extend_configs_dict = {}
        for item in self.vendor_extend_configs:
            self.extend_configs_dict[item[0]] = item
        try:
            for key, func, has_backward in self.config:
                if key not in self.unused_ops and key not in self.all_ops:
                    self.register_impl(key, func, has_backward)
        except Exception as e:
            error.register_error(e)

    def get_all_ops(self):
        return self.all_ops

    def get_unused_ops(self):
        return self.unused_ops

    def get_vendor_name(self):
        return self.device.vendor_name

    def get_current_device(self):
        return self.device.name
