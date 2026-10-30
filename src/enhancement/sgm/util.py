import importlib
from contextlib import nullcontext
from functools import wraps
from inspect import isfunction

import torch


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def inference_autocast(device, dtype=None):
    """Keep MPS in float32; mixed precision is only enabled for CUDA."""
    if device.type != 'cuda' or dtype == torch.float32:
        return nullcontext()
    if dtype is None:
        dtype = (torch.get_autocast_dtype('cuda') if hasattr(torch, 'get_autocast_dtype')
                 else torch.get_autocast_gpu_dtype())
    return torch.autocast('cuda', dtype=dtype)


def autocast(f, enabled=True):
    @wraps(f)
    def do_autocast(*args, **kwargs):
        device = next(args[0].parameters()).device
        with inference_autocast(device) if enabled else nullcontext():
            return f(*args, **kwargs)

    return do_autocast


def exists(x):
    return x is not None


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    # Upstream YAML/checkpoint targets name vendored packages at the top level.
    # Resolve them under whichever package name the caller imported.
    if module.split(".")[0] in {"sgm", "SUPIR"}:
        module = __package__.rsplit(".", 1)[0] + "." + module
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]
