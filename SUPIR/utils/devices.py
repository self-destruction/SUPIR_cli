import sys
import contextlib
from functools import lru_cache
import gc
import torch

if sys.platform == "darwin":
    from modules import mac_specific


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps


def get_cuda_device_string():
    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    return get_optimal_device()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        mac_specific.torch_mps_gc()
    gc.collect()


def enable_tf32():
    if torch.cuda.is_available():
        if cuda_no_autocast():
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
fp8: bool = False
cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = torch.device("cuda")
dtype = torch.float16
dtype_vae = torch.float16
dtype_unet = torch.float16
dtype_inference: torch.dtype = torch.float16
unet_needs_upcast = False

patch_module_list = [
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.MultiheadAttention,
    torch.nn.GroupNorm,
    torch.nn.LayerNorm,
]


def cuda_no_autocast(device_id=None) -> bool:
    if device_id is None:
        device_id = torch.cuda.current_device()
    return (
            torch.cuda.get_device_capability(device_id) == (7, 5)
            and torch.cuda.get_device_name(device_id).startswith("NVIDIA GeForce GTX 16")
    )


def cond_cast_unet(tgt_input):
    return tgt_input.to(dtype_unet) if unet_needs_upcast else tgt_input


def cond_cast_float(tgt_input):
    return tgt_input.float() if unet_needs_upcast else tgt_input


def randn(seed, shape):
    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    return torch.randn(shape, device=device)


# def autocast(disable=False):
#     if disable:
#         return contextlib.nullcontext()
#
#     return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda",
                          enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if not torch.all(torch.isnan(x)).item():
        return

    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."

    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."

    else:
        message = "A tensor with all NaNs was produced."

    message += " Use --disable-nan-check commandline argument to disable this check."

    raise NansException(message)


@lru_cache
def first_time_calculation():
    """
    just do any calculation with pytorch layers - the first time this is done it allocaltes about 700MB of memory and
    spends about 2.7 seconds doing that, at least wih NVidia.
    """

    x = torch.zeros((1, 1)).to(device, dtype)
    linear = torch.nn.Linear(1, 1).to(device, dtype)
    linear(x)

    x = torch.zeros((1, 1, 3, 3)).to(device, dtype)
    conv2d = torch.nn.Conv2d(1, 1, (3, 3)).to(device, dtype)
    conv2d(x)


def manual_cast_forward(target_dtype):
    def forward_wrapper(self, *args, **kwargs):
        if any(
                isinstance(arg, torch.Tensor) and arg.dtype != target_dtype
                for arg in args
        ):
            args = [arg.to(target_dtype) if isinstance(arg, torch.Tensor) else arg for arg in args]
            kwargs = {k: v.to(target_dtype) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        org_dtype = target_dtype
        for param in self.parameters():
            if param.dtype != target_dtype:
                org_dtype = param.dtype
                break

        if org_dtype != target_dtype:
            self.to(target_dtype)
        result = self.org_forward(*args, **kwargs)
        if org_dtype != target_dtype:
            self.to(org_dtype)

        if target_dtype != dtype_inference:
            if isinstance(result, tuple):
                result = tuple(
                    i.to(dtype_inference)
                    if isinstance(i, torch.Tensor)
                    else i
                    for i in result
                )
            elif isinstance(result, torch.Tensor):
                result = result.to(dtype_inference)
        return result

    return forward_wrapper


@contextlib.contextmanager
def manual_cast(target_dtype):
    applied = False
    for module_type in patch_module_list:
        if hasattr(module_type, "org_forward"):
            continue
        applied = True
        org_forward = module_type.forward
        if module_type == torch.nn.MultiheadAttention:
            module_type.forward = manual_cast_forward(torch.float32)
        else:
            module_type.forward = manual_cast_forward(target_dtype)
        module_type.org_forward = org_forward
    try:
        yield None
    finally:
        if applied:
            for module_type in patch_module_list:
                if hasattr(module_type, "org_forward"):
                    module_type.forward = module_type.org_forward
                    delattr(module_type, "org_forward")


def autocast(disable=False, _dtype=torch.bfloat16):
    if disable:
        return contextlib.nullcontext()

    if fp8 and device == cpu:
        return torch.autocast("cpu", dtype=_dtype, enabled=True)

    if fp8 and dtype_inference == torch.float32:
        return manual_cast(dtype)

    if dtype == torch.float32 or dtype_inference == torch.float32:
        return contextlib.nullcontext()

    if has_mps() or cuda_no_autocast():
        return manual_cast(dtype)
    else:
        return torch.autocast("cuda", dtype=_dtype, enabled=True)
    
