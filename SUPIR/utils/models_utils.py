import torch

from SUPIR.utils import shared, devices


def check_fp8(model):
    if model is None:
        return None
    if devices.get_optimal_device_name() == "mps":
        enable_fp8 = False
    elif shared.opts.fp8_storage:
        enable_fp8 = True
    else:
        enable_fp8 = False
    return enable_fp8


def load_model_weights(model, state_dict):
    if devices.fp8:
        model.half()

    model.load_state_dict(state_dict, strict=False)

    del state_dict

    if shared.opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)
        # print('apply channels_last')

    if not shared.opts.half_mode:
        model.float()
        devices.dtype_unet = torch.float32
        # print('apply float')
    else:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)

        if shared.opts.half_mode:
            model.half()

        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model
        # print('apply half')

    for module in model.modules():
        if hasattr(module, 'fp16_weight'):
            del module.fp16_weight
        if hasattr(module, 'fp16_bias'):
            del module.fp16_bias

    if check_fp8(model):
        devices.fp8 = True
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                module.to(torch.float8_e4m3fn)
        # print("apply fp8")
    else:
        devices.fp8 = False

    devices.unet_needs_upcast = shared.opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16

    model.first_stage_model.to(devices.dtype_vae)

    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(devices.device)
