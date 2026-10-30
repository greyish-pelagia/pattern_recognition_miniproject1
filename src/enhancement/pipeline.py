"""SUPIR inference without application, captioning, or image file I/O."""

import os
from pathlib import Path
from threading import RLock

from PIL import Image

_INFERENCE_LOCK = RLock()
_RUNTIME = None
_RUNTIME_KEY = None


def _load_runtime(
    sdxl_checkpoint, supir_checkpoint, clip1_path, clip2_path, device=None
):
    global _RUNTIME, _RUNTIME_KEY
    key = tuple(
        str(Path(path).expanduser().resolve()) if path is not None else None
        for path in (sdxl_checkpoint, supir_checkpoint, clip1_path, clip2_path)
    ) + (str(device) if device is not None else None,)
    if key != _RUNTIME_KEY:
        # Release GPU parameters before constructing a replacement model.
        _RUNTIME = None
        _RUNTIME_KEY = None
        _RUNTIME = _build_runtime(*key)
        _RUNTIME_KEY = key
    return _RUNTIME


def _select_device(torch, requested):
    if requested is None:
        if torch.cuda.is_available():
            requested = "cuda:0"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            raise RuntimeError(
                "SUPIR restoration requires an available MPS or CUDA accelerator."
            )
    device = torch.device(requested)
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is not available. Use an MPS-enabled PyTorch build on a supported Mac."
            )
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device.index}")
    else:
        raise ValueError("device must be mps or a CUDA device, such as cuda:0")
    return device


def _configure_for_device(config, device):
    config.model.params.sampler_config.params.device = str(device)
    if device.type == "mps":
        # Avoid CUDA-specific attention and mixed precision in the VAE/text models.
        config.model.params.ae_dtype = "fp32"
        config.model.params.diffusion_dtype = "fp32"
        for key in ("network_config", "control_stage_config"):
            params = config.model.params[key].params
            params.spatial_transformer_attn_type = "softmax"
            params.use_checkpoint = False
        config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"


def _build_runtime(
    sdxl_checkpoint, supir_checkpoint, clip1_path, clip2_path, device=None
):
    if not sdxl_checkpoint or not supir_checkpoint:
        raise ValueError(
            "Set sdxl_checkpoint and supir_checkpoint, or the environment variables "
            "SUPIR_SDXL_CHECKPOINT and SUPIR_CHECKPOINT."
        )
    checkpoint_paths = [
        Path(path).expanduser().resolve()
        for path in (sdxl_checkpoint, supir_checkpoint)
    ]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    import torch
    from omegaconf import OmegaConf

    from .SUPIR.util import PIL2Tensor, Tensor2PIL, load_state_dict
    from .sgm.util import instantiate_from_config

    device = _select_device(torch, device)

    config = OmegaConf.load(Path(__file__).parent / "options" / "SUPIR_v0.yaml")
    _configure_for_device(config, device)
    embedders = config.model.params.conditioner_config.params.emb_models
    if clip1_path:
        embedders[0].params.version = str(Path(clip1_path).expanduser())
    if clip2_path:
        embedders[1].params.version = str(Path(clip2_path).expanduser())
    model = instantiate_from_config(config.model).cpu()
    for path in checkpoint_paths:
        model.load_state_dict(load_state_dict(str(path)), strict=False)
    model = model.eval().requires_grad_(False).to(device)
    return model, PIL2Tensor, Tensor2PIL


def restore_images(
    images: list[Image.Image],
    *,
    sdxl_checkpoint: str
    | os.PathLike
    | None = "/models/sd_xl_base_1.0_0.9vae.safetensors",
    supir_checkpoint: str | os.PathLike | None = "/models/SUPIR-v0Q.ckpt",
    clip1_path: str | os.PathLike | None = None,
    clip2_path: str | os.PathLike | None = None,
    device: str | None = None,
    dry_run: bool = True,
) -> list[Image.Image]:
    """Return one restored RGB PIL image per input, in order and at its original size.

    Images may have different sizes and modes. Inputs are never modified. The
    accelerator model is loaded lazily and reused for subsequent calls with the same
    checkpoints. Only the most recent model is cached; calls are serialized.
    CLIP weights use upstream model identifiers unless local paths are supplied.
    Device selection prefers CUDA, then MPS; pass device='mps' to choose explicitly.
    With dry_run=True, return the original list and image objects unchanged,
    without loading models, accessing checkpoints, or checking accelerators.
    """
    if not isinstance(images, list) or any(
        not isinstance(image, Image.Image) for image in images
    ):
        raise TypeError("images must be a list of PIL.Image.Image objects")
    if dry_run:
        return images
    if not images:
        return []
    if any(image.width <= 0 or image.height <= 0 for image in images):
        raise ValueError("Input images must have nonzero width and height")

    with _INFERENCE_LOCK:
        model, prepare, finish = _load_runtime(
            os.fspath(sdxl_checkpoint)
            if sdxl_checkpoint is not None
            else os.environ.get("SUPIR_SDXL_CHECKPOINT"),
            os.fspath(supir_checkpoint)
            if supir_checkpoint is not None
            else os.environ.get("SUPIR_CHECKPOINT"),
            os.fspath(clip1_path) if clip1_path is not None else None,
            os.fspath(clip2_path) if clip2_path is not None else None,
            device,
        )
        model_device = next(model.parameters()).device
        outputs = []
        for image in images:
            tensor, height, width = prepare(image.convert("RGB"))
            samples = model.batchify_sample(
                tensor.unsqueeze(0).to(model_device),
                [""],
                num_steps=50,
                restoration_scale=-1,
                s_churn=5,
                s_noise=1.01,
                cfg_scale=4.0,
                control_scale=1.0,
                seed=1234,
                num_samples=1,
                color_fix_type="Wavelet",
                use_linear_CFG=True,
                cfg_scale_start=1.0,
            )
            outputs.append(finish(samples[0], height, width))
        return outputs
