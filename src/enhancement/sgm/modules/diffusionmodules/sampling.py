"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import to_d
from ...util import append_dims, default, instantiate_from_config

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class RestoreEDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, restore_cfg=4.0,
            restore_cfg_s_tmin=0.05, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.restore_cfg = restore_cfg
        self.restore_cfg_s_tmin = restore_cfg_s_tmin
        self.sigma_max = 14.6146

    def denoise(self, x, denoiser, sigma, cond, uc, control_scale=1.0):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), control_scale)
        denoised = self.guider(denoised, sigma)
        return denoised

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0, x_center=None, eps_noise=None,
                     control_scale=1.0, use_linear_control_scale=False, control_scale_start=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            if eps_noise is not None:
                eps = eps_noise * self.s_noise
            else:
                eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        if use_linear_control_scale:
            control_scale = (sigma[0].item() / self.sigma_max) * (control_scale_start - control_scale) + control_scale

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc, control_scale=control_scale)

        if (next_sigma[0] > self.restore_cfg_s_tmin) and (self.restore_cfg > 0):
            d_center = (denoised - x_center)
            denoised = denoised - d_center * ((sigma.view(-1, 1, 1, 1) / self.sigma_max) ** self.restore_cfg)

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        x = self.euler_step(x, d, dt)
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, x_center=None, control_scale=1.0,
                 use_linear_control_scale=False, control_scale_start=0.0):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for _idx, i in enumerate(self.get_sigma_gen(num_sigmas)):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
                x_center,
                control_scale=control_scale,
                use_linear_control_scale=use_linear_control_scale,
                control_scale_start=control_scale_start,
            )
        return x
