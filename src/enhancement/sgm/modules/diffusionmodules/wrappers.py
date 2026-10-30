import torch
import torch.nn as nn
from packaging import version
from ...util import inference_autocast
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.cache_size_limit = 512


class ControlWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False, dtype=torch.float32):
        super().__init__()
        self.compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = self.compile(diffusion_model)
        self.control_model = None
        self.dtype = dtype

    def load_control_model(self, control_model):
        self.control_model = self.compile(control_model)

    def forward(
            self, x: torch.Tensor, t: torch.Tensor, c: dict, control_scale=1, **kwargs
    ) -> torch.Tensor:
        with inference_autocast(x.device, self.dtype):
            control = self.control_model(x=c.get("control", None), timesteps=t, xt=x,
                                         control_vector=c.get("control_vector", None),
                                         mask_x=c.get("mask_x", None),
                                         context=c.get("crossattn", None),
                                         y=c.get("vector", None))
            out = self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                control=control,
                control_scale=control_scale,
                **kwargs,
            )
        return out.float()
