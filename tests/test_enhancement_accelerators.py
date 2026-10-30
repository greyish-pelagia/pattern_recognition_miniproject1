"""Real tensor checks; no checkpoints are needed for these small model components."""

from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import torch
    from omegaconf import OmegaConf
    import einops
    import tqdm
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install torch, einops, and omegaconf for tensor checks')
class AcceleratorTests(unittest.TestCase):
    def devices(self):
        return ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])

    def test_mps_configuration_uses_float32_and_native_attention(self):
        from src.enhancement.pipeline import _configure_for_device

        path = Path(__file__).resolve().parents[1] / 'src/enhancement/options/SUPIR_v0.yaml'
        config = OmegaConf.load(path)
        _configure_for_device(config, torch.device('mps'))
        params = config.model.params
        self.assertEqual(params.ae_dtype, 'fp32')
        self.assertEqual(params.diffusion_dtype, 'fp32')
        self.assertEqual(params.sampler_config.params.device, 'mps')
        self.assertEqual(params.network_config.params.spatial_transformer_attn_type, 'softmax')
        self.assertEqual(params.control_stage_config.params.spatial_transformer_attn_type, 'softmax')
        self.assertEqual(params.first_stage_config.params.ddconfig.attn_type, 'vanilla')

    def test_attention_runs_without_cuda_or_xformers(self):
        from src.enhancement.sgm.modules.attention import CrossAttention, MemoryEfficientCrossAttention
        from src.enhancement.sgm.modules.diffusionmodules.model import AttnBlock, MemoryEfficientAttnBlock

        for device in self.devices():
            with self.subTest(device=device), torch.no_grad():
                q = torch.randn(1, 8, 32, device=device)
                context = torch.randn(1, 6, 32, device=device)
                native = CrossAttention(32, heads=2, dim_head=16).to(device).eval()
                fallback = MemoryEfficientCrossAttention(32, heads=2, dim_head=16).to(device).eval()
                fallback.load_state_dict(native.state_dict())
                with patch('src.enhancement.sgm.modules.attention.sdp_kernel', side_effect=AssertionError('CUDA called')):
                    expected = native(q, context)
                    actual = fallback(q, context)
                torch.testing.assert_close(actual, expected)
                image = torch.randn(1, 32, 4, 4, device=device)
                vae_native = AttnBlock(32).to(device).eval()
                vae_fallback = MemoryEfficientAttnBlock(32).to(device).eval()
                vae_fallback.load_state_dict(vae_native.state_dict())
                torch.testing.assert_close(vae_fallback(image), vae_native(image))
                self.assertTrue(torch.isfinite(actual).all().item())

    def test_vae_color_correction_and_image_conversion(self):
        from PIL import Image
        from src.enhancement.SUPIR.util import PIL2Tensor, Tensor2PIL
        from src.enhancement.SUPIR.utils.colorfix import wavelet_reconstruction
        from src.enhancement.sgm.models.autoencoder import AutoencoderKLInferenceWrapper

        ddconfig = dict(double_z=True, z_channels=4, resolution=32, in_channels=3,
                        out_ch=3, ch=32, ch_mult=[1, 2], num_res_blocks=1,
                        attn_resolutions=[], dropout=0.0, attn_type='vanilla')
        for device in self.devices():
            with self.subTest(device=device), torch.no_grad():
                model = AutoencoderKLInferenceWrapper(embed_dim=4, ddconfig=ddconfig).eval().to(device)
                tensor, height, width = PIL2Tensor(Image.new('RGB', (32, 24), 'red'), min_size=64)
                tensor = tensor.unsqueeze(0).to(device)
                restored = model.decode(model.encode(tensor)).float()
                restored = wavelet_reconstruction(restored, tensor)
                self.assertTrue(torch.isfinite(restored).all().item())
                result = Tensor2PIL(restored[0], height, width)
                self.assertEqual(result.size, (32, 24))
                self.assertEqual(result.mode, 'RGB')

    def test_sampler_places_schedule_on_input_device(self):
        from src.enhancement.sgm.modules.diffusionmodules.sampling import RestoreEDMSampler

        for device in self.devices():
            sampler = RestoreEDMSampler(
                discretization_config={'target': 'sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization'},
                guider_config={'target': 'sgm.modules.diffusionmodules.guiders.IdentityGuider'},
                num_steps=2, device=device, restore_cfg=-1,
            )
            sample = torch.randn(1, 4, 4, 4, device=device)

            def denoise(x, sigma, condition, control_scale):
                self.assertEqual(sigma.device.type, device)
                return x / (1 + sigma[:, None, None, None])

            result = sampler(denoise, sample, cond={})
            self.assertEqual(result.device.type, device)
            self.assertTrue(torch.isfinite(result).all().item())

    def test_non_cuda_inference_never_enters_cuda_autocast(self):
        from src.enhancement.sgm.util import inference_autocast

        with patch('torch.autocast', side_effect=AssertionError('autocast called')):
            with inference_autocast(torch.device('mps'), torch.float32):
                pass

    def test_cuda_context_preserves_the_outer_autocast_dtype(self):
        from contextlib import nullcontext
        from src.enhancement.sgm.util import inference_autocast

        get_dtype = (lambda: torch.get_autocast_dtype('cuda')) if hasattr(torch, 'get_autocast_dtype') else torch.get_autocast_gpu_dtype
        set_dtype = (lambda value: torch.set_autocast_dtype('cuda', value)) if hasattr(torch, 'set_autocast_dtype') else torch.set_autocast_gpu_dtype
        previous = get_dtype()
        try:
            set_dtype(torch.bfloat16)
            with patch('torch.autocast', return_value=nullcontext()) as autocast:
                with inference_autocast(torch.device('cuda')):
                    pass
                self.assertEqual(autocast.call_args.kwargs['dtype'], torch.bfloat16)
        finally:
            set_dtype(previous)
