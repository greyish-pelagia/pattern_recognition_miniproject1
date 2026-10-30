# Python image restoration

This package contains the inference portion of SUPIR. Its single public
entrypoint accepts a list of PIL images and returns one restored RGB PIL image
per input, in the same order and at the original dimensions. Images may have
different sizes and modes. Inputs are not modified. No images are read from or
written to disk by the pipeline.

```python
from PIL import Image
from src.enhancement import restore_images

images = [Image.open("photo1.png"), Image.open("photo2.png")]
restored = restore_images(
    images,
)
# restored is list[PIL.Image.Image]
```

Use `restore_images(images, dry_run=True)` to return the exact input list and
image objects unchanged, preserving their modes and sizes. Dry runs validate
that the input is a list of PIL images but require no checkpoints, PyTorch, or
accelerator. The flag defaults to `True`.

The checkpoint arguments may be omitted when `SUPIR_SDXL_CHECKPOINT` and
`SUPIR_CHECKPOINT` are set. Choose a SUPIR-v0Q checkpoint for general restoration
or SUPIR-v0F for light degradation. Both the SDXL and SUPIR checkpoints are
required and must be downloaded separately.

An available MPS accelerator on macOS or a CUDA GPU, plus the dependencies in
`requirements.txt`, are required for inference. Device selection prefers CUDA,
then MPS. Pass `device="mps"` or `device="cuda:0"` to choose explicitly. An
unavailable accelerator raises an error; the pipeline does not silently use CPU.
On macOS, use an MPS-enabled PyTorch build and check
`torch.backends.mps.is_available()`; see the
[PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

MPS uses float32 for the VAE, diffusion model and text encoders, with native
PyTorch scaled dot product attention. This avoids CUDA autocast and xFormers,
but requires enough unified memory for the full SUPIR and SDXL models.
xFormers is optional for CUDA and is not required on macOS.

Importing the package and calling `restore_images([])` do not load PyTorch or
model weights. The most recently used model is cached, and inference calls are
serialized. Images are processed sequentially to support different sizes and
limit peak GPU memory. As with upstream SUPIR, each image is internally resized
to at least 1024 pixels on its shorter side, with dimensions aligned to 64,
then restored to its original dimensions for the output.

The SDXL text encoders load `openai/clip-vit-large-patch14` and
`laion2b_s39b_b160k` for `ViT-bigG-14` through Transformers and OpenCLIP. For local
weights, pass `clip1_path` (a Transformers model/tokenizer directory) and
`clip2_path` (an OpenCLIP checkpoint file). These are text encoders needed by
SUPIR; LLaVA and automatic image captioning have been removed.

The fixed positive and negative prompts are in `options/SUPIR_v0.yaml`.
Restoration uses the original script's 50-step, fidelity-oriented settings and
seed 1234. There is one output per input. CLI, Gradio, HTTP serving, training,
evaluation, face-specific processing, alternate samplers and tiled VAE support
are not included.

Contract tests run with `python -m unittest discover -s tests -v` from the project
root. The tensor tests exercise CPU and MPS when available and skip if their
dependencies are missing. Full restoration additionally requires an accelerator
and the model weights.

Validation on macOS with PyTorch 2.15.0 and OmegaConf 2.3.0: all 18 tests passed,
including real CPU and MPS attention, small VAE, color-correction, conversion and
sampler operations. Full restoration with trained checkpoints has not been run.

## Attribution and use restrictions

Based on _Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic
Image Restoration In the Wild_ (Yu et al., CVPR 2024),
[SUPIR](https://github.com/Fanghua-Yu/SUPIR).
The upstream license is preserved in `LICENSE`.

The upstream SUPIR declaration restricts use, reproduction and distribution to
non-commercial purposes. Commercial use requires prior written permission from
Dr. Jinjin Gu (jinjin.gu@suppixel.ai). The software is provided without warranty;
see the upstream project and the included license for the applicable terms.
