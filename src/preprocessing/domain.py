from dataclasses import dataclass
from typing import Literal


@dataclass
class PreprocessingSettings:
    """Settings for image preprocessing; preserve color and texture by default."""

    brightness_adjustment: float | str = 1.0
    contrast_adjustment: float | str = 1.0
    brightness_reference: float = 128.0
    contrast_reference: float = 40.0
    histogram_equalization: bool = True

    normalize: bool = True
    normalization_method: str = "min-max"

    apply_filter: bool = True
    filter_type: str = "gaussian"

    binarize: bool = False
    binarize_per_channel: bool = True

    apply_morphology: bool = False
    morphology_operation: str = "opening-closing"
    morphology_kernel_size: int = 3

    remove_background: bool = False
    leaf_mask_stage: Literal[
        "start", "after_adjustment", "after_filtering", "after_equalization"
    ] = "start"
