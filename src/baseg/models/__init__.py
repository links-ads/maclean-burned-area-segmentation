from baseg.models.encoder_decoder import CustomEncoderDecoder
from baseg.models.heads import CustomBaseDecodeHead
from mmseg.registry import MODELS

__all__ = [
    "CustomEncoderDecoder",
    "CustomBaseDecodeHead",
]


def build_model(cfg, **kwargs: dict):
    """Build model."""
    return MODELS.build(cfg, **kwargs)
