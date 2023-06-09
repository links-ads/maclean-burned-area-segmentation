from typing import Optional
from torch import Tensor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList)
from mmseg.registry import MODELS

@MODELS.register_module()
class CustomEncoderDecoder(EncoderDecoder):
    def __init__(self, backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(backbone,
                 decode_head,
                 neck,
                 auxiliary_head,
                 train_cfg,
                 test_cfg,
                 data_preprocessor,
                 pretrained,
                 init_cfg)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        x_h1 = self.decode_head.forward(x)
        if self.auxiliary_head is not None:
            x_h2 = self.auxiliary_head.forward(x)
            return x_h1, x_h2
        return x_h1