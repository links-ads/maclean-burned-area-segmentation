from typing import Optional
from torch import Tensor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList)
from mmseg.registry import MODELS
from torch.nn import functional as F    
@MODELS.register_module()
class CustomEncoderDecoder(EncoderDecoder):

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
        x_h1 = F.interpolate(x_h1, size=inputs.shape[2:], mode="bilinear", align_corners=True)
        if self.auxiliary_head is not None:
            x_h2 = self.auxiliary_head.forward(x)
            x_h2 = F.interpolate(x_h2, size=inputs.shape[2:], mode="bilinear", align_corners=True)
            return x_h1, x_h2
        return x_h1