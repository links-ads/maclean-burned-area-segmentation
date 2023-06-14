from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.utils import OptSampleList
from torch import Tensor
from torch.nn import functional as F


@MODELS.register_module()
class CustomEncoderDecoder(EncoderDecoder):
    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Tensor:
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
        feat = self.decode_head(x)
        out = self.decode_head.cls_seg(feat)
        out = F.interpolate(out, size=inputs.shape[2:], mode="bilinear", align_corners=True)

        if self.decode_head.has_aux_output():
            aux = self.decode_head.cls_seg_aux(feat)
            aux = F.interpolate(aux, size=inputs.shape[2:], mode="bilinear", align_corners=True)
            return out, aux

        return out
