# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod

import torch
from mmengine.model import BaseModule
from mmseg.models.builder import build_loss
from mmseg.structures import build_pixel_sampler
from torch import nn

from baseg.models.utils import resize


class CustomBaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Custom class for BaseDecodeHead to simply remove the loss from the head."""

    def __init__(
        self,
        in_channels,
        channels,
        *,
        num_classes,
        aux_classes=None,
        out_channels=None,
        threshold=None,
        dropout_ratio=0.1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        in_index=-1,
        input_transform=None,
        loss_decode=None,
        ignore_index=255,
        sampler=None,
        align_corners=False,
        init_cfg=dict(type="Normal", std=0.01, override=dict(name="conv_seg")),
    ):
        super().__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if out_channels is None:
            if num_classes == 2:
                warnings.warn(
                    "For binary segmentation, we suggest using"
                    "`out_channels = 1` to define the output"
                    "channels of segmentor, and use `threshold`"
                    "to convert `seg_logits` into a prediction"
                    "applying a threshold"
                )
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                "out_channels should be equal to num_classes,"
                "except binary segmentation set out_channels == 1 and"
                f"num_classes == 2, but got out_channels={out_channels}"
                f"and num_classes={num_classes}"
            )

        if out_channels == 1 and threshold is None:
            threshold = 0.5
            warnings.warn("threshold is not defined for binary, and defaults" "to 0.5")
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            warnings.warn("Loss not instantiated, use manual .forward() calls")
            self.loss_decode = None

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if aux_classes is not None:
            self.conv_seg_aux = nn.Conv2d(channels, aux_classes, kernel_size=1)
        else:
            self.conv_seg_aux = None
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @abstractmethod
    def forward(self, inputs, return_feat: bool = False):
        """Placeholder of forward function."""
        pass

    def has_aux_output(self):
        """Whether the head has auxiliary output."""
        return self.conv_seg_aux is not None

    def cls_seg(self, feat: torch.Tensor) -> torch.Tensor:
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def cls_seg_aux(self, feat: torch.Tensor) -> torch.Tensor:
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg_aux(feat)
        return output
