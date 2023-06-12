# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from loguru import logger
from mmengine.runner import CheckpointLoader


def convert_resnet(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith("module."):
            logger.info(f"Removing 'module' prefix: {k}")
            k = k.replace("module.", "")

        if k.startswith("backbone."):
            logger.info(f"Removing 'backbone' prefix: {k}")
            k = k.replace("backbone.", "")

        if k.startswith("head"):
            logger.info(f"Skipping head layer: {k}")
            continue

        if k.startswith("norm"):
            logger.info(f"Converting norm layer: {k}")
            new_k = k.replace("norm.", "ln1.")
        elif k.startswith("patch_embed"):
            logger.info(f"Converting patch_embed layer: {k}")
            if "proj" in k:
                new_k = k.replace("proj", "projection")
            else:
                new_k = k
        elif k.startswith("blocks"):
            logger.info(f"Converting blocks layer: {k}")
            if "norm" in k:
                new_k = k.replace("norm", "ln")
            elif "mlp.fc1" in k:
                new_k = k.replace("mlp.fc1", "ffn.layers.0.0")
            elif "mlp.fc2" in k:
                new_k = k.replace("mlp.fc2", "ffn.layers.1")
            elif "attn.qkv" in k:
                new_k = k.replace("attn.qkv.", "attn.attn.in_proj_")
            elif "attn.proj" in k:
                new_k = k.replace("attn.proj", "attn.attn.out_proj")
            else:
                new_k = k
            new_k = new_k.replace("blocks.", "layers.")
        else:
            new_k = k
        new_ckpt[new_k] = v

    # convert to 12 L2A channels instead of 13 L1C channels
    logger.info("Converting patch_embed layer to 12 channels")
    inputs = new_ckpt["conv1.weight"]
    logger.info(f"Original conv1.weight shape: {inputs.shape}")
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    new_ckpt["conv1.weight"] = inputs[:, indices, :, :]
    logger.info(f"New conv1.weight shape: {new_ckpt['conv1.weight'].shape}")
    logger.info("Conversion finished:")
    for k, v in new_ckpt.items():
        logger.info(f"{k:<20s}: {v.shape}")
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Convert keys in timm pretrained vit models to " "MMSegmentation style."
    )
    parser.add_argument("src", help="src model path or url")
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument("dst", help="save path")
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location="cpu")
    if "state_dict" in checkpoint:
        # timm checkpoint
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        # deit checkpoint
        state_dict = checkpoint["model"]
    elif "teacher" in checkpoint:
        # SSL4EO DINO checkpoint
        state_dict = checkpoint["teacher"]
    else:
        state_dict = checkpoint
    weight = convert_resnet(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == "__main__":
    main()
    main()
