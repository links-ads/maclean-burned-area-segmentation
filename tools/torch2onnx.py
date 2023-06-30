from pathlib import Path

import numpy as np
import onnx
import torch
from argdantic import ArgField, ArgParser
from loguru import logger as log
from torch import nn


class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x


nn.AdaptiveAvgPool2d = AdaptiveAvgPool2dCustom


cli = ArgParser(name="torch2onnx", description="Convert a PyTorch model to ONNX format.")


@cli.command()
def torch2onnx(
    config: Path = ArgField("-c", description="Path to the experiment config file."),
    checkpoint: Path = ArgField("-m", description="Path to the checkpoint file."),
    include_aux: bool = ArgField("-aux", description="Include auxiliary heads.", default=False),
    work_dir: Path = ArgField("-w", default=None, description="Working directory."),
    out_name: str = ArgField("-o", default=None, description="Optional ONNX model name."),
    opset_version: int = ArgField("-ov", description="ONNX opset version.", default=11),
    input_names: list[str] = ArgField("-in", default=["input"], description="Input names."),
    output_names: list[str] = ArgField("-out", default=["output"], description="Output names."),
    device: str = ArgField("-d", description="Device.", default="cpu"),
    optimize: bool = ArgField("-opt", description="ONNX Optimize.", default=False),
    run_model: bool = ArgField("-run", description="Validate ONNX model.", default=False),
):
    from mmengine import Config

    from baseg.modules import MultiTaskModule, SingleTaskModule

    log.info(f"Converting {checkpoint} to ONNX format.")
    assert checkpoint.exists() and checkpoint.is_file(), f"{checkpoint} is not a file."
    assert config.exists() and config.is_file(), f"{config} is not a file."
    assert device == "cpu" or device.startswith("cuda"), f"{device} is not a valid device."

    if work_dir is not None:
        assert work_dir.exists() and work_dir.is_dir(), f"{work_dir} is not a directory."
    else:
        work_dir = config.parent
    log.info(f"Working directory: {work_dir}")

    config = Config.fromfile(config)
    if not include_aux:
        config.model.decode_head.pop("aux_classes", None)
        config.model.decode_head.pop("aux_factor", None)

    # prepare the model
    log.info(f"Loading model from {checkpoint}")
    model_config = config["model"]
    module_class = MultiTaskModule if "aux_classes" in model_config["decode_head"] else SingleTaskModule
    string_loading = include_aux
    module = module_class.load_from_checkpoint(
        checkpoint,
        config=model_config,
        loss="bce",
        map_location=device,
        strict=string_loading,
    )
    module.eval()

    # prepare inputs and outputs
    log.info("Preparing inputs and outputs")
    inputs = torch.randn(1, 12, 512, 512, dtype=torch.float32, requires_grad=True).to(device)
    outputs = module.model(inputs)
    log.info(f"Input shape: {inputs.shape}")
    log.info(f"Output shape: {outputs.shape}")

    # export to ONNX
    log.info("Exporting to ONNX")
    out_name = out_name or f"{checkpoint.stem}"
    out_path = work_dir / f"{out_name}.onnx"
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    # input_metas = {"shape": inputs.shape, "dtype": str(inputs.dtype), "mode": "predict"}

    log.info(f"Saving ONNX model to {out_path}")
    torch.onnx.export(
        module.model,
        inputs,
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=optimize,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    log.info(f"ONNX model saved to {out_path}")
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    log.info("Model checked!")

    if run_model:
        import onnxruntime as rt

        log.info("Validating ONNX model")
        provider = "CUDAExecutionProvider" if device.startswith("cuda") else "CPUExecutionProvider"
        ort_session = rt.InferenceSession(str(out_path), providers=[provider])
        ort_inputs = {ort_session.get_inputs()[0].name: inputs.detach().cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        log.info("torch output shape: {}".format(outputs.shape))
        log.info("onnx output shape: {}".format(ort_outs[0].shape))
        log.info("torch output: {}".format(outputs.detach().cpu().numpy()))
        log.info("onnx output: {}".format(ort_outs[0]))

        np.testing.assert_allclose(outputs.detach().cpu().numpy(), ort_outs[0], rtol=1e-02, atol=1e-03)
        log.info("ONNX model validated!")


if __name__ == "__main__":
    cli()
