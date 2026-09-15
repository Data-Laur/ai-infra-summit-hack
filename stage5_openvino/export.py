"""Export a trained LeRobot ACT checkpoint to OpenVINO IR (PyTorch -> ONNX -> IR).

    python -m stage5_openvino.export
    python -m stage5_openvino.export --checkpoint outputs/train/<job>/checkpoints/last/pretrained_model

Inputs of the exported graph (see stage3_policy/README.md, "Lauren (OpenVINO)"):
    observation.state            [1, 12] float32, normalised by the saved preprocessor
    observation.images.overhead  [1, 3, 480, 640] float32 in 0..1, normalised by the saved preprocessor
Output:
    action                       [1, 12] normalised joint targets (first step of the ACT chunk);
                                 the saved postprocessor maps them back to radians.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stage3_policy.learned import schema

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "train" / "act_butler_smoke" / "checkpoints" / "last" / "pretrained_model"
DEFAULT_ONNX = REPO_ROOT / "assets" / "models" / "act_pick.onnx"
DEFAULT_IR = REPO_ROOT / "assets" / "models" / "policy_openvino.xml"  # configs/default.yaml: models.openvino_ir
STATE_SHAPE = (1, len(schema.MOTOR_NAMES))
IMAGE_SHAPE = (1, schema.IMAGE_SHAPE[2], schema.IMAGE_SHAPE[0], schema.IMAGE_SHAPE[1])  # NCHW


def build_wrapper(checkpoint: Path):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    policy = ACTPolicy.from_pretrained(str(checkpoint))
    policy.eval()

    class ActFirstAction(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, state, image):
            actions = self.model({OBS_STATE: state, OBS_IMAGES: [image]})[0]
            return actions[:, 0, :]

    return ActFirstAction(policy.model).eval()


def export(checkpoint: Path, onnx_path: Path, ir_path: Path, compress_to_fp16: bool) -> int:
    import openvino as ov
    import torch

    wrapper = build_wrapper(checkpoint)
    state = torch.zeros(STATE_SHAPE, dtype=torch.float32)
    image = torch.rand(IMAGE_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        reference = wrapper(state, image).numpy()
    print(f"pytorch output: {tuple(reference.shape)} {reference.dtype}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (state, image),
        str(onnx_path),
        input_names=[schema.STATE_KEY, schema.IMAGE_KEY],
        output_names=[schema.ACTION_KEY],
        dynamo=True,
    )
    data_path = onnx_path.with_name(onnx_path.name + ".data")  # the exporter stores weights beside the graph
    data_mb = data_path.stat().st_size / 1e6 if data_path.exists() else 0.0
    print(f"onnx: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB graph + {data_mb:.1f} MB {data_path.name})")

    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(ir_path), compress_to_fp16=compress_to_fp16)
    bin_path = ir_path.with_suffix(".bin")
    print(f"ir: {ir_path} + {bin_path.name} ({bin_path.stat().st_size / 1e6:.1f} MB, fp16 weights={compress_to_fp16})")

    core = ov.Core()
    inputs = {schema.STATE_KEY: state.numpy(), schema.IMAGE_KEY: image.numpy()}
    # Correctness check at f32: the CPU plugin defaults to f16 inference on ARM Macs, which adds ~1e-2 noise.
    f32_hint = {"INFERENCE_PRECISION_HINT": "f32"}
    fp32_output = next(iter(core.compile_model(model, "CPU", f32_hint)(inputs).values()))
    fp32_diff = float(np.max(np.abs(fp32_output - reference)))
    compiled = core.compile_model(str(ir_path), "CPU")
    for port in compiled.inputs:
        print(f"ir input  {port.any_name}: {list(port.shape)} {port.element_type}")
    for port in compiled.outputs:
        print(f"ir output {port.any_name}: {list(port.shape)} {port.element_type}")
    output = next(iter(compiled(inputs).values()))
    saved_diff = float(np.max(np.abs(output - reference)))
    precision = compiled.get_property("INFERENCE_PRECISION_HINT")
    print(f"max |openvino - pytorch|: {fp32_diff:.3e} at f32; {saved_diff:.3e} for the saved IR at this CPU's default ({precision})")

    problems = []
    if [list(p.shape) for p in compiled.inputs] != [list(STATE_SHAPE), list(IMAGE_SHAPE)]:
        problems.append(f"input shapes are not {[list(STATE_SHAPE), list(IMAGE_SHAPE)]}")
    if list(compiled.outputs[0].shape) != list(STATE_SHAPE):
        problems.append(f"output shape is not {list(STATE_SHAPE)}")
    if fp32_diff > 1e-3:
        problems.append(f"fp32 OpenVINO graph does not match PyTorch (max diff {fp32_diff:.3e} > 1e-3)")
    if not np.isfinite(output).all():
        problems.append("saved IR produced non-finite values")
    print("export:", "OK" if not problems else "FAILED")
    for problem in problems:
        print(f"  - {problem}")
    return 0 if not problems else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stage5_openvino.export", description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="LeRobot pretrained_model directory")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--fp32", action="store_true", help="keep fp32 weights instead of compressing to fp16")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        print(f"no checkpoint at {checkpoint} (expected config.json + model.safetensors)")
        return 2
    return export(checkpoint, args.onnx, args.ir, compress_to_fp16=not args.fp32)


if __name__ == "__main__":
    sys.exit(main())
