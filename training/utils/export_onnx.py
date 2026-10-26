"""
Exporta checkpoints treinados do ResNet18WithEmbed (e opcionalmente
ResNet18NoEmbed) para ONNX, para inferencia on-device no app Flutter.
Verifica numericamente que a saida do ONNX bate com a do PyTorch antes de
aceitar o arquivo exportado.

Uso:
    cd training
    python utils/export_onnx.py --checkpoints-dir models/ablation_ckpt_clean_swa \
        --tag with_embedding --seeds 42,43,44,45,46 --output-dir ../mobile_models
"""
import argparse
import os

import numpy as np
import onnxruntime as ort
import torch

from run_ablation import ResNet18NoEmbed, ResNet18WithEmbed


class WithEmbedWrapper(torch.nn.Module):
    """Softmax applied inside the graph so the app only has to argmax/read."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, region_idx):
        return torch.softmax(self.model(x, region_idx), dim=1)


class NoEmbedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def export_one(ckpt_path, out_path, with_embed, arch="resnet18"):
    model = (ResNet18WithEmbed(arch=arch) if with_embed else ResNet18NoEmbed(arch=arch))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    wrapper = (WithEmbedWrapper(model) if with_embed else NoEmbedWrapper(model)).eval()

    dummy_x = torch.randn(1, 3, 224, 224)
    if with_embed:
        dummy_region = torch.zeros(1, dtype=torch.long)
        args = (dummy_x, dummy_region)
        input_names = ["image", "region_idx"]
        dynamic_axes = {"image": {0: "batch"}, "region_idx": {0: "batch"}, "probs": {0: "batch"}}
    else:
        args = (dummy_x,)
        input_names = ["image"]
        dynamic_axes = {"image": {0: "batch"}, "probs": {0: "batch"}}

    torch.onnx.export(
        wrapper, args, out_path,
        input_names=input_names, output_names=["probs"],
        dynamic_axes=dynamic_axes, opset_version=17,
    )

    with torch.no_grad():
        torch_out = wrapper(*args).numpy()

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    feed = {input_names[i]: args[i].numpy() for i in range(len(args))}
    onnx_out = sess.run(None, feed)[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  {os.path.basename(out_path)}: max_diff={max_diff:.2e}  size={size_mb:.1f} MB")
    assert max_diff < 1e-4, f"ONNX output diverges from PyTorch by {max_diff}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", default="models/ablation_ckpt_clean_swa")
    p.add_argument("--tag", default="with_embedding", choices=["with_embedding", "no_embedding"])
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--output-dir", default="../mobile_models")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with_embed = args.tag == "with_embedding"
    prefix = "with_embedding_seed" if with_embed else "no_embedding_seed"

    for seed in args.seeds.split(","):
        ckpt = os.path.join(args.checkpoints_dir, f"{prefix}{seed}.pth")
        out = os.path.join(args.output_dir, f"region_model_seed{seed}.onnx")
        print(f"Exportando {ckpt} -> {out}")
        export_one(ckpt, out, with_embed, args.arch)

    print("\nOK: todos os modelos exportados batem numericamente com o PyTorch original.")


if __name__ == "__main__":
    main()
