"""Smoke demo for SemiWaferNet (HybridCNN-ViT + ConvoFormer-UNet)."""

from __future__ import annotations

import torch

from papers.semiwafernet.models import ConvoFormerUNet, SemiWaferNet
from papers.semiwafernet.utils.experiment import build_experiment_info, count_params, format_experiment_info


def main() -> None:
    print("1. HybridCNN-ViT (classification)")
    cls_model = SemiWaferNet(mode="classification")
    x_cls = torch.randn(2, 1, 32, 32)
    out_cls = cls_model(x_cls)
    print(f"   params={count_params(cls_model):,}")
    print(f"   classification={tuple(out_cls['classification'].shape)}")
    print(format_experiment_info(build_experiment_info(cls_model)))

    print("\n2. ConvoFormer-UNet (segmentation)")
    seg_model = SemiWaferNet(mode="segmentation")
    x_seg = torch.randn(2, 1, 64, 64)
    out_seg = seg_model(x_seg)
    print(f"   params={count_params(seg_model):,}")
    print(f"   segmentation={tuple(out_seg['segmentation'].shape)}")

    print("\n3. Standalone ConvoFormerUNet + deep supervision")
    unet = ConvoFormerUNet()
    aux = unet(x_seg, return_aux=True)
    print(f"   main={tuple(aux['main'].shape)} aux1={tuple(aux['aux1'].shape)} aux2={tuple(aux['aux2'].shape)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
