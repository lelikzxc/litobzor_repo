from papers.vit_tiny.models import ViTTiny
import torch

model = ViTTiny()

x = torch.randn(4,1,32,32)

y = model(x)

print(model)

print()

print("Output shape:", y.shape)

print("Sample logits:")

print(y)

print()

print("Parameters:", sum(p.numel() for p in model.parameters()))