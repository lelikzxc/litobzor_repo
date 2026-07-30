"""Quick test: do gradients actually update weights?"""
import torch
from papers.radon_cnn.models.radon_cnn import RadonCNN

m = RadonCNN(1, 7, 64)
opt = torch.optim.Adam(m.parameters(), lr=0.01)
x = torch.randn(4, 1, 64, 64)
y = torch.randint(0, 7, (4,))

w_before = m.classifier.fc3.weight.clone()

for step in range(5):
    opt.zero_grad()
    out = m(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    opt.step()
    grad_sum = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print(f"Step {step}: loss={loss.item():.4f}, grad_sum={grad_sum:.6f}")

w_after = m.classifier.fc3.weight
print(f"\nWeights changed: {not torch.allclose(w_before, w_after)}")
print(f"Max weight diff: {(w_before - w_after).abs().max().item():.8f}")