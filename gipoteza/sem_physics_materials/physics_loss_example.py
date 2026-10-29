"""Compact, executable illustration of the historical strong_soft loss.

No SEM data or model training. The smoke check uses synthetic tensors only.
Historical results come from the preserved reports, never from this example.
"""
from pathlib import Path
import argparse
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


def make_segmenter():
    """The 10,681-parameter CNN used in the SEM comparison."""
    return nn.Sequential(
        nn.Conv2d(1, 24, 3, padding=1), nn.GELU(),
        nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.GELU(),
        nn.Conv2d(24, 24, 3, padding=4, dilation=4), nn.GELU(),
        nn.Conv2d(24, 1, 1),
    )


def soft_conductance(probability):
    """Normalized left-to-right conductance of a four-neighbour 12x12 grid.

    Boundary potentials: left=0, right=1. Edges have conductance min(p_i,p_j).
    The small linear system is solved on CPU in float64, preserving autograd.
    This is the positive continuous relaxation; hard disconnected masks need
    a separate connectivity-aware solver and must not be labelled by this one.
    """
    assert probability.ndim == 4 and probability.shape[1:] == (1, 12, 12)
    device, dtype = probability.device, probability.dtype
    size = 12
    q = probability.reshape(-1, size * size).to(device='cpu', dtype=torch.float64)
    q = q.clamp(1e-5, 1 - 1e-5)
    edges = [(y * size + x, yy * size + xx)
             for y in range(size) for x in range(size)
             for yy, xx in ((y + 1, x), (y, x + 1)) if yy < size and xx < size]
    src = torch.tensor([u for u, _ in edges])
    dst = torch.tensor([v for _, v in edges])
    incidence = torch.zeros((len(edges), size * size), dtype=q.dtype)
    incidence[torch.arange(len(edges)), src] = 1
    incidence[torch.arange(len(edges)), dst] = 1
    weight = torch.minimum(q[:, src], q[:, dst])
    laplacian = torch.diag_embed(weight @ incidence)
    laplacian[:, src, dst] = -weight
    laplacian[:, dst, src] = -weight
    free = torch.tensor([y * size + x for y in range(size) for x in range(1, size - 1)])
    right = torch.tensor([y * size + size - 1 for y in range(size)])
    a = laplacian[:, free][:, :, free]
    b = -laplacian[:, free][:, :, right].sum(-1)
    voltage = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
    current = laplacian[:, right][:, :, right].sum((1, 2)) - (b * voltage).sum(-1)
    return (current / (size / (size - 1))).clamp_min(1e-8).to(device=device, dtype=dtype)


def geometry_loss(logits, directional_class):
    """directional_class: int64 [N,2], each entry in {0,1,2}."""
    assert directional_class.shape == (len(logits), 2)
    assert directional_class.dtype == torch.long
    assert bool(((directional_class >= 0) & (directional_class <= 2)).all())
    p = torch.sigmoid(4.0 * logits)
    g = torch.stack((soft_conductance(p), soft_conductance(p.transpose(-1, -2))), dim=1)
    g0 = 0.5 / ((12 * 12 - 1) * (12 / 11))
    ordinal_logits = torch.stack((4.0 * (g.log() - math.log(g0)), 40.0 * (g - 0.5)), dim=-1)
    thresholds = torch.arange(2, device=logits.device)
    target = (directional_class.unsqueeze(-1) > thresholds).float()
    return F.binary_cross_entropy_with_logits(ordinal_logits, target)


def smoke_check():
    """Check analytic grids, original source agreement and gradient propagation."""
    torch.set_num_threads(2)
    torch.manual_seed(2701)
    levels = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    uniform = levels[:, None, None, None].expand(-1, 1, 12, 12)
    uniform_error = float((soft_conductance(uniform) - levels).abs().max())
    bars = torch.zeros((11, 1, 12, 12), dtype=torch.float64)
    for width in range(1, 12):
        bars[width - 1, 0, :width, :] = 1
    bar_error = float((soft_conductance(bars) - torch.arange(1, 12) / 12).abs().max())
    # Compare to the exact archived function without executing its module.
    import ast
    import zipfile
    import numpy as np
    archive = Path(__file__).resolve().parents[1] / 'retained_evidence_20261027/SELECTED_SOURCES.zip'
    with zipfile.ZipFile(archive) as z:
        text = z.read('gipoteza/validation_20261026/baselines/soft_grid.py').decode()
    tree = ast.parse(text)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('response', 'auxiliary')]
    ids = torch.arange(144).reshape(12, 12)
    src = torch.cat((ids[:-1].reshape(-1), ids[:, :-1].reshape(-1)))
    dst = torch.cat((ids[1:].reshape(-1), ids[:, 1:].reshape(-1)))
    incidence = torch.zeros((len(src), 144), dtype=torch.float64)
    incidence[torch.arange(len(src)), src] = 1
    incidence[torch.arange(len(src)), dst] = 1
    from types import SimpleNamespace
    scope = dict(torch=torch, np=np, F=F, SRC=src, DST=dst, ABS=incidence,
                 FREE=ids[:, 1:-1].reshape(-1), RIGHT=ids[:, -1],
                 G0=0.5 / (143 * (12 / 11)),
                 e6=SimpleNamespace(physics=SimpleNamespace(FULL=12 / 11)))
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, '<archived response and auxiliary only>', 'exec'), scope)
    logits = torch.randn((3, 1, 12, 12), dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0, 1], [1, 2], [2, 0]])
    loss = geometry_loss(logits, targets)
    reference = scope['auxiliary'](logits, targets, dict(kappa=4., family='log', a=4., b=40., kernel='min'))
    gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, logits)[0]
    loss_error = abs(float(loss.detach() - reference.detach()))
    gradient_error = float((gradient - reference_gradient).abs().max())
    assert uniform_error < 1e-10 and bar_error < 3e-4
    assert loss_error < 1e-10 and gradient_error < 1e-10, (loss_error, gradient_error)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    net = make_segmenter()
    assert sum(p.numel() for p in net.parameters()) == 10681
    assert net(torch.zeros((2, 1, 12, 12))).shape == (2, 1, 12, 12)
    return {'status': 'PASS', 'model_parameters': 10681, 'uniform_grid_max_error': uniform_error,
            'binary_bar_approximation_max_error': bar_error,
            'archived_loss_absolute_error': loss_error,
            'archived_gradient_max_error': gradient_error,
            'finite_nonzero_gradients': True,
            'scope': 'Synthetic tensors and archived-function equivalence only; no SEM training or empirical performance result.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', required=True)
    args = parser.parse_args()
    result = smoke_check()
    out = Path(__file__).resolve().parent / 'CODE_CHECK.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))
