"""C4 weight sharing inside the segmentation CNN, without a new loss.

Channel layout is (base feature, orientation), flattened in that order.
Group action: F_r(R_k x) = R_k F_(r-k)(x). Inputs and masks are scalar
fields. Quarter turns need no image interpolation. This is a known group-CNN
construction; this file makes no originality claim.
"""
import math
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


class LiftingConv(nn.Module):
    def __init__(self, features=6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features, 1, 3, 3))
        self.bias = nn.Parameter(torch.empty(features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / 3, 1 / 3)

    def expanded(self):
        weight = torch.stack([torch.rot90(self.weight, r, (-2, -1))
                              for r in range(4)], dim=1)
        return weight.reshape(-1, 1, 3, 3).contiguous(), self.bias.repeat_interleave(4)

    def forward(self, x):
        weight, bias = self.expanded()
        return F.conv2d(x, weight, bias, padding=1)


class GroupConv(nn.Module):
    def __init__(self, features=6, dilation=1):
        super().__init__()
        self.features = features
        self.dilation = dilation
        # One kernel for each relative input orientation, shared across the
        # four absolute output orientations. Fan-in is 6*4*3*3, not6*3*3.
        self.weight = nn.Parameter(torch.empty(features, features, 4, 3, 3))
        self.bias = nn.Parameter(torch.empty(features))
        bound = 1 / math.sqrt(features * 4 * 9)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def expanded(self):
        orientations = []
        for r in range(4):
            relative = [torch.rot90(self.weight[:, :, (t - r) % 4],
                                    r, (-2, -1)) for t in range(4)]
            orientations.append(torch.stack(relative, dim=2))
        weight = torch.stack(orientations, dim=1)
        return (weight.reshape(self.features * 4, self.features * 4, 3, 3).contiguous(),
                self.bias.repeat_interleave(4))

    def forward(self, x):
        weight, bias = self.expanded()
        return F.conv2d(x, weight, bias, padding=self.dilation,
                        dilation=self.dilation)


class GroupCNN(nn.Module):
    def __init__(self, features=6):
        super().__init__()
        self.features = features
        self.lift = LiftingConv(features)
        self.group2 = GroupConv(features, dilation=2)
        self.group4 = GroupConv(features, dilation=4)
        self.head = nn.Conv2d(features, 1, 1)

    def forward_features(self, x):
        x = F.gelu(self.lift(x))
        x = F.gelu(self.group2(x))
        return F.gelu(self.group4(x))

    def forward(self, x):
        z = self.forward_features(x)
        pooled = z.reshape(len(z), self.features, 4, *z.shape[-2:]).mean(2)
        return self.head(pooled)

    @torch.no_grad()
    def deploy(self):
        """Materialize tied kernels once, returning ordinary dense convolutions.

        Training/checkpoint parameters remain2671; materialization stores
        repeated coefficients10681. Timing must state that setup is excluded.
        """
        result = make_plain(width=self.features * 4).to(device=self.head.weight.device,
                                 dtype=self.head.weight.dtype)
        for target, source in [(result[0], self.lift),
                               (result[2], self.group2), (result[4], self.group4)]:
            weight, bias = source.expanded()
            target.weight.copy_(weight); target.bias.copy_(bias)
        result[6].weight.copy_(self.head.weight.repeat_interleave(4, dim=1) / 4)
        result[6].bias.copy_(self.head.bias)
        return result.requires_grad_(False).eval()


def make_plain(width=24):
    return nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.GELU(),
        nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
        nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GELU(),
        nn.Conv2d(width, 1, 1))


def parameters(model):
    return sum(p.numel() for p in model.parameters())


TRAINING_ARMS = ['group6', 'plain12_C4pooled', 'plain24_C4pooled']
ROUTES = {'group6': [('group6_single', 'single')],
          'plain12_C4pooled': [('plain12_identity', 'identity'), ('plain12_C4', 'C4')],
          'plain24_C4pooled': [('plain24_identity', 'identity'), ('plain24_C4', 'C4')]}
FREE_PARAMETERS = {'group6': 2671, 'plain12_C4pooled': 2749, 'plain24_C4pooled': 10681}
MACS = {'group6_single': 1527552, 'plain12_identity': 390528, 'plain12_C4': 1562112,
        'plain24_identity': 1527552, 'plain24_C4': 6110208}


def state_digest(state):
    result = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous().numpy()
        result.update(key.encode()); result.update(str(value.dtype).encode())
        result.update(str(value.shape).encode()); result.update(value.tobytes())
    return result.hexdigest()


def make_model(architecture, seed):
    assert architecture in TRAINING_ARMS
    torch.manual_seed(int(seed))
    result = GroupCNN(6) if architecture == 'group6' else make_plain(12 if architecture == 'plain12_C4pooled' else 24)
    assert parameters(result) == FREE_PARAMETERS[architecture]
    return result


def aligned_view_logits(network, x):
    rotated = torch.stack([torch.rot90(x, k, (-2, -1)) for k in range(4)])
    output = network(rotated.reshape(4 * len(x), *x.shape[1:])).reshape(4, len(x), 1, *x.shape[-2:])
    return torch.stack([torch.rot90(output[k], -k, (-2, -1)) for k in range(4)])


def training_details(network, architecture, x):
    if architecture == 'group6':
        logits = network(x)
        return dict(logits=logits, probability=logits.sigmoid(), aligned_logits=None)
    views = aligned_view_logits(network, x)
    logits = torch.logsumexp(F.logsigmoid(views), 0) - torch.logsumexp(F.logsigmoid(-views), 0)
    return dict(logits=logits, probability=views.sigmoid().mean(0), aligned_logits=views)


def predict_probability(network, x, mode):
    if mode in ['single', 'identity']:
        return network(x).sigmoid()
    assert mode == 'C4'
    return aligned_view_logits(network, x).sigmoid().mean(0)


def architecture_spec():
    return dict(family='known_C4_group_CNN', base_features=6, orientations=4,
        feature_channels=24, trainable_parameters=2671,
        original_parameters=10681, materialized_parameters=10681,
        narrow_comparator_parameters=2749,
        training_arms=TRAINING_ARMS, routes=ROUTES, MACs_per_image=MACS,
        kernel_sizes=[3, 3, 3, 1], dilations=[1, 2, 4, 1],
        activations='GELU after each3x3layer', orientation_pool='mean before1x1head',
        initialization='fresh fan-in-matched; no projection of historical CNN weights',
        plain_C4_execution='one batchedCNNcall on4*Nviews, inversealign, meanprobability',
        objective='unchanged historical BCE+geometry_loss',
        claim='Architecture/efficiency experiment; no new loss or originality claim')
