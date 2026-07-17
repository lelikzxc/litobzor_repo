"""Vendored official VMamba implementation (trimmed).

Source: https://github.com/MzeroMiko/VMamba (classification/models/vmamba.py)

Contains only the classes needed for SS2D and VSSBlock:
    - ``Linear2d``, ``LayerNorm2d``, ``Permute``, ``Mlp``, ``gMlp``
    - ``mamba_init`` (parameter initialisation helpers)
    - ``SS2Dv0``, ``SS2Dv2``, ``SS2Dv3``, ``SS2Dm0`` (SS2D version mixins)
    - ``SS2D`` (multi-version 2D Selective Scan)
    - ``VSSBlock`` (official VMamba State-Space block)
    - ``DropPath`` (stochastic depth)

All imports use the vendored ``papers.vmamba.kernels`` package for
``selective_scan_fn``, ``cross_scan_fn``, and ``cross_merge_fn``.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from papers.vmamba.kernels import cross_merge_fn, cross_scan_fn, selective_scan_fn


# =====================================================
# Helper layers
# =====================================================


class Linear2d(nn.Linear):
    """Linear layer that operates on 2D spatial inputs via grouped conv."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm that operates on 2D spatial inputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class Permute(nn.Module):
    """Dimension permutation layer."""

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims).contiguous()


class Mlp(nn.Module):
    """MLP with optional channel-first support."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        channels_first: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    """Gated MLP."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        channels_first: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    """Softmax over spatial dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


# =====================================================
# mamba_init: parameter initialisation helpers
# =====================================================


class mamba_init:
    """Parameter initialisation for SSM (dt, A, D)."""

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank ** (-0.5) * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(
        d_state: int,
        d_inner: int,
        copies: int = -1,
        device: torch.device | None = None,
        merge: bool = True,
    ) -> nn.Parameter:
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True  # type: ignore[attr-defined]
        return A_log

    @staticmethod
    def D_init(
        d_inner: int,
        copies: int = -1,
        device: torch.device | None = None,
        merge: bool = True,
    ) -> nn.Parameter:
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True  # type: ignore[attr-defined]
        return D

    @classmethod
    def init_dt_A_D(
        cls,
        d_state: int,
        dt_rank: int,
        d_inner: int,
        dt_scale: float,
        dt_init: str,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        k_group: int = 4,
    ) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs

        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# =====================================================
# SS2D version mixins
# =====================================================


class SS2Dv0:
    """SS2D version v0 (original Mamba-style selective scan)."""

    def __initv0__(
        self,
        d_model: int = 96,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        dropout: float = 0.0,
        seq: bool = False,
        force_fp32: bool = True,
        **kwargs,
    ) -> None:
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()  # type: ignore[call-arg]
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0  # type: ignore[attr-defined]
        if seq:
            self.forward = partial(self.forwardv0, seq=True)  # type: ignore[attr-defined]
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)  # type: ignore[attr-defined]

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False) for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq: bool = False, force_fp32: bool = True, **kwargs) -> torch.Tensor:
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend="mamba")

        B, D, H, W = x.shape
        N = self.A_logs.shape[1]
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()

        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i],
                    dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i],
                    Bs[:, i].unsqueeze(1),
                    Cs[:, i].unsqueeze(1),
                    Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2Dv2:
    """SS2D version v2 (default — uses cross_scan/cross_merge + selective_scan)."""

    def __initv2__(
        self,
        d_model: int = 96,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        act_layer: type[nn.Module] = nn.SiLU,
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        initialize: str = "v0",
        forward_type: str = "v2",
        channel_first: bool = False,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()  # type: ignore[call-arg]
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2  # type: ignore[attr-defined]

        # tags for forward_type
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type dispatch
        FORWARD_TYPES: dict[str, Any] = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False),
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # out proj
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor | None = None,
        force_fp32: bool = False,
        ssoflex: bool = True,
        no_einsum: bool = False,
        selective_scan_backend: str | None = None,
        scan_mode: str | int = "cross2d",
        scan_force_torch: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert selective_scan_backend in [None, "oflex", "mamba", "torch", "core"], f"Unknown backend: {selective_scan_backend}"
        _scan_mode: int = (
            {"cross2d": 0, "unidi": 1, "bidi": 2, "cascade2d": -1}.get(scan_mode, None)  # type: ignore[arg-type]
            if isinstance(scan_mode, str)
            else scan_mode
        )
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)

        if _scan_mode == -1:
            # cascade2d mode (row then column)
            x_proj_bias = getattr(self, "x_proj_bias", None)

            def scan_rowcol(x, proj_weight, proj_bias, dt_weight, dt_bias, _As, _Ds, width=True):
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)
                if no_einsum:
                    x_dbl = F.conv1d(
                        xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                        bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2,
                    )
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(_B, 2, -1, _L)
                return ys

            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous()
            y_row = scan_rowcol(
                x,
                proj_weight=self.x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous()
            y_col = scan_rowcol(
                y_row,
                proj_weight=self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(
                    xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1),
                    bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K,
                )
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()
            Ds = self.Ds.to(torch.float)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, H, W)
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(
        forward_type: str = "", d_inner: int = 192, channel_first: bool = True
    ) -> tuple[nn.Module, str]:
        def checkpostfix(tag: str, value: str) -> tuple[bool, str]:
            ret = value[-len(tag):] == tag
            if ret:
                value = value[: -len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm: nn.Module = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag: str, value: str) -> tuple[bool, str]:
        ret = value[-len(tag):] == tag
        if ret:
            value = value[: -len(tag)]
        return ret, value


class SS2Dv3:
    """SS2D version v3 (xv variant — placeholder, delegates to v2)."""

    def __initxv__(self, **kwargs) -> None:
        super().__init__()  # type: ignore[call-arg]
        # Simplified: delegate to v2 init
        self.__initv2__(**kwargs)  # type: ignore[attr-defined]

    def forwardxv(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forwardv2(x, **kwargs)  # type: ignore[attr-defined]


class SS2Dm0:
    """SS2D version m0 (mamba2 variant — placeholder, delegates to v2)."""

    def __initm0__(self, **kwargs) -> None:
        super().__init__()  # type: ignore[call-arg]
        self.__initv2__(**kwargs)  # type: ignore[attr-defined]

    def forwardm0(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forwardv2(x, **kwargs)  # type: ignore[attr-defined]


# =====================================================
# SS2D: multi-version 2D Selective Scan
# =====================================================


class SS2D(nn.Module, SS2Dv0, SS2Dv2, SS2Dv3, SS2Dm0):
    """2D Selective Scan Module (official VMamba implementation).

    Supports multiple forward versions selected via ``forward_type``:
        - ``"v0"`` / ``"v0seq"``: Original Mamba-style selective scan
        - ``"v2"`` (default): Cross-scan + cross-merge + selective scan
        - ``"xv*"``: Extended variant (delegates to v2)
        - ``"m*"``: Mamba2 variant (delegates to v2)

    Args:
        d_model: Model / channel dimension.
        d_state: State dimension (default: 16).
        ssm_ratio: SSM expansion ratio (default: 2.0).
        dt_rank: Rank of dt projection (``"auto"`` = ``ceil(d_model / 16)``).
        act_layer: Activation layer (default: ``SiLU``).
        d_conv: Depthwise convolution kernel size (default: 3; ``<2`` disables).
        conv_bias: Whether conv has bias.
        dropout: Dropout rate.
        bias: Whether linear layers have bias.
        dt_min: Minimum dt value for init.
        dt_max: Maximum dt value for init.
        dt_init: dt initialisation strategy.
        dt_scale: dt initialisation scale.
        dt_init_floor: Floor for dt init.
        initialize: Initialisation version (``"v0"``, ``"v1"``, ``"v2"``).
        forward_type: Forward version (default: ``"v2"``).
        channel_first: Whether to use channel-first layout.
    """

    def __init__(
        self,
        d_model: int = 96,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        act_layer: type[nn.Module] = nn.SiLU,
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        initialize: str = "v0",
        forward_type: str = "v2",
        channel_first: bool = False,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(**kwargs)
        else:
            self.__initv2__(**kwargs)


# =====================================================
# VSSBlock: official VMamba State-Space block
# =====================================================


class DropPath(nn.Module):
    """Stochastic depth (drop path) for residual blocks.

    Args:
        drop_prob: Probability of dropping a path.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * mask


class VSSBlock(nn.Module):
    """Official VMamba State-Space Block.

    Applies: LayerNorm → SS2D → DropPath → residual → LayerNorm → MLP → DropPath → residual.

    Args:
        hidden_dim: Hidden / model dimension.
        drop_path: Stochastic depth drop rate.
        norm_layer: Normalisation layer.
        channel_first: Whether to use channel-first layout.
        ssm_d_state: SSM state dimension.
        ssm_ratio: SSM expansion ratio.
        ssm_dt_rank: dt projection rank.
        ssm_act_layer: SSM activation layer.
        ssm_conv: Depthwise convolution kernel size.
        ssm_conv_bias: Whether conv has bias.
        ssm_drop_rate: SSM dropout rate.
        ssm_init: SSM initialisation version.
        forward_type: SS2D forward version.
        mlp_ratio: MLP hidden dimension ratio.
        mlp_act_layer: MLP activation layer.
        mlp_drop_rate: MLP dropout rate.
        gmlp: Use gated MLP.
        use_checkpoint: Use gradient checkpointing.
        post_norm: Apply norm after (instead of before) SS2D/MLP.
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        channel_first: bool = False,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer: type[nn.Module] = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0,
        ssm_init: str = "v0",
        forward_type: str = "v2",
        mlp_ratio: float = 4.0,
        mlp_act_layer: type[nn.Module] = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        _SS2D: type = SS2D,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=channel_first,
            )

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)  # type: ignore[attr-defined]
        else:
            return self._forward(input)