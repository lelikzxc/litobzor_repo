"""Tiny Vision Transformer model for wafer map defect classification.

Reference: "Semiconductor Wafer Map Defect Classification with
           Tiny Vision Transformers" (arXiv:2504.02494)

Architecture:
    - Patch Embedding (Conv2d)
    - Learnable CLS token
    - Learnable Positional Embedding
    - Transformer Encoder blocks (LayerNorm → MSA → residual → MLP → residual)
    - Final LayerNorm
    - Classification head (CLS token → LayerNorm → Linear)

The model returns logits only — no Softmax applied.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings via Conv2d.

    Uses a convolutional layer with kernel_size = stride = patch_size
    to extract non-overlapping patch embeddings.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Patch embeddings of shape [B, num_patches, embed_dim].
        """
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with dropout."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape [B, N, embed_dim].

        Returns:
            Output of shape [B, N, embed_dim].
        """
        B, N, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP with GELU activation and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape [B, N, in_dim].

        Returns:
            Output of shape [B, N, in_dim].
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block.

    LayerNorm → MSA → residual → LayerNorm → MLP → residual.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape [B, N, embed_dim].

        Returns:
            Output of shape [B, N, embed_dim].
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """Tiny Vision Transformer for wafer map defect classification.

    Implements the ViT-Tiny architecture described in:
        "Semiconductor Wafer Map Defect Classification with
         Tiny Vision Transformers" (arXiv:2504.02494)

    Architecture per the paper (Table II, Section III-A):
        - Layers: 12
        - Hidden size (embed_dim): 192
        - Attention heads: 3
        - MLP size: 768 (mlp_ratio=4.0)
        - Patch size: 16
        - Image size: 64×64 (Section III-B: N=16 patches, 4×4 grid)
        - Grayscale→RGB conversion via 1×1 Conv (Section III-A.4)

    The model returns logits only. Softmax is applied externally
    during inference (e.g., in predict.py).

    Args:
        image_size: Input image spatial dimension (H == W).
        patch_size: Size of each patch.
        in_channels: Number of input channels (1 for grayscale wafer maps).
        num_classes: Number of output classes.
        embed_dim: Embedding dimension.
        num_layers: Number of transformer encoder blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embed dim.
        dropout: Dropout rate for attention and MLP.
        emb_dropout: Dropout rate for positional embeddings.
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 8,
        embed_dim: int = 192,
        num_layers: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Store architecture parameters as class attributes
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes

        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        num_patches = (image_size // patch_size) ** 2

        # Grayscale → RGB conversion via 1×1 Conv (Section III-A.4)
        # "a 1×1 convolutional layer was used to convert single-channel
        #  grayscale images into 3-channel RGB images"
        self.grayscale_to_rgb = nn.Conv2d(1, 3, kernel_size=1)

        # Patch embedding (operates on 3-channel RGB)
        self.patch_embed = PatchEmbedding(3, embed_dim, patch_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(emb_dropout)

        # Transformer encoder blocks (12 layers per Table II)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear, conv, and layer norm weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Accepts grayscale (1-channel) or RGB (3-channel) input.
        Grayscale input is converted to RGB via 1×1 Conv (Section III-A.4).

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Logits of shape [B, num_classes].

        Raises:
            ValueError: If input tensor does not match expected dimensions.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input [B, C, H, W], got {x.dim()}D tensor "
                f"with shape {list(x.shape)}"
            )
        if x.shape[1] not in (1, 3):
            raise ValueError(
                f"Expected 1 (grayscale) or 3 (RGB) input channels, "
                f"got {x.shape[1]}"
            )
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            raise ValueError(
                f"Expected input spatial size {self.image_size}x{self.image_size}, "
                f"got {x.shape[2]}x{x.shape[3]}"
            )

        B = x.shape[0]

        # Grayscale → RGB conversion via 1×1 Conv (Section III-A.4)
        if x.shape[1] == 1:
            x = self.grayscale_to_rgb(x)  # [B, 3, H, W]

        # Patch embedding: [B, num_patches, embed_dim]
        x = self.patch_embed(x)

        # CLS token: [1, 1, embed_dim] → [B, 1, embed_dim]
        cls_token = self.cls_token.expand(B, -1, -1)

        # Concatenate CLS before patch embeddings
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        x = self.norm(x)

        # Classification head: use CLS token only
        cls_out = x[:, 0]  # [B, embed_dim]
        logits = self.head(cls_out)  # [B, num_classes]

        return logits

    @classmethod
    def from_config(cls, config: Any) -> ViTTiny:
        """Instantiate model from a config object.

        Args:
            config: Config object with ``model.arch.*`` attributes
                    accessible via ``config.get("model.arch.<key>")``.

        Returns:
            Configured ViTTiny instance.
        """
        return cls(
            image_size=config.get("model.arch.image_size", 64),
            patch_size=config.get("model.arch.patch_size", 16),
            in_channels=config.get("model.arch.in_channels", 1),
            num_classes=config.get("model.arch.num_classes", 8),
            embed_dim=config.get("model.arch.embed_dim", 192),
            num_layers=config.get("model.arch.num_layers", 12),
            num_heads=config.get("model.arch.num_heads", 3),
            mlp_ratio=config.get("model.arch.mlp_ratio", 4.0),
            dropout=config.get("model.arch.dropout", 0.1),
            emb_dropout=config.get("model.arch.emb_dropout", 0.1),
        )


class PretrainedViTTiny(nn.Module):
    """Pretrained Tiny Vision Transformer for wafer defect classification.

    Wraps the ImageNet-pretrained ``WinKawaks/vit-tiny-patch16-224`` model
    from HuggingFace and replaces its classification head with a new head
    for the WM-38k dataset (38 classes).

    This matches the paper's approach of fine-tuning a pretrained ViT-Tiny
    (Section III-A, "domain-specific fine-tuning of the pretrained ViT-Tiny
    model"). The architecture matches Table II:
        - Layers: 12, Hidden: 192, Heads: 3, MLP: 768
        - Patch size: 16, Image size: 224x224

    The model accepts RGB (3-channel) input of size 224x224 and returns
    logits of shape ``[B, num_classes]``.

    Args:
        num_classes: Number of output classes (38 for WM-38k).
        pretrained_name: HuggingFace model identifier for the pretrained
            ViT-Tiny. Defaults to ``WinKawaks/vit-tiny-patch16-224``.
        dropout: Dropout rate for the new classification head.
    """

    def __init__(
        self,
        num_classes: int = 38,
        pretrained_name: str = "WinKawaks/vit-tiny-patch16-224",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import ViTModel

        self.num_classes = num_classes
        self.pretrained_name = pretrained_name

        # Load pretrained ViT-Tiny backbone (no classification head)
        self.backbone = ViTModel.from_pretrained(pretrained_name)

        hidden_size = self.backbone.config.hidden_size  # 192

        # New classification head for WM-38k
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, 3, 224, 224]`` (RGB, normalized).

        Returns:
            Logits of shape ``[B, num_classes]``.
        """
        outputs = self.backbone(pixel_values=x)
        cls_out = outputs.last_hidden_state[:, 0]  # [B, hidden_size]
        logits = self.head(cls_out)  # [B, num_classes]
        return logits

    @classmethod
    def from_config(cls, config: Any) -> PretrainedViTTiny:
        """Instantiate a pretrained model from a config object.

        Args:
            config: Config object with ``model.arch.num_classes`` accessible via
                ``config.get("model.arch.num_classes")``.

        Returns:
            Configured ``PretrainedViTTiny`` instance.
        """
        return cls(
            num_classes=config.get("model.arch.num_classes", 38),
            pretrained_name=config.get(
                "model.arch.pretrained_name", "WinKawaks/vit-tiny-patch16-224"
            ),
            dropout=config.get("model.arch.dropout", 0.1),
        )