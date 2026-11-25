# The encoder the panel's JEPA, its untrained twin and its supervised model share: patches of
# hourly input rows, causal ConvNeXt-style 1-D blocks, causal attention with rotary positions and
# LayerNorm only. Each token sees only itself and earlier tokens; z is the last token's output.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROTARY_BASE = 100.0


@dataclass(frozen=True)
class Shape:
    patch: int
    width: int
    conv_blocks: int
    attention_blocks: int
    heads: int
    kernel: int
    expansion: int


class ConvBlock(nn.Module):
    def __init__(self, width: int, kernel: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(width, width, kernel, groups=width)
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, expansion * width)
        self.down = nn.Linear(expansion * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv(F.pad(x.transpose(1, 2), (self.kernel - 1, 0))).transpose(1, 2)
        return x + self.drop(self.down(F.gelu(self.up(self.norm(h)))))


def rotary(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    freq = ROTARY_BASE ** (-torch.arange(half, dtype=x.dtype) / half)
    angle = torch.arange(x.shape[-2], dtype=x.dtype)[:, None] * freq
    cos, sin = angle.cos(), angle.sin()
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, expansion * width)
        self.down = nn.Linear(expansion * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, d // self.heads)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(2))
        a = F.scaled_dot_product_attention(rotary(q), rotary(k), v, is_causal=True)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.down(F.gelu(self.up(self.norm2(x)))))


class Encoder(nn.Module):
    def __init__(self, channels: int, shape: Shape, dropout: float = 0.0) -> None:
        super().__init__()
        self.patch = shape.patch
        self.embed = nn.Linear(channels * shape.patch, shape.width)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(shape.width, shape.kernel, shape.expansion, dropout)
                for _ in range(shape.conv_blocks)
            ]
            + [
                AttentionBlock(shape.width, shape.heads, shape.expansion, dropout)
                for _ in range(shape.attention_blocks)
            ]
        )
        self.norm = nn.LayerNorm(shape.width)

    def tokens(self, x: Tensor) -> Tensor:
        b, length, c = x.shape
        h = self.embed(x.reshape(b, length // self.patch, self.patch * c))
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(self, x: Tensor) -> Tensor:
        return self.tokens(x)[:, -1]


class Predictor(nn.Module):
    def __init__(self, width: int, hidden: int, slots: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(slots, width))
        nn.init.normal_(self.query, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(2 * width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )

    def forward(self, z: Tensor) -> Tensor:
        slots = self.query.expand(z.shape[0], -1, -1)
        return self.net(torch.cat([z.unsqueeze(1).expand_as(slots), slots], dim=-1))


def encoder(channels: int, shape: Shape, seed: int, dropout: float = 0.0) -> Encoder:
    torch.manual_seed(seed)
    return Encoder(channels, shape, dropout)


def parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
