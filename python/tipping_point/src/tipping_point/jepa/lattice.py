"""The lattice side of the JEPA: spins on an L x L torus in, one latent per PATCH x PATCH cell out.

The torus's symmetries are built in rather than learned:
  translations  every convolution pads circularly, masks wrap, and there are no positional
                embeddings (a torus has no origin); the patch grid makes the encoder equivariant to
                rolls by multiples of PATCH, and random rolls in training cover the rest
  D4            augmentation, one random rotation or reflection per burst
  Z2            canonicalisation: a burst (training) or a frame (readouts) is flipped so its total
                magnetisation is positive; a tie goes by its first spin, so canonical(-s) equals
                canonical(s) exactly and every readout is even under a global flip

Shapes: frames (B, L, L) and bursts (B, F, L, L) of int8 +-1 spins; n = L / PATCH cells a side;
features (B, dim, n, n). Masks are in cells, so they mean the same at every L: target blocks
2-4 cells (4-8 sites) a side, wrapping around the torus, covering 20-25% of the cells, with a
one-cell buffer around them removed from the context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PATCH = 2  # sites per cell side


# ---------------------------------------------------------------- spins


def unpack(packed: Tensor, L: int) -> Tensor:
    """Spins packed as tipping_point.ising.io.pack does them (one bit per site, 1 = +1, row-major,
    most significant bit first) -> int8 +-1: (..., ceil(L*L/8)) uint8 -> (..., L, L), unpacked by
    shifts on whatever device `packed` is on, so bursts cross to the GPU at one bit per spin."""
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=packed.device)
    bits = torch.bitwise_right_shift(packed.unsqueeze(-1), shifts) & 1
    sites = bits.flatten(-2)[..., : L * L]
    return (sites.to(torch.int8) * 2 - 1).unflatten(-1, (L, L))


def canonical(spins: Tensor, dims: int) -> Tensor:
    """Flip each group of the last `dims` dimensions (2: a frame, 3: a burst of frames) so its
    total magnetisation is positive; a group summing to 0 goes by its first spin, which keeps
    canonical(-s) == canonical(s) exact."""
    flat = spins.flatten(-dims)
    total = flat.sum(-1, dtype=torch.int64)
    sign = torch.where(total != 0, total.sign(), flat[..., 0].to(torch.int64))
    return spins * sign.view(*sign.shape, *([1] * dims)).to(spins.dtype)


def augment(bursts: Tensor, gen: torch.Generator, d4: bool = True, roll: bool = True) -> Tensor:
    """One random symmetry of the torus per burst, the same for each of its frames, so the
    dynamics inside the burst stays consistent: (B, F, L, L) -> (B, F, L, L). A roll by (dy, dx),
    then one of D4's eight elements (a transpose or not, then 0-3 quarter turns)."""
    if not (d4 or roll):
        return bursts
    B, frames, L, _ = bursts.shape
    dev = bursts.device
    a = torch.arange(L, device=dev).view(1, L, 1).expand(B, L, L)
    b = torch.arange(L, device=dev).view(1, 1, L).expand(B, L, L)
    if roll:
        shift = torch.randint(0, L, (B, 2, 1, 1), generator=gen, device=dev)
        a, b = (a + shift[:, 0]) % L, (b + shift[:, 1]) % L
    if d4:
        g = torch.randint(0, 8, (B, 1, 1), generator=gen, device=dev)
        swap = g >= 4
        a, b = torch.where(swap, b, a), torch.where(swap, a, b)
        for quarter in (1, 2, 3):
            turn = g % 4 >= quarter
            a, b = torch.where(turn, b, a), torch.where(turn, L - 1 - a, b)
    source = (a * L + b).flatten(1).unsqueeze(1).expand(B, frames, L * L)
    return bursts.flatten(2).gather(2, source).view(B, frames, L, L)


def pairs(bursts: Tensor, max_delta: int, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """A training pair per burst: frame c and frame c + delta, delta uniform in 0..max_delta and
    c uniform over the frames that leave room for it. Returns (context frames, target frames,
    delta), frames (B, L, L)."""
    B, frames = bursts.shape[:2]
    if not 0 <= max_delta < frames:
        raise ValueError(f"max_delta must lie in 0..{frames - 1} for bursts of {frames} frames")
    dev = bursts.device
    delta = torch.randint(0, max_delta + 1, (B,), generator=gen, device=dev)
    start = (torch.rand(B, generator=gen, device=dev) * (frames - delta)).long()
    rows = torch.arange(B, device=dev)
    return bursts[rows, start], bursts[rows, start + delta], delta


# ---------------------------------------------------------------- masks


@dataclass(frozen=True)
class MaskSpec:
    block: tuple[int, int] = (4, 8)  # target block side in sites, inclusive; multiples of PATCH
    coverage: tuple[float, float] = (0.20, 0.25)  # share of cells hidden, drawn per sample
    buffer: int = 1  # cells around the hidden ones that the context does not see either


def masks(
    B: int, n: int, spec: MaskSpec, gen: torch.Generator, device: torch.device | str
) -> tuple[Tensor, Tensor]:
    """(hidden, keep), each (B, n, n) bool: the target cells, and the context cells (neither
    hidden nor within `buffer` cells of a hidden one). Per sample, blocks of random size and
    position, wrapping around the torus, are added until they cover a share of the cells drawn
    uniformly from `coverage`."""
    lo, hi = (side // PATCH for side in spec.block)
    if lo < 1 or lo > hi or any(side % PATCH for side in spec.block):
        raise ValueError(f"block sides {spec.block} must be multiples of {PATCH}, smallest first")
    if hi + 2 * spec.buffer >= n:
        raise ValueError(
            f"{n} cells a side cannot hold a {hi}-cell block and its buffer with context left "
            f"(L >= {PATCH * (hi + 2 * spec.buffer + 1)})"
        )
    blocks = 2 * math.ceil(spec.coverage[1] * n * n / (lo * lo))  # enough to reach any share
    shape = (B, blocks, 1, 1)
    h = torch.randint(lo, hi + 1, shape, generator=gen, device=device)
    w = torch.randint(lo, hi + 1, shape, generator=gen, device=device)
    y0 = torch.randint(0, n, shape, generator=gen, device=device)
    x0 = torch.randint(0, n, shape, generator=gen, device=device)
    share = torch.rand((B, 1), generator=gen, device=device)
    share = spec.coverage[0] + share * (spec.coverage[1] - spec.coverage[0])
    i = torch.arange(n, device=device).view(1, 1, n, 1)
    j = torch.arange(n, device=device).view(1, 1, 1, n)
    inside = (((i - y0) % n) < h) & (((j - x0) % n) < w)
    union = inside.to(torch.int32).cumsum(1) > 0  # (B, blocks, n, n): the first k blocks
    reached = union.flatten(2).sum(-1) >= share * (n * n)
    first = torch.where(reached.any(1), reached.to(torch.int32).argmax(1), blocks - 1)
    hidden = union[torch.arange(B, device=device), first]
    grown = hidden.clone()
    for dy in range(-spec.buffer, spec.buffer + 1):
        for dx in range(-spec.buffer, spec.buffer + 1):
            grown |= hidden.roll((dy, dx), (-2, -1))
    return hidden, ~grown


def fixed_masks(n: int, spec: MaskSpec, seed: int, count: int) -> tuple[Tensor, Tensor]:
    """`count` masks drawn once from `seed` on the CPU: the evaluation masks of readouts, the same
    for every frame of a size and for every checkpoint of a run."""
    return masks(count, n, spec, torch.Generator().manual_seed(seed), "cpu")


# ---------------------------------------------------------------- networks


class ChannelNorm(nn.LayerNorm):
    """LayerNorm over the channels of an NCHW tensor."""

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def channel_norm(x: Tensor) -> Tensor:
    """LayerNorm over the channels of an NCHW tensor, without affine parameters."""
    return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


class Block(nn.Module):
    """ConvNeXt block: depthwise kernel x kernel circular convolution, LayerNorm, MLP dim -> 4 dim
    -> dim with GELU, residual. No BatchNorm: it would leak statistics across samples and through
    the EMA."""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError(f"kernel must be odd, got {kernel}")
        self.dw = nn.Conv2d(
            dim, dim, kernel, padding=kernel // 2, padding_mode="circular", groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.down(F.gelu(self.up(self.norm(y))))
        return x + y.permute(0, 3, 1, 2)


class Encoder(nn.Module):
    """The per-frame encoder, one function at every L: a PATCH x PATCH patchify, `depth` ConvNeXt
    blocks, LayerNorm. With dim 128, depth 3 and kernel 3 it has 0.40M parameters and a
    receptive field of 14 sites, below the smallest L, so it is fully convolutional and local.

    Cells outside `keep` are zero at the input and re-zeroed after every layer (the dense form of
    a sparse encoder), so nothing about a hidden cell's spins reaches any output."""

    def __init__(self, dim: int = 128, depth: int = 3, kernel: int = 3) -> None:
        super().__init__()
        self.dim, self.depth, self.kernel = dim, depth, kernel
        self.patchify = nn.Conv2d(1, dim, PATCH, stride=PATCH)
        self.stem = ChannelNorm(dim)
        self.blocks = nn.ModuleList([Block(dim, kernel) for _ in range(depth)])
        self.norm = ChannelNorm(dim)

    @property
    def receptive_field(self) -> int:
        """Sites a side that can influence one cell's output."""
        return PATCH * (1 + self.depth * (self.kernel - 1))

    def forward(self, spins: Tensor, keep: Tensor | None = None) -> Tensor:
        """(B, L, L) spins -> (B, dim, n, n); with `keep` (B, n, n) bool, the context path."""
        x = spins.unsqueeze(1).to(torch.float32)
        if keep is None:
            x = self.stem(self.patchify(x))
            for block in self.blocks:
                x = block(x)
            return self.norm(x)
        cells = keep.unsqueeze(1).to(torch.float32)
        sites = cells.repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3)
        x = self.stem(self.patchify(x * sites)) * cells
        for block in self.blocks:
            x = block(x) * cells
        return self.norm(x) * cells


class Predictor(nn.Module):
    """From context features to every cell's target features at a frame delta ahead: the context
    projected to `width` channels, a learned mask token at the cells the context does not hold,
    an embedding of delta everywhere, `depth` ConvNeXt blocks with kernel x kernel circular
    convolutions (a range of ~50 sites with the defaults), then LayerNorm and a projection back."""

    def __init__(
        self, dim: int = 128, width: int = 64, depth: int = 4, kernel: int = 7, deltas: int = 8
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, width)
        self.mask_token = nn.Parameter(torch.zeros(width))
        self.delta = nn.Embedding(deltas, width)
        self.blocks = nn.ModuleList([Block(width, kernel) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, dim)

    def forward(self, context: Tensor, keep: Tensor, delta: Tensor) -> Tensor:
        """context (B, dim, n, n), keep (B, n, n) bool, delta (B,) -> (B, dim, n, n)."""
        x = self.proj(context.permute(0, 2, 3, 1))
        x = torch.where(keep.unsqueeze(-1), x, self.mask_token.to(x.dtype))
        x = (x + self.delta(delta).to(x.dtype)[:, None, None, :]).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)


@dataclass(frozen=True)
class ModelSpec:
    dim: int = 128
    depth: int = 3
    kernel: int = 3
    pred_width: int = 64
    pred_depth: int = 4
    pred_kernel: int = 7
    max_delta: int = 7


class Jepa(nn.Module):
    """The context encoder and predictor (trained), and the target encoder: an EMA of the
    context encoder that no gradient reaches."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.context = Encoder(spec.dim, spec.depth, spec.kernel)
        self.target = Encoder(spec.dim, spec.depth, spec.kernel)
        self.predictor = Predictor(
            spec.dim, spec.pred_width, spec.pred_depth, spec.pred_kernel, spec.max_delta + 1
        )
        self.target.requires_grad_(False)

    def initialise(self, gen: torch.Generator) -> None:
        """PyTorch's default initialisation, drawn from `gen` alone so that it never depends on
        the global RNG; the target encoder starts as a copy of the context encoder."""
        with torch.no_grad():
            for module in (*self.context.modules(), *self.predictor.modules()):
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=gen)
                    if module.bias is not None:
                        bound = 1 / math.sqrt(module.weight[0].numel())
                        nn.init.uniform_(module.bias, -bound, bound, generator=gen)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, generator=gen)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            nn.init.zeros_(self.predictor.mask_token)
        self.target.load_state_dict(self.context.state_dict())

    def trainable(self) -> list[nn.Parameter]:
        return [*self.context.parameters(), *self.predictor.parameters()]

    def targets(self, spins: Tensor) -> Tensor:
        """The regression targets: target-encoder features of whole frames, LayerNorm'd per cell
        without affine parameters (I-JEPA)."""
        with torch.no_grad():
            return channel_norm(self.target(spins))

    def predict(self, spins: Tensor, keep: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
        """(context features, predicted target features) for frames seen through `keep`."""
        context = self.context(spins, keep)
        return context, self.predictor(context, keep, delta)

    def embed(self, spins: Tensor) -> Tensor:
        """The frame embedding readouts use: target-encoder features averaged over cells (a mean,
        not AdaptiveAvgPool2d, whose CUDA backward is nondeterministic), (B, dim)."""
        return self.target(spins).mean((-2, -1))


def parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
