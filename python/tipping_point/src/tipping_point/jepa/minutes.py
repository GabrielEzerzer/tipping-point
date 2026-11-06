"""The market side of the JEPA (#30, decisions 3 and 4): one UTC minute of 1-second book features
in, one latent per minute out, and a causal predictor over past minutes' latents.

This is the Ising design moved to time. The Ising encoder sees one frame and the predictor carries
the dynamics across frames; here the encoder sees one minute (60 seconds x F features) on its own,
so neighbouring latents share no input (their AC1 is not inflated by construction), and a causal
transformer over a chunk of `context` minutes predicts the target encoder's latent of minutes
t + 1 ... t + horizons from z_1 ... z_t.

  encoder    Conv1d patch embedding (`patch` seconds a token), `depth` 1-D ConvNeXt blocks (zero
             padding: a minute is not periodic), mean over tokens, LayerNorm -> z_t (dim,)
  target     the encoder's EMA copy, no gradient, LayerNorm'd without affine parameters (I-JEPA)
  predictor  z_t projected, a mask token at invalid minutes, a learned position, `pred_depth`
             pre-norm causal attention blocks, LayerNorm, one linear head per delta = 1..horizons
  loss       objective.prediction_loss and objective.vicreg unchanged: a chunk's predictions pass
             as (B, dim, context, horizons) with the valid targets as the hidden cells, its frame
             latents as (B, dim, context, 1) with the valid minutes as the kept cells

Readouts: at minute t the predictor sees exactly the previous context - 1 minutes and z_t is
predicted from them, so surprise has no position effect. The latent is the target encoder's z_t,
as Ising's embedding is the target features averaged over cells.

Sizes (decision 4): S (dim 64, two predictor blocks, 0.24M parameters) and M (dim 96, three
blocks, 0.63M); the CPU pilot (M4) picks one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import monitor, objective
from .days import CONFIRMATORY, MINUTE, DataError, Dataset, Minutes, Span
from .monitor import Floors
from .readout import Pca, basis


@dataclass(frozen=True)
class MinuteSpec:
    features: int = 16
    dim: int = 96
    depth: int = 3
    kernel: int = 5
    patch: int = 4  # seconds a token: 15 tokens a minute
    pred_depth: int = 3
    heads: int = 4
    context: int = 96  # minutes a predictor chunk holds
    horizons: int = 5  # delta = 1 .. horizons minutes ahead


PRESETS = {"S": MinuteSpec(dim=64, pred_depth=2), "M": MinuteSpec()}


# ---------------------------------------------------------------- networks


class Block(nn.Module):
    """1-D ConvNeXt block: depthwise convolution over seconds (zero padding), LayerNorm, MLP dim ->
    4 dim -> dim with GELU, residual. No BatchNorm (lattice.Block)."""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError(f"kernel must be odd, got {kernel}")
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.dw(x).transpose(1, 2)
        y = self.down(F.gelu(self.up(self.norm(y))))
        return x + y.transpose(1, 2)


class Encoder(nn.Module):
    """One minute (N, MINUTE, features) -> z (N, dim)."""

    def __init__(self, spec: MinuteSpec) -> None:
        super().__init__()
        if MINUTE % spec.patch:
            raise ValueError(f"patch {spec.patch} must divide {MINUTE} seconds")
        self.patchify = nn.Conv1d(spec.features, spec.dim, spec.patch, stride=spec.patch)
        self.stem = nn.LayerNorm(spec.dim)
        self.blocks = nn.ModuleList([Block(spec.dim, spec.kernel) for _ in range(spec.depth)])
        self.norm = nn.LayerNorm(spec.dim)

    def forward(self, frames: Tensor) -> Tensor:
        x = self.patchify(frames.transpose(1, 2).to(torch.float32))
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.norm(x.mean(-1))


class CausalBlock(nn.Module):
    """Pre-norm transformer block whose position t attends to positions <= t only."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"{heads} heads must divide width {dim}")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, dim = x.shape
        qkv = self.qkv(self.norm1(x)).view(B, C, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(y.transpose(1, 2).reshape(B, C, dim))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class Predictor(nn.Module):
    """Frame latents z (B, C, dim) of a chunk and its valid minutes (B, C) -> predictions
    (B, C, horizons, dim): [b, t, k] predicts the target latent of minute t + k + 1 from minutes
    <= t."""

    def __init__(self, spec: MinuteSpec) -> None:
        super().__init__()
        dim = spec.dim
        self.proj = nn.Linear(dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.position = nn.Embedding(spec.context, dim)
        self.blocks = nn.ModuleList([CausalBlock(dim, spec.heads) for _ in range(spec.pred_depth)])
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleList([nn.Linear(dim, dim) for _ in range(spec.horizons)])

    def forward(self, z: Tensor, valid: Tensor) -> Tensor:
        x = self.proj(z)
        x = torch.where(valid.unsqueeze(-1), x, self.mask_token.to(x.dtype))
        x = x + self.position.weight[: x.shape[1]].to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return torch.stack([head(x) for head in self.heads], 2)


class MarketJepa(nn.Module):
    """The context encoder and predictor (trained), and the target encoder: an EMA of the context
    encoder that no gradient reaches. The same interface as lattice.Jepa where the loop needs it."""

    def __init__(self, spec: MinuteSpec) -> None:
        super().__init__()
        self.spec = spec
        self.context = Encoder(spec)
        self.target = Encoder(spec)
        self.predictor = Predictor(spec)
        self.target.requires_grad_(False)

    def initialise(self, gen: torch.Generator) -> None:
        """PyTorch's default initialisation drawn from `gen` alone (lattice.Jepa.initialise), the
        positions at std 0.02 as in ViTs; the target starts as a copy of the context encoder."""
        with torch.no_grad():
            for module in (*self.context.modules(), *self.predictor.modules()):
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=gen)
                    if module.bias is not None:
                        bound = 1 / math.sqrt(module.weight[0].numel())
                        nn.init.uniform_(module.bias, -bound, bound, generator=gen)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02, generator=gen)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            nn.init.zeros_(self.predictor.mask_token)
        self.target.load_state_dict(self.context.state_dict())

    def trainable(self) -> list[nn.Parameter]:
        return [*self.context.parameters(), *self.predictor.parameters()]

    def targets(self, frames: Tensor) -> Tensor:
        """Target-encoder latents of minutes (N, MINUTE, F), LayerNorm'd without affine parameters:
        (N, dim)."""
        with torch.no_grad():
            z = self.target(frames)
            return F.layer_norm(z, (z.shape[-1],))

    def chunk(self, frames: Tensor, valid: Tensor) -> dict[str, Tensor]:
        """The loss inputs of chunks (B, C, MINUTE, F) with valid minutes (B, C), in the shapes
        objective.py takes: pred and target (B, dim, C, horizons), hidden (B, C, horizons) (the
        target minute exists in the chunk and is valid), features (B, dim, C, 1) (the context
        latents), keep (B, C, 1) (the valid minutes)."""
        B, C = valid.shape
        flat = frames.flatten(0, 1)
        targets = self.targets(flat).view(B, C, -1)
        z = self.context(flat).view(B, C, -1)
        pred = self.predictor(z, valid)
        H = pred.shape[2]
        padded = F.pad(targets, (0, 0, 0, H))
        future = torch.stack([padded[:, k : k + C] for k in range(1, H + 1)], 2)
        seen = F.pad(valid, (0, H))
        hidden = torch.stack([seen[:, k : k + C] for k in range(1, H + 1)], 2)
        return {
            "pred": pred.permute(0, 3, 1, 2),
            "target": future.permute(0, 3, 1, 2),
            "hidden": hidden,
            "features": z.permute(0, 2, 1).unsqueeze(-1),
            "keep": valid.unsqueeze(-1),
        }


def parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------- readouts


@torch.no_grad()
def readouts(
    model: MarketJepa, frames: Tensor, valid: Tensor, windows: int = 1024
) -> tuple[Tensor, Tensor]:
    """For a contiguous series of minutes (M, MINUTE, F) with valid (M,): the latent z_t (M, dim)
    and the surprise (M, horizons), [t, k] the error of the delta = k + 1 prediction of z_t from
    the minutes before it in the window [t - context + 1, t]. Surprise is NaN where minute t is
    invalid or fewer than context - 1 minutes precede it. Float32 on the CPU, autocast off."""
    device = next(model.parameters()).device
    C, H = model.spec.context, model.spec.horizons
    M = len(frames)
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        z, latent, target = [], [], []
        for start in range(0, M, windows):
            x = frames[start : start + windows].to(device)
            latent.append(model.target(x))
            target.append(F.layer_norm(latent[-1], (latent[-1].shape[-1],)))
            z.append(model.context(x))
        latent_t, target_t, z_t = torch.cat(latent), torch.cat(target), torch.cat(z)
        surprise = torch.full((M, H), float("nan"), device=device)
        ends = torch.arange(C - 1, M, device=device)
        ok = valid.to(device)
        for first in range(0, len(ends), windows):
            end = ends[first : first + windows]
            index = end.unsqueeze(1) + torch.arange(-C + 1, 1, device=device)
            pred = model.predictor(z_t[index], ok[index])  # (W, C, H, dim)
            for k in range(H):
                guess = pred[:, C - 2 - k, k].unsqueeze(-1).unsqueeze(-1)
                truth = target_t[end].unsqueeze(-1).unsqueeze(-1)
                cells = torch.ones(len(end), 1, 1, dtype=torch.bool, device=device)
                surprise[end, k] = objective.errors(guess, truth, cells)
        surprise[~ok] = float("nan")
    return latent_t.cpu(), surprise.cpu()


def fit_pca(
    model: MarketJepa, data: Minutes, count: int, k: int, gen: torch.Generator, device: torch.device
) -> Pca:
    """A PCA basis of the latent fitted on `count` valid training minutes drawn by `gen` (CPU)."""
    candidates = torch.nonzero(data.valid).squeeze(1)
    which = candidates[torch.randint(len(candidates), (count,), generator=gen)]
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        rows = [
            model.target(data.frames[chunk].to(device)).to("cpu", torch.float64)
            for chunk in which.split(1024)
        ]
    return basis(torch.cat(rows), k)


# ---------------------------------------------------------------- training


@dataclass(frozen=True)
class Config:
    """Everything that defines a market training besides its data. Each step encodes `chunks` x
    model.context minute-frames (768 at the defaults, the research pass's batch). The schedule
    defaults are the Ising ones until the CPU pilot (M4) sets the step count by its rule."""

    symbol: str = "BTCUSDT"
    train_start: str = "2023-01-18"
    train_end: str = "2023-10-01"  # exclusive: the exploratory read starts after a 1-day embargo
    steps: int = 10_000
    model: MinuteSpec = field(default_factory=lambda: PRESETS["M"])
    chunks: int = 8
    lr: float = 1e-3
    lr_final: float = 1e-6
    warmup: int = 500
    weight_decay: tuple[float, float] = (0.04, 0.4)
    ema: tuple[float, float] = (0.996, 1.0)
    var_weight: float = 0.025
    cov_weight: float = 0.001
    var_gamma: float = 1.0
    seed: int = 1
    checkpoint_every: int = 1_000
    monitor_every: int = 250
    monitor_chunks: int = 16
    pca_frames: int = 4_096
    pca_components: int = 8
    floors: Floors = field(default_factory=Floors)
    patience: int = 3
    stop_on_collapse: bool = True

    @property
    def span(self) -> Span:
        return Span(
            self.symbol, date.fromisoformat(self.train_start), date.fromisoformat(self.train_end)
        )

    def to_json(self) -> dict[str, Any]:
        return {"domain": "market", **asdict(self)}

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> Config:
        r = {k: v for k, v in record.items() if k != "domain"}
        r["model"] = MinuteSpec(**r["model"])
        r["floors"] = Floors(**r["floors"])
        r["weight_decay"], r["ema"] = tuple(r["weight_decay"]), tuple(r["ema"])
        return cls(**r)


def preset(name: str, **changes: Any) -> Config:
    """The default config with size preset S or M."""
    return replace(Config(model=PRESETS[name]), **changes)


class Domain:
    """The market half of a training (train.Domain): chunks of consecutive minutes from one
    symbol's training span, one group, tables keyed by step alone."""

    name = "jepa-market"
    keys: tuple[tuple[str, Any], ...] = ()

    def __init__(
        self, config: Config, data: Dataset, device: torch.device, seeds: dict[str, int]
    ) -> None:
        self.config, self.data, self.device, self.seeds = config, data, device, seeds
        span, sources = config.span, data.sources
        if span.end > CONFIRMATORY or span.start >= span.end:
            raise DataError(f"training span [{span.start}, {span.end}) must end by {CONFIRMATORY}")
        if sources.get("kind") != "market" or sources.get("symbol") != config.symbol:
            raise DataError(f"a {config.symbol} model reads {config.symbol} days only")
        if sources["span"] != [span.start.isoformat(), span.end.isoformat()]:
            raise DataError(f"days were loaded for span {sources['span']}, not this model's")
        if len(sources["features"]) != config.model.features:
            raise DataError(
                f"{len(sources['features'])} features, the model takes {config.model.features}"
            )
        model = MarketJepa(config.model)
        model.initialise(torch.Generator().manual_seed(seeds["init"]))
        self.model = model.to(device)

    def groups(self) -> list[None]:
        return [None]

    def key(self, group: None) -> dict[str, Any]:
        return {}

    def monitor_batches(self, gen: torch.Generator) -> dict[None, dict[str, Tensor]]:
        """A fixed validation batch of wholly valid chunks, drawn once."""
        C = self.config.model.context
        starts = self.data.val.starts(C, whole=True)
        if len(starts) == 0:
            raise DataError(f"no wholly valid run of {C} validation minutes for the monitor")
        pick = starts[torch.randint(len(starts), (self.config.monitor_chunks,), generator=gen)]
        frames, valid = self.data.val.chunks(pick, C)
        return {None: {"frames": frames.to(self.device), "valid": valid.to(self.device)}}

    def sample(
        self, group: None, data_gen: torch.Generator, aug_gen: torch.Generator
    ) -> dict[str, Tensor]:
        frames, valid = self.data.train.sample(
            self.config.chunks, self.config.model.context, data_gen
        )
        return {"frames": frames.to(self.device), "valid": valid.to(self.device)}

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model.chunk(batch["frames"], batch["valid"])

    def evaluate(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        out = self.model.chunk(batch["frames"], batch["valid"])
        error = float(objective.prediction_loss(out["pred"], out["target"], out["hidden"]))
        return {"val_error": error, **monitor.spread(out["features"])}

    def pca(self, gen: torch.Generator) -> Pca:
        cfg = self.config
        return fit_pca(
            self.model, self.data.train, cfg.pca_frames, cfg.pca_components, gen, self.device
        )

    def params(self) -> dict[str, Any]:
        return {
            "frames_per_step": self.config.chunks * self.config.model.context,
            "parameters": {
                "encoder": parameters(self.model.context),
                "predictor": parameters(self.model.predictor),
            },
        }

    def check_resume(self, manifest: dict[str, Any]) -> None:
        if manifest["data"] != self.data.sources:
            raise DataError("the days handed to resume differ from the ones this training read")
