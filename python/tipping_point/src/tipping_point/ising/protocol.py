"""Temperature protocols as pure functions of the integer sweep index.

Step t counts completed sweeps: sweep t (t >= 1) runs at the bath temperature of step t and
produces the configuration recorded at step t; step 0 is the initial state. Nothing is
accumulated, so any step can be asked about on its own, and a resumed run sees exactly the
schedule a straight run sees.

Every schedule offers the run loop the same surface (see `Schedule`):
  total             last step of the run
  blocks            temperature blocks; the batch holds `replicas` lattices per block
  temperatures(t)   bath temperature of each block at step t
  segment(t)        plateau index (staircase), phase (ramp) or 0 (fixed)
  phase(t)          0 thermalise, 1 measure (ramp: 0 pre-equilibrate, 1 ramp, 2 hold)
  records_stats(t)  whether step t writes a stats row
  frames            {step: (burst, frame)} for every saved configuration
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from .exact import TC

T_HOT, T_COLD = 3.5, 1.5
GRID = tuple(float(t) for t in np.linspace(T_COLD, T_HOT, 40)[::-1])  # staircase order: hot first


@dataclass(frozen=True)
class Burst:
    """A save point: `frames` consecutive configurations, `stride` sweeps apart."""

    frames: int = 8
    stride: int = 1

    @property
    def span(self) -> int:
        """Sweeps a burst occupies before the next one may start."""
        return self.frames * self.stride


BURST = Burst()
BAND = (10.0, -5.0)  # the EWS analysis band in epsilon_L = L (T - Tc) / Tc, hot edge first (#26)
BAND_PER_EPSILON_L = 28.0  # ramp bursts per unit epsilon_L inside it: what dT = 0.005 gives L = 16


class Schedule(Protocol):
    kind: str
    total: int
    blocks: int
    frames: dict[int, tuple[int, int]]

    def temperatures(self, step: int) -> tuple[float, ...]: ...
    def segment(self, step: int) -> int: ...
    def phase(self, step: int) -> int: ...
    def records_stats(self, step: int) -> bool: ...
    def to_json(self) -> dict[str, Any]: ...


def dwell(tau: float, min_therm: int = 500, min_meas: int = 2000) -> tuple[int, int]:
    """Sweeps to thermalise (20 tau) and then measure (100 tau) at one temperature."""
    return max(min_therm, math.ceil(20 * tau)), max(min_meas, math.ceil(100 * tau))


def rate_for(L: int, x: float, A: float, z: float) -> int:
    """Sweeps per unit T that put size L at the Kibble-Zurek point x of the true tau_c = A L^z
    (the pilot record's fit; nu = 1):

        r = A L^(z + 1) / (x Tc),   that is   x = tau_c(L) / (r Tc / L),

    tau_c over the sweeps the bath spends per unit of epsilon_L = L (T - Tc) / Tc, the same at
    every L. The critical window |epsilon_L| < 1 takes 2/x tau_c to cross: the slow ramp, x = 0.02,
    spends 50 tau_c per unit epsilon_L and ~100 tau_c in the window; the fast one, x = 2, crosses
    it in ~1 tau_c, the Kibble-Zurek regime #19 asked for. Before #50 the rate was
    L^3.17 / (x Tc), with no A: its nominal x = 0.3 and 30 are 0.3 A and 30 A here, 0.016 and 1.6
    on the pre-#47 kernel (A = 0.052).
    """
    return max(1, round(A * L ** (z + 1) / (x * TC)))


def _bursts(first: int, n: int, saves: int, burst: Burst) -> list[int]:
    """First steps of up to `saves` evenly spaced bursts within n steps from `first`; no overlap."""
    count = min(saves, n // burst.span)
    spacing = n // count if count else 0
    return [first + b * spacing for b in range(count)]


def _frames(starts: Iterable[int], burst: Burst) -> dict[int, tuple[int, int]]:
    """{step: (burst, frame)}: frame f of burst b is saved at step starts[b] + f * stride."""
    return {s + f * burst.stride: (b, f) for b, s in enumerate(starts) for f in range(burst.frames)}


class Staircase:
    """Quasi-static sweep: hold each grid temperature in turn, hot to cold, carrying the state."""

    kind = "staircase"
    blocks = 1

    def __init__(
        self,
        taus: Sequence[float],
        temperatures: Sequence[float] = GRID,
        warmup: int = 1000,
        min_therm: int = 500,
        min_meas: int = 2000,
        saves: int = 200,
        burst: Burst = BURST,
    ) -> None:
        if len(taus) != len(temperatures):
            raise ValueError(f"{len(taus)} taus for {len(temperatures)} temperatures")
        self.params = {
            "taus": list(taus),
            "temperatures": list(temperatures),
            "warmup": warmup,
            "min_therm": min_therm,
            "min_meas": min_meas,
            "saves": saves,
            "burst": asdict(burst),
        }
        self.temps = tuple(temperatures)
        dwells = [dwell(tau, min_therm, min_meas) for tau in taus]
        self.therm = [th for th, _ in dwells]
        self.therm[0] += warmup  # the random start thermalises at the first (hottest) plateau
        self.meas = [me for _, me in dwells]
        self.strides = [max(1, int(tau // 50)) for tau in taus]  # >= 50 stats rows per tau
        self.starts: list[int] = []
        self.meas_starts: list[int] = []
        step = 1
        for th, me in zip(self.therm, self.meas, strict=True):
            self.starts.append(step)
            self.meas_starts.append(step + th)
            step += th + me
        self.total = step - 1
        starts = [
            b
            for m, n in zip(self.meas_starts, self.meas, strict=True)
            for b in _bursts(m, n, saves, burst)
        ]
        self.frames = _frames(starts, burst)

    def segment(self, step: int) -> int:
        return bisect.bisect_right(self.starts, step) - 1

    def temperatures(self, step: int) -> tuple[float, ...]:
        return (self.temps[self.segment(step)],)

    def phase(self, step: int) -> int:
        return int(step >= self.meas_starts[self.segment(step)])

    def records_stats(self, step: int) -> bool:
        j = self.segment(step)
        return (step - self.starts[j]) % self.strides[j] == 0 or step == self.total

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "params": self.params,
            "total": self.total,
            "therm": self.therm,
            "meas": self.meas,
            "stats_strides": self.strides,
        }


class Ramp:
    """Linear cooling T0 -> T1 at `rate` sweeps per unit T, between an equilibration at T0 and a
    hold at T1. The bath crosses Tc at the real-valued step t_c = t0 + rate * (T0 - Tc).

    Bursts are saved every save_dT of cooling, except inside the analysis band of epsilon_L =
    L (T - Tc) / Tc, where they come every 1/per_epsilon_L of epsilon_L: evenly spaced in T, a
    fixed dT gives L = 128 an eighth of L = 16's bursts per unit epsilon_L. A ramp too fast for
    either spacing saves its bursts back to back."""

    kind = "ramp"
    blocks = 1

    def __init__(
        self,
        rate: int,
        L: int,
        T0: float = T_HOT,
        T1: float = T_COLD,
        pre: int = 1000,
        hold: int = 1000,
        save_dT: float = 0.005,
        band: Sequence[float] = BAND,
        per_epsilon_L: float = BAND_PER_EPSILON_L,
        burst: Burst = BURST,
        x: float | None = None,  # the Kibble-Zurek point the rate came from, kept for the record
        A: float | None = None,  # ... and the tau_c = A L^z fit it was computed against
        z: float | None = None,
    ) -> None:
        self.params = {
            "rate": rate,
            "L": L,
            "T0": T0,
            "T1": T1,
            "pre": pre,
            "hold": hold,
            "save_dT": save_dT,
            "band": list(band),
            "per_epsilon_L": per_epsilon_L,
            "burst": asdict(burst),
            "x": x,
            "A": A,
            "z": z,
        }
        self.rate, self.T0, self.T1 = rate, T0, T1
        self.t0 = pre  # last step at T0
        self.t1 = pre + round((T0 - T1) * rate)  # the ramp is sweeps t0+1 .. t1, ending at T1
        self.total = self.t1 + hold
        self.t_c = self.t0 + rate * (T0 - TC)
        self.stride = max(1, self.total // 100_000)  # ~1e5 stats rows per replica at most
        hot, cold = (TC * (1 + e / L) for e in band)
        # the band's steps: the first at or below its hot edge, the last at or above its cold one
        first = max(self.t0 + 1, self.t0 + math.ceil((T0 - hot) * rate))
        last = min(self.t1, self.t0 + math.floor((T0 - cold) * rate))
        outside = max(burst.span, round(save_dT * rate))
        inside = max(burst.span, round(TC / (per_epsilon_L * L) * rate))
        starts = []
        step = self.t0 + 1
        while step <= self.t1 - burst.span + 1:
            starts.append(step)
            gap = inside if first <= step <= last else outside
            if step < first < step + gap:  # enter the band at its edge rather than past it
                gap = max(burst.span, first - step)
            step += gap
        self.frames = _frames(starts, burst)
        self.saves = {
            "gap": outside,
            "band_steps": [first, last],
            "band_gap": inside,
            "bursts": len(starts),
            "bursts_in_band": sum(first <= s <= last for s in starts),
        }

    def temperatures(self, step: int) -> tuple[float, ...]:
        if step <= self.t0:
            return (self.T0,)
        if step >= self.t1:
            return (self.T1,)
        return (self.T0 - (step - self.t0) / self.rate,)

    def phase(self, step: int) -> int:
        return 0 if step <= self.t0 else 1 if step <= self.t1 else 2

    def segment(self, step: int) -> int:
        return self.phase(step)

    def records_stats(self, step: int) -> bool:
        return step % self.stride == 0 or step == self.total

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "params": self.params,
            "total": self.total,
            "t0": self.t0,
            "t1": self.t1,
            "t_c": self.t_c,
            "stats_stride": self.stride,
            "saves": self.saves,
        }


class Fixed:
    """Constant temperatures, one per block of replicas: pilot and validation runs."""

    kind = "fixed"

    def __init__(
        self,
        temperatures: Sequence[float],
        therm: int,
        sweeps: int,
        stats_stride: int = 1,
        saves: int = 0,
        burst: Burst = BURST,
    ) -> None:
        self.params = {
            "temperatures": list(temperatures),
            "therm": therm,
            "sweeps": sweeps,
            "stats_stride": stats_stride,
            "saves": saves,
            "burst": asdict(burst),
        }
        self.temps = tuple(temperatures)
        self.blocks = len(self.temps)
        self.therm, self.stats_stride = therm, stats_stride
        self.total = therm + sweeps
        self.frames = _frames(_bursts(therm + 1, sweeps, saves, burst), burst)

    def temperatures(self, step: int) -> tuple[float, ...]:
        return self.temps

    def segment(self, step: int) -> int:
        return 0

    def phase(self, step: int) -> int:
        return int(step > self.therm)

    def records_stats(self, step: int) -> bool:
        return step % self.stats_stride == 0 or step == self.total

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": self.params, "total": self.total}


def from_json(record: dict[str, Any]) -> Schedule:
    """Rebuild a schedule from its manifest record (resume relies on this)."""
    params = dict(record["params"])
    params["burst"] = Burst(**params["burst"])
    kinds: dict[str, Any] = {"staircase": Staircase, "ramp": Ramp, "fixed": Fixed}
    return kinds[record["kind"]](**params)
