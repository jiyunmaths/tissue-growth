from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    backend: str = "fenicsx"
    geometry: str = "square"
    n: int = 48
    dt: float = 0.02
    min_dt: float = 1e-7
    t_end: float = 160.0
    length: float = 3.0
    max_length: float = 12.0
    growth_rate: float = 0.015
    a: float = 0.1
    b: float = 0.9
    du: float = 1.0
    dv: float = 20.0
    reaction_scale: float = 1.0
    noise: float = 0.01
    seed: int = 42
    ksp_rtol: float = 1e-10
    pc: str = "jacobi"
    snapshot_every: float = 2.0
    diagnostic_every: float = 0.5

    def __post_init__(self):
        if self.geometry not in {"square", "sphere"}:
            raise ValueError("geometry must be square or sphere")
        if self.backend not in {"fenicsx", "scipy"}:
            raise ValueError("backend must be fenicsx or scipy; fallback is never automatic")
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if type(self.n) is not int or not 4 <= self.n <= 1024:
            raise ValueError("n must be an integer between 4 and 1024")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("dt", "min_dt", "t_end", "length", "max_length", "a", "b", "ksp_rtol", "snapshot_every", "diagnostic_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("growth_rate", "du", "dv", "reaction_scale"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.noise < 1:
            raise ValueError("noise must be in [0, 1)")
        if self.max_length < self.length or self.min_dt > self.dt:
            raise ValueError("Require max_length >= length and min_dt <= dt")
        if not 0 < self.ksp_rtol < 1 or self.pc not in {"jacobi", "gamg", "auto"}:
            raise ValueError("Require 0 < ksp_rtol < 1 and pc = jacobi, gamg, or auto")

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def read(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
