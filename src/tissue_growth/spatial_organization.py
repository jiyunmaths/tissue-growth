"""Candidate spatial niche model: local activation and mobile substrate depletion.

Patterns are dynamical states of four local fields; no target map is stored.
The legacy homogeneous model remains available as a control.
"""
from dataclasses import dataclass, fields
import json
from pathlib import Path
import numpy as np

from .organization import Aim1Protocol, OrganizationModel, OrganizationParameters
from .operators import ScipyOperators


@dataclass(frozen=True)
class SpatialParameters(OrganizationParameters):
    renewing_diffusion: float = 0.0002
    differentiated_diffusion: float = 0.003
    niche_diffusion: float = 1/144
    substrate_diffusion: float = 20/144
    renewal_rate: float = 2.4
    differentiation_rate: float = 0.5
    differentiated_turnover: float = 0.08
    differentiated_feedback: float = 0.1
    niche_basal: float = 0.025
    substrate_supply: float = 1.24
    niche_half: float = 1.2
    activation: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.niche_half <= 0 or self.niche_basal+self.substrate_supply <= 0:
            raise ValueError("Require positive niche_half and combined signal supply")


class FixedSurfaceOperators(ScipyOperators):
    """Cache exact sparse LU factors for the fixed mesh and time step."""
    def solve(self, rhs, alpha, field):
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu
        if field not in self._cache or self._cache[field][0] != alpha:
            self._cache[field] = (alpha, splu((diags(self.mass)+alpha*self.stiffness).tocsc()))
        self.iterations = 1
        return self._cache[field][1].solve(rhs)


@dataclass(frozen=True)
class SpatialProtocol(Aim1Protocol):
    model: str = "spatial"
    dt: float = 0.04
    settle_time: float = 300
    turnover_time: float = 60
    repair_time: float = 180
    parameters: SpatialParameters = SpatialParameters()

    def __post_init__(self):
        super().__post_init__()
        if self.model != "spatial":
            raise ValueError("SpatialProtocol requires model=spatial")

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).read_text())
        parameters = SpatialParameters.from_dict(data.pop("parameters", {}))
        unknown = set(data)-{field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown spatial protocol keys: {sorted(unknown)}")
        return cls(parameters=parameters, **data)


class SpatialOrganizationModel(OrganizationModel):
    FIELD_NAMES = ("renewing", "differentiated", "niche", "substrate")
    PARAMETERS = SpatialParameters

    def __init__(self, config=None, parameters=None, seed=42, initial_condition="random"):
        super().__init__(config, parameters or SpatialParameters(), seed, initial_condition)
        self.ops.close()
        self.ops = FixedSurfaceOperators(self.config)

    @property
    def diffusivities(self):
        return super().diffusivities+(self.parameters.substrate_diffusion,)

    def initial_state(self, kind="random"):
        cells = super().initial_state("random" if kind == "near_uniform" else kind)[:2]
        p = self.parameters
        # A spatially constant reference level plus noise is an initialization,
        # never a field toward which the reaction restores the simulation.
        level = p.niche_basal+p.substrate_supply
        niche = level*(1+0.1*self._smooth_field())
        substrate = p.substrate_supply/level**2*(1+0.1*self._smooth_field())
        if kind == "near_uniform":
            occupancy = 0.6*(1+0.01*self._smooth_field())
            fraction = 0.36+0.005*self._smooth_field()
            cells = np.array([occupancy*fraction, occupancy*(1-fraction)])
            niche = level*(1+0.01*self._smooth_field())
            substrate = p.substrate_supply/level**2*(1+0.01*self._smooth_field())
        if kind == "imposed_pattern":
            coordinate = self.ops.points[:, 2] if self.config.geometry == "sphere" else self.ops.points[:, 0]
            niche = level*(1+0.4*np.cos(6*np.pi*coordinate))
        return np.vstack((cells, niche, substrate))

    def fluxes(self, state=None):
        state = self.state if state is None else state
        r, d, niche, substrate = state
        p = self.parameters
        f = super().fluxes(state[:3])
        response = niche**4/(p.niche_half**4+niche**4)
        response = 1-p.niche_coupling+p.niche_coupling*response
        vacancy = np.maximum(0, 1-p.mechanical_feedback*(r+d)/p.carrying_capacity)
        f["proliferation"] = p.renewal_rate*(0.2+0.8*response)*r*vacancy
        f["differentiation"] = p.differentiation_rate*(1-0.98*response)*r
        f["activation"] = p.activation*niche*niche*substrate
        return f

    def reaction(self, state=None):
        state = self.state if state is None else state
        r, d, niche, substrate = state
        p, f = self.parameters, self.fluxes(state)
        return np.array([
            f["proliferation"]-f["differentiation"]-f["renewing_loss"]-f["extrusion_renewing"],
            f["differentiation"]-f["differentiated_loss"]-f["extrusion_differentiated"],
            p.niche_basal+f["activation"]-niche-p.differentiated_feedback*d*niche,
            p.substrate_supply-f["activation"],
        ])

    def diagnostics(self):
        result = super().diagnostics()
        fraction = self.state[0]/np.maximum(self.occupancy, 1e-15)
        mean = self._weighted_mean(fraction)
        result.update({
            "mean_substrate": self._weighted_mean(self.state[3]),
            "std_renewing_fraction": float(np.sqrt(self._weighted_mean((fraction-mean)**2))),
            "renewing_fraction_min": float(fraction.min()),
            "renewing_fraction_max": float(fraction.max()),
            "renewing_rich_area_fraction": self._weighted_mean((fraction > 0.4).astype(float)),
            "differentiated_rich_area_fraction": self._weighted_mean((fraction < 0.2).astype(float)),
        })
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        mask = fraction > 0.4
        cells = self.ops.triangles
        a, b = cells.ravel(), np.roll(cells, -1, axis=1).ravel()
        graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(mask), len(mask))).tocsr()
        result["renewing_rich_components"] = int(connected_components(graph[mask][:, mask], directed=False, return_labels=False)) if np.any(mask) else 0
        result["pattern_wavenumber"] = float(np.sqrt(max(0, fraction @ (self.ops.stiffness @ fraction))/max(1e-15, ((fraction-mean)**2) @ self.ops.mass)))
        return result

    def perturb(self, fraction=0.6, radius=0.35, center=(0.0, 0.5), population="both", erase_signals=False):
        # Erase local signaling memory with the same lesion footprint when
        # requested; repair then includes rebuilding the dynamic niche fields.
        result = super().perturb(fraction, radius, center, population)
        if erase_signals:
            from .geometry import sphere_direction
            if self.config.geometry == "sphere":
                points = self.ops.points/np.linalg.norm(self.ops.points, axis=1)[:, None]
                distance = np.arccos(np.clip(points @ sphere_direction(*center), -1, 1))
            else:
                distance = np.linalg.norm(self.ops.points[:, :2]-np.asarray(center), axis=1)
            self.state[2:] *= 1-fraction*np.exp(-distance**2/(2*radius**2))
            result = self.diagnostics()
        return result
