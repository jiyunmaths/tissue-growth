"""Aim 1: local-feedback model for tissue establishment, turnover, and repair.

The model is deliberately isolated from the Schnakenberg demonstration. It
contains renewing cells (R), differentiated cells (D), and a permissive niche
signal (N). Its only mechanical variable is local occupancy C=R+D. No target
map, target-pattern error, or position-dependent restoring force is supplied.
"""
from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import Config
from .geometry import sphere_direction
from .operators import ScipyOperators


@dataclass(frozen=True)
class OrganizationParameters:
    renewing_diffusion: float = 0.003
    differentiated_diffusion: float = 0.001
    niche_diffusion: float = 0.01
    renewal_rate: float = 1.4
    differentiation_rate: float = 0.30
    renewing_turnover: float = 0.03
    differentiated_turnover: float = 0.18
    niche_recovery: float = 0.60
    differentiated_feedback: float = 0.80
    niche_coupling: float = 1.0
    carrying_capacity: float = 1.0
    crowding_extrusion: float = 2.0
    mechanical_feedback: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive")
        if not 0 <= self.niche_coupling <= 1:
            raise ValueError("niche_coupling must be in [0, 1]")

    @classmethod
    def from_dict(cls, data):
        unknown = set(data)-{field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown organization parameters: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class Aim1Protocol:
    geometry: str = "sphere"
    n: int = 24
    dt: float = 0.02
    settle_time: float = 80.0
    turnover_time: float = 20.0
    repair_time: float = 60.0
    seed: int = 42
    parameters: OrganizationParameters = OrganizationParameters()

    def __post_init__(self):
        if self.geometry not in {"square", "sphere"}:
            raise ValueError("geometry must be square or sphere")
        if type(self.n) is not int or not 4 <= self.n <= 256:
            raise ValueError("n must be an integer between 4 and 256")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("dt", "settle_time", "turnover_time", "repair_time"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("model") == "spatial":
            from .spatial_organization import SpatialProtocol
            return SpatialProtocol.read(path)
        parameters = OrganizationParameters.from_dict(data.pop("parameters", {}))
        unknown = set(data)-{field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown Aim 1 protocol keys: {sorted(unknown)}")
        return cls(parameters=parameters, **data)

    def to_dict(self):
        return asdict(self)


class OrganizationModel:
    """Three-field surface model driven only by local state.

    Diffusion is implicit and reactions are explicit. Trial steps that would
    create invalid fields are halved without committing state or flux totals.
    """

    FIELD_NAMES = ("renewing", "differentiated", "niche")

    def __init__(self, config=None, parameters=None, seed=42, initial_condition="random"):
        config = config or Config(
            backend="scipy", geometry="sphere", n=24, dt=0.02,
            length=1, max_length=1, growth_rate=0,
        )
        if config.backend != "scipy":
            raise ValueError("OrganizationModel currently requires the scipy backend")
        if config.length != config.max_length or config.growth_rate != 0:
            raise ValueError("Aim 1 currently uses a fixed reference tissue")
        self.config = config
        self.parameters = parameters or OrganizationParameters()
        self.ops = ScipyOperators(config)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.steps = 0
        self.rejected = 0
        self.cumulative = {
            "proliferation": 0.0,
            "differentiation": 0.0,
            "renewing_loss": 0.0,
            "differentiated_loss": 0.0,
            "extrusion": 0.0,
        }
        self.state = self.initial_state(initial_condition)

    def _smooth_field(self):
        points = self.ops.points
        values = np.zeros(len(points))
        if self.config.geometry == "sphere":
            for _ in range(8):
                direction = self.rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                frequency = self.rng.integers(1, 5)
                phase = self.rng.uniform(0, 2*np.pi)
                values += self.rng.uniform(-1, 1)*np.cos(np.pi*frequency*(points @ direction)+phase)
        else:
            for _ in range(8):
                k = self.rng.integers(1, 5, size=2)
                values += self.rng.uniform(-1, 1)*np.cos(np.pi*k[0]*points[:, 0])*np.cos(np.pi*k[1]*points[:, 1])
        values -= np.average(values, weights=self.ops.mass)
        values /= max(float(np.max(np.abs(values))), 1e-12)
        return values

    def initial_state(self, kind="random"):
        """Create a named initial state without supplying a normal target map."""
        if kind not in {"random", "low", "high", "mosaic", "segregated", "imposed_pattern"}:
            raise ValueError(f"Unknown initial condition: {kind}")
        a, b, c = self._smooth_field(), self._smooth_field(), self._smooth_field()
        if kind == "low":
            occupancy = 0.14*(1+0.55*a)
            renewing_fraction = np.clip(0.55+0.20*b, 0.1, 0.9)
            niche = 0.90*(1+0.08*c)
        elif kind == "high":
            occupancy = 1.20*(1+0.08*a)
            renewing_fraction = np.clip(0.45+0.15*b, 0.1, 0.9)
            niche = 0.35*(1+0.15*c)
        elif kind == "mosaic":
            axis = self.ops.points[:, 2] if self.config.geometry == "sphere" else 2*self.ops.points[:, 0]-1
            occupancy = np.where(axis >= 0, 0.95, 0.22)*(1+0.08*a)
            renewing_fraction = np.where(axis >= 0, 0.20, 0.75)
            niche = np.where(axis >= 0, 0.35, 0.90)*(1+0.05*c)
        elif kind == "segregated":
            axis = self.ops.points[:, 0]
            occupancy = 0.60*(1+0.18*a)
            renewing_fraction = np.where(axis >= 0, 0.85, 0.08)
            niche = 0.65*(1+0.18*c)
        elif kind == "imposed_pattern":
            coordinate = self.ops.points[:, 2] if self.config.geometry == "sphere" else self.ops.points[:, 0]
            occupancy = 0.58*(1+0.42*np.cos(6*np.pi*coordinate))
            renewing_fraction = np.clip(0.36+0.12*b, 0.1, 0.9)
            niche = 0.65*(1+0.08*c)
        else:
            occupancy = 0.58*(1+0.35*a)
            renewing_fraction = np.clip(0.36+0.20*b, 0.1, 0.9)
            niche = 0.65*(1+0.20*c)
        state = np.array([occupancy*renewing_fraction, occupancy*(1-renewing_fraction), niche])
        if not np.all(np.isfinite(state)) or np.min(state) <= 0:
            raise RuntimeError("Initial condition generation produced an invalid state")
        return state

    @property
    def occupancy(self):
        return self.state[0]+self.state[1]

    def fluxes(self, state=None):
        p = self.parameters
        renewing, differentiated, niche = self.state if state is None else state
        occupancy = renewing+differentiated
        free_fraction = np.maximum(
            0.0, 1.0-p.mechanical_feedback*occupancy/p.carrying_capacity,
        )
        effective_niche = 1-p.niche_coupling+p.niche_coupling*niche
        proliferation = p.renewal_rate*effective_niche*renewing*free_fraction
        differentiation = p.differentiation_rate*renewing
        renewing_loss = p.renewing_turnover*renewing
        differentiated_loss = p.differentiated_turnover*differentiated
        total_extrusion = (
            p.mechanical_feedback*p.crowding_extrusion
            *np.maximum(0.0, occupancy-p.carrying_capacity)
        )
        denominator = np.maximum(occupancy, 1e-15)
        extrusion_renewing = total_extrusion*renewing/denominator
        extrusion_differentiated = total_extrusion*differentiated/denominator
        niche_recovery = p.niche_recovery*(1-niche)
        niche_suppression = p.differentiated_feedback*differentiated*niche
        return {
            "proliferation": proliferation,
            "differentiation": differentiation,
            "renewing_loss": renewing_loss,
            "differentiated_loss": differentiated_loss,
            "extrusion_renewing": extrusion_renewing,
            "extrusion_differentiated": extrusion_differentiated,
            "extrusion": total_extrusion,
            "niche_recovery": niche_recovery,
            "niche_suppression": niche_suppression,
        }

    def reaction(self, state=None):
        f = self.fluxes(state)
        return np.array([
            f["proliferation"]-f["differentiation"]-f["renewing_loss"]-f["extrusion_renewing"],
            f["differentiation"]-f["differentiated_loss"]-f["extrusion_differentiated"],
            f["niche_recovery"]-f["niche_suppression"],
        ])

    def _weighted_mean(self, values):
        return float(values @ self.ops.mass/self.ops.mass.sum())

    @property
    def diffusivities(self):
        return (self.parameters.renewing_diffusion, self.parameters.differentiated_diffusion,
                self.parameters.niche_diffusion)

    def step(self, dt=None):
        trial_dt = self.config.dt if dt is None else float(dt)
        if not np.isfinite(trial_dt) or trial_dt <= 0:
            raise ValueError("dt must be positive and finite")
        old = self.state.copy()
        fluxes = self.fluxes(old)
        reaction = self.reaction(old)
        while True:
            source = old+trial_dt*reaction
            if np.all(np.isfinite(source)) and np.min(source) >= 0:
                candidate = np.array([
                    self.ops.solve(self.ops.mass*source[index], trial_dt*diffusion/(self.config.length**2), index)
                    for index, diffusion in enumerate(self.diffusivities)
                ])
                if np.all(np.isfinite(candidate)) and np.min(candidate) >= -1e-12:
                    break
            self.rejected += 1
            trial_dt *= 0.5
            if trial_dt < min(self.config.min_dt, self.config.dt):
                raise RuntimeError("Organization step rejected below min_dt")
        self.state = candidate
        self.t += trial_dt
        self.steps += 1
        for name in self.cumulative:
            self.cumulative[name] += trial_dt*self._weighted_mean(fluxes[name])
        return self.state.copy()

    def run_until(self, end_time):
        end_time = float(end_time)
        if not np.isfinite(end_time) or end_time < self.t:
            raise ValueError("end_time must be finite and no earlier than current time")
        while end_time-self.t > 1e-12:
            self.step(min(self.config.dt, end_time-self.t))
        return self.diagnostics()

    def perturb(self, fraction=0.6, radius=0.35, center=(0.0, 0.5), population="both"):
        """Remove cells locally, using angular distance on a sphere."""
        if population not in {"renewing", "differentiated", "both"}:
            raise ValueError("population must be renewing, differentiated, or both")
        values = np.asarray([fraction, radius, *center], dtype=float)
        if not np.all(np.isfinite(values)) or not 0 <= fraction <= 1 or radius <= 0:
            raise ValueError("fraction must be in [0, 1] and radius must be positive")
        if len(center) != 2 or not 0 <= center[0] <= 1 or not 0 <= center[1] <= 1:
            raise ValueError("center coordinates must lie in [0, 1]")
        if self.config.geometry == "sphere":
            points = self.ops.points/np.linalg.norm(self.ops.points, axis=1)[:, None]
            distance = np.arccos(np.clip(points @ sphere_direction(*center), -1, 1))
        else:
            distance = np.linalg.norm(self.ops.points[:, :2]-np.asarray(center), axis=1)
        multiplier = 1-fraction*np.exp(-distance**2/(2*radius**2))
        if population in {"renewing", "both"}:
            self.state[0] *= multiplier
        if population in {"differentiated", "both"}:
            self.state[1] *= multiplier
        return self.diagnostics()

    def diagnostics(self):
        renewing, differentiated, niche = self.state[:3]
        occupancy = self.occupancy
        means = np.array([self._weighted_mean(field) for field in self.state])
        mean_occupancy = means[0]+means[1]
        occupancy_std = float(np.sqrt(self._weighted_mean((occupancy-mean_occupancy)**2)))
        differentiated_fraction = self._weighted_mean(differentiated/np.maximum(occupancy, 1e-15))
        fluxes = self.fluxes()
        flux_means = {f"flux_{name}": self._weighted_mean(fluxes[name]) for name in self.cumulative}
        net_population_source = (
            flux_means["flux_proliferation"]-flux_means["flux_renewing_loss"]
            -flux_means["flux_differentiated_loss"]-flux_means["flux_extrusion"]
        )
        return {
            "time": self.t,
            "mean_renewing": float(means[0]),
            "mean_differentiated": float(means[1]),
            "mean_niche": float(means[2]),
            "mean_occupancy": float(mean_occupancy),
            "differentiated_fraction": differentiated_fraction,
            "std_occupancy": occupancy_std,
            "cv_occupancy": occupancy_std/max(float(mean_occupancy), 1e-15),
            "net_population_source": net_population_source,
            "steps": self.steps,
            "rejected": self.rejected,
            **flux_means,
            **{f"cumulative_{name}": value for name, value in self.cumulative.items()},
        }

    def close(self):
        self.ops.close()


def run_aim1_protocol(protocol):
    """Run establishment, turnover, repair, and imposed-pattern controls."""
    if getattr(protocol, "model", None) == "spatial":
        from .spatial_protocol import run_spatial_protocol
        return run_spatial_protocol(protocol)
    if not isinstance(protocol, Aim1Protocol):
        raise TypeError("protocol must be an Aim1Protocol")
    config = Config(
        backend="scipy", geometry=protocol.geometry, n=protocol.n, dt=protocol.dt,
        length=1, max_length=1, growth_rate=0,
        t_end=protocol.settle_time+protocol.turnover_time+protocol.repair_time,
    )
    started = perf_counter()
    report = {
        "protocol": protocol.to_dict(), "establishment": {}, "repair": {},
        "maintenance_control": {}, "mechanism_controls": {},
    }
    for offset, kind in enumerate(("low", "high", "mosaic", "segregated", "random")):
        model = OrganizationModel(config, protocol.parameters, protocol.seed+offset, kind)
        try:
            initial = model.diagnostics()
            model.run_until(protocol.settle_time)
            report["establishment"][kind] = {"initial": initial, "settled": model.diagnostics()}
        finally:
            model.close()
    model = OrganizationModel(config, protocol.parameters, protocol.seed, "random")
    try:
        model.run_until(protocol.settle_time)
        before_cumulative = model.cumulative.copy()
        before = model.diagnostics()
        model.run_until(protocol.settle_time+protocol.turnover_time)
        report["turnover"] = {
            "before": before,
            "after": model.diagnostics(),
            "interval_fluxes": {name: model.cumulative[name]-before_cumulative[name] for name in model.cumulative},
        }
    finally:
        model.close()
    for radius, fraction in ((0.20, 0.40), (0.35, 0.60), (0.50, 0.80)):
        model = OrganizationModel(config, protocol.parameters, protocol.seed, "random")
        try:
            model.run_until(protocol.settle_time)
            baseline = model.diagnostics()
            disturbed = model.perturb(fraction=fraction, radius=radius, population="both")
            model.run_until(protocol.settle_time+protocol.repair_time)
            label = f"radius_{radius:.2f}_fraction_{fraction:.2f}"
            report["repair"][label] = {
                "baseline": baseline, "disturbed": disturbed, "recovered": model.diagnostics(),
            }
        finally:
            model.close()
    imposed = OrganizationModel(config, protocol.parameters, protocol.seed, "imposed_pattern")
    try:
        initial = imposed.diagnostics()
        imposed.run_until(protocol.settle_time)
        report["maintenance_control"] = {"initial": initial, "settled": imposed.diagnostics()}
    finally:
        imposed.close()
    controls = {
        "no_differentiated_feedback": replace(protocol.parameters, differentiated_feedback=0),
        "no_niche_coupling": replace(protocol.parameters, niche_coupling=0),
        "no_mechanical_feedback": replace(protocol.parameters, mechanical_feedback=0),
    }
    for offset, (label, parameters) in enumerate(controls.items(), start=20):
        model = OrganizationModel(config, parameters, protocol.seed+offset, "random")
        try:
            model.run_until(protocol.settle_time)
            settled = model.diagnostics()
            disturbed = model.perturb(fraction=0.60, radius=0.35, population="both")
            model.run_until(protocol.settle_time+protocol.repair_time)
            report["mechanism_controls"][label] = {
                "changed_parameter": {
                    "differentiated_feedback": parameters.differentiated_feedback,
                    "niche_coupling": parameters.niche_coupling,
                    "mechanical_feedback": parameters.mechanical_feedback,
                },
                "settled": settled,
                "disturbed": disturbed,
                "recovered": model.diagnostics(),
            }
        finally:
            model.close()
    report["criteria"] = evaluate_aim1_report(report)
    report["wall_seconds"] = perf_counter()-started
    return report


def evaluate_aim1_report(report):
    """Apply preregistered, target-free acceptance thresholds to a report."""
    establishment = report["establishment"]
    settled = [entry["settled"] for entry in establishment.values()]
    occupancies = np.array([item["mean_occupancy"] for item in settled])
    fractions = np.array([item["differentiated_fraction"] for item in settled])
    establishment_checks = {
        "all_cv_below_0.002": bool(all(item["cv_occupancy"] < 0.002 for item in settled)),
        "occupancy_range_below_1_percent": bool(np.ptp(occupancies)/np.mean(occupancies) < 0.01),
        "lineage_fraction_range_below_1_percent": bool(np.ptp(fractions) < 0.01),
    }
    turnover = report["turnover"]
    interval = turnover["interval_fluxes"]
    turnover_checks = {
        "positive_replacement_flux": bool(
            interval["proliferation"] > 0.1
            and interval["differentiation"] > 0.1
            and interval["differentiated_loss"] > 0.1
        ),
        "occupancy_drift_below_1_percent": bool(
            abs(turnover["after"]["mean_occupancy"]-turnover["before"]["mean_occupancy"])
            /turnover["before"]["mean_occupancy"] < 0.01
        ),
    }
    repair_checks = {}
    for label, entry in report["repair"].items():
        baseline, disturbed, recovered = entry["baseline"], entry["disturbed"], entry["recovered"]
        repair_checks[label] = {
            "occupancy_error_below_1_percent": bool(
                abs(recovered["mean_occupancy"]-baseline["mean_occupancy"])
                /baseline["mean_occupancy"] < 0.01
            ),
            "cv_below_0.01": bool(recovered["cv_occupancy"] < 0.01),
            "spatial_defect_reduced_90_percent": bool(
                recovered["cv_occupancy"] < 0.1*disturbed["cv_occupancy"]
            ),
        }
    maintenance = report["maintenance_control"]
    reference_occupancy = float(np.mean(occupancies))
    maintenance_checks = {
        "imposed_pattern_not_preserved": bool(
            maintenance["settled"]["cv_occupancy"]
            < 0.1*maintenance["initial"]["cv_occupancy"]
        ),
        "settles_to_common_occupancy": bool(
            abs(maintenance["settled"]["mean_occupancy"]-reference_occupancy)
            /reference_occupancy < 0.01
        ),
    }
    mechanisms = report["mechanism_controls"]
    mechanism_effects = {
        label: {
            "settled_occupancy_ratio_to_normal": entry["settled"]["mean_occupancy"]/reference_occupancy,
            "recovered_cv": entry["recovered"]["cv_occupancy"],
        }
        for label, entry in mechanisms.items()
    }
    groups = [establishment_checks, turnover_checks, maintenance_checks]
    all_passed = all(value for group in groups for value in group.values()) and all(
        value for lesion in repair_checks.values() for value in lesion.values()
    )
    return {
        "establishment": establishment_checks,
        "turnover": turnover_checks,
        "repair": repair_checks,
        "maintenance_control": maintenance_checks,
        "mechanism_effects": mechanism_effects,
        "all_required_evidence_passed": bool(all_passed),
    }
