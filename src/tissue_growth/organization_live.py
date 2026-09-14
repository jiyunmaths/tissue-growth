"""Live Aim 1 adapter for the shared worker and recorder."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .organization import OrganizationModel, OrganizationParameters


class OrganizationSimulation(OrganizationModel):
    PARAMETERS = OrganizationParameters
    MODEL_NAME = "aim1"

    def __init__(self, config, options):
        self.initial_condition = options["initial_condition"]
        super().__init__(config, self.PARAMETERS.from_dict(options["parameters"]),
                         config.seed, self.initial_condition)
        self.events = []
        self.growth_rate = 0.0
        self.last_dt = self.last_step_seconds = 0.0

    @property
    def c(self):
        return self.state

    def provenance(self):
        return {"model": self.MODEL_NAME, "parameters": asdict(self.parameters),
                "initial_condition": self.initial_condition, "seed": self.seed}

    def step(self):
        if self.config.t_end-self.t <= 1e-12:
            return False
        start, before = perf_counter(), self.t
        super().step(min(self.config.dt, self.config.t_end-self.t))
        self.last_dt = self.t-before
        self.last_step_seconds = perf_counter()-start
        return True

    def set_growth_rate(self, rate):
        raise ValueError("Aim 1 uses a fixed surface")

    def perturb(self, x=0.5, y=0.5, radius=0.35, fraction=0.6, population="both", erase_signals=False):
        before = self.state[:2] @ self.ops.mass
        if self.MODEL_NAME == "aim1-spatial":
            super().perturb(fraction, radius, (x, y), population, erase_signals=erase_signals)
        else:
            super().perturb(fraction, radius, (x, y), population)
        removed = before-self.state[:2] @ self.ops.mass
        self.events.append({"time": self.t, "type": "cell_depletion", "population": population,
                            "x": x, "y": y, "radius": radius, "fraction": fraction,
                            "erase_signals": erase_signals,
                            "removed_renewing": float(removed[0]),
                            "removed_differentiated": float(removed[1])})

    def diagnostics(self):
        import psutil
        try:
            memory = psutil.Process().memory_info().rss/1024**2
        except (psutil.Error, OSError):
            memory = None
        return {**super().diagnostics(), "length": self.config.length,
                "worker_rss_mb": memory, "dofs": int(self.state.size),
                "step_ms": 1000*self.last_step_seconds, "last_dt": self.last_dt}

    def checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema": "aim1-snapshot-1", **self.provenance(),
                    "config": self.config.to_dict(), "time": self.t,
                    "steps": self.steps, "rejected": self.rejected,
                    "cumulative": self.cumulative, "events": self.events,
                    "fields": list(self.FIELD_NAMES)}
        temporary = path.with_name(path.name+".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata), c=self.state,
                                points=self.ops.points, triangles=self.ops.triangles)
        temporary.replace(path)


from .spatial_organization import SpatialOrganizationModel, SpatialParameters


class SpatialOrganizationSimulation(OrganizationSimulation, SpatialOrganizationModel):
    PARAMETERS = SpatialParameters
    MODEL_NAME = "aim1-spatial"
