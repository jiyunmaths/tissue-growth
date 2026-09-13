"""Conservative reference-surface IMEX stepping under isotropic growth."""
import json
import sys
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from .config import Config
from .operators import make_operators
from .geometry import sphere_direction


class Simulation:
    def __init__(self, config):
        start = perf_counter()
        self.config = config
        self.ops = make_operators(config)
        self.t, self.length = 0.0, config.length
        self.growth_rate = config.growth_rate
        self.steps = self.rejected = 0
        self.compute_seconds = 0.0
        self.last_dt = self.last_step_seconds = self.balance_error = 0.0
        self.last_iterations = 0
        self.events = []
        # Coordinate-based modes give the same initial field under DOF reordering.
        rng = np.random.default_rng(config.seed)
        xy = self.ops.points[:, :2]
        noise = np.zeros(len(xy))
        for _ in range(12):
            if config.geometry == "sphere":
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                frequency = rng.integers(1, 9)
                phase = rng.uniform(0, 2*np.pi)
                noise += rng.uniform(-1, 1)*np.cos(np.pi*frequency*(self.ops.points @ direction)+phase)/12
            else:
                k = rng.integers(1, 9, size=2)
                noise += rng.uniform(-1, 1) * np.cos(np.pi*k[0]*xy[:,0]) * np.cos(np.pi*k[1]*xy[:,1]) / 12
        equilibrium = np.array([config.a + config.b, config.b/(config.a + config.b)**2])
        self.c = equilibrium[:,None] * (1 + config.noise*noise[None,:])
        self.setup_seconds = perf_counter() - start

    def reaction(self):
        u, v = self.c
        product = u*u*v
        return self.config.reaction_scale * np.array([self.config.a-u+product, self.config.b-product])

    def step(self, stop_at=None):
        end = self.config.t_end if stop_at is None else min(stop_at, self.config.t_end)
        if end - self.t <= 1e-12:
            return False
        started = perf_counter()
        h = min(self.config.dt, end-self.t)
        old_j = self.length**2
        old_q = old_j*self.c
        reaction = self.reaction()
        while True:
            # Evaluate the cap without exponent overflow. J is an area Jacobian.
            log_ratio = np.log(self.config.max_length/self.length)
            new_length = self.length * np.exp(min(self.growth_rate*h, log_ratio))
            new_j = new_length**2
            source = old_q + h*old_j*reaction
            if np.all(np.isfinite(source)) and np.min(source) >= 0:
                candidate = []
                iterations = 0
                for i, diffusion in enumerate((self.config.du, self.config.dv)):
                    q = self.ops.solve(self.ops.mass*source[i], h*diffusion/new_j, i)
                    candidate.append(q/new_j)
                    iterations += self.ops.iterations
                candidate = np.array(candidate)
                if np.all(np.isfinite(candidate)) and np.min(candidate) >= -1e-12:
                    break
            self.rejected += 1
            h *= 0.5
            if h < self.config.min_dt:
                raise RuntimeError("Step rejected below min_dt: reduce dt or inspect reaction parameters. No concentration clipping was applied.")
        residual = new_j * (candidate @ self.ops.mass) - source @ self.ops.mass
        scale = max(float(np.max(np.abs(source @ self.ops.mass))), 1e-15)
        self.balance_error = float(np.max(np.abs(residual))/scale)
        self.c, self.length = candidate, float(new_length)
        self.t = min(end, self.t+h)
        self.steps += 1
        self.last_dt, self.last_iterations = h, iterations
        self.last_step_seconds = perf_counter()-started
        self.compute_seconds += self.last_step_seconds
        return True

    def set_growth_rate(self, rate):
        rate = float(rate)
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("Growth rate must be finite and nonnegative")
        self.events.append({"time": self.t, "type": "growth_rate", "old": self.growth_rate, "new": rate})
        self.growth_rate = rate

    def perturb(self, x=0.5, y=0.5, radius=0.12, fraction=0.5):
        values = np.array([x,y,radius,fraction], dtype=float)
        if not np.all(np.isfinite(values)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < radius <= 1 and 0 <= fraction <= 1):
            raise ValueError("Perturbation requires x,y,fraction in [0,1], radius in (0,1]")
        if self.config.geometry == "sphere":
            # Great-circle distance on the unit sphere; no longitude seam or
            # depletion through the interior onto the opposite hemisphere.
            points = self.ops.points / np.linalg.norm(self.ops.points, axis=1)[:, None]
            distance2 = np.arccos(np.clip(points @ sphere_direction(x, y), -1, 1))**2
        else:
            distance2 = np.sum((self.ops.points[:,:2]-[x,y])**2, axis=1)
        multiplier = 1-fraction*np.exp(-distance2/(2*radius**2))
        before = float(self.length**2 * (self.c[0] @ self.ops.mass))
        self.c[0] *= multiplier
        after = float(self.length**2 * (self.c[0] @ self.ops.mass))
        self.events.append({"time":self.t, "type":"activator_depletion", "x":x,"y":y,"radius":radius,"fraction":fraction,"removed_amount":before-after})

    def diagnostics(self):
        import psutil
        try:
            process = None
            if sys.platform.startswith("linux"):
                rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**2
            else:
                process = psutil.Process()
                rss = process.memory_info().rss / 1024**2
        except (psutil.Error, OSError):
            process, rss = None, None
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
        except ImportError:
            peak = (getattr(process.memory_info(), "peak_wset", process.memory_info().rss) / 1024**2) if process is not None else None
        means = self.c @ self.ops.mass / self.ops.mass.sum()
        std_u = float(np.sqrt(((self.c[0]-means[0])**2) @ self.ops.mass / self.ops.mass.sum()))
        return {"time":self.t, "length":self.length, "area":float(self.length**2*self.ops.mass.sum()),
                "mean_u":float(means[0]), "mean_v":float(means[1]), "std_u":std_u,
                "amount_u":float(self.length**2*(self.c[0] @ self.ops.mass)),
                "amount_v":float(self.length**2*(self.c[1] @ self.ops.mass)),
                "min_u":float(self.c[0].min()),"min_v":float(self.c[1].min()),
                "steps":self.steps,"rejected":self.rejected,"last_dt":self.last_dt,
                "step_ms":1000*self.last_step_seconds,"compute_seconds":self.compute_seconds,
                "setup_seconds":self.setup_seconds,"balance_error":self.balance_error,
                "linear_iterations":self.last_iterations,"dofs":int(self.c.size),
                "worker_rss_mb":rss,"process_peak_rss_mb":peak}

    def checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema":1,"config":self.config.to_dict(),"time":self.t,"length":self.length,
                    "growth_rate":self.growth_rate,"steps":self.steps,"rejected":self.rejected,
                    "compute_seconds":self.compute_seconds,"events":self.events}
        temporary = path.with_name(path.name+".tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(f, metadata=json.dumps(metadata), c=self.c, points=self.ops.points,
                                triangles=self.ops.triangles)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("schema") != 1:
                raise ValueError("Unsupported checkpoint schema")
            sim = cls(Config.from_dict(metadata["config"]))
            try:
                # Remap by reference coordinates, so local DOF ordering may differ.
                from scipy.spatial import cKDTree
                distance, order = cKDTree(data["points"]).query(sim.ops.points)
                if data["c"].shape != sim.c.shape or distance.max() > 1e-12 or len(np.unique(order)) != len(order):
                    raise ValueError("Checkpoint mesh does not match reconstructed mesh")
                # A different triangulation is not a restart of the same discretization.
                old_cells = np.sort(data["triangles"], axis=1)
                new_cells = np.sort(order[sim.ops.triangles], axis=1)
                sort_rows = lambda x: x[np.lexsort(x.T[::-1])]
                if not np.array_equal(sort_rows(old_cells), sort_rows(new_cells)):
                    raise ValueError("Checkpoint triangulation changed; use the original solver environment")
                sim.c = data["c"][:,order].copy()
                if not np.all(np.isfinite(sim.c)) or sim.c.min() < -1e-12:
                    raise ValueError("Invalid checkpoint concentrations")
                sim.t = float(metadata["time"])
                sim.length = float(metadata["length"])
                sim.growth_rate = float(metadata["growth_rate"])
                if not (0 <= sim.t <= sim.config.t_end and sim.config.length <= sim.length <= sim.config.max_length*(1+1e-12) and np.isfinite(sim.growth_rate) and sim.growth_rate >= 0):
                    raise ValueError("Invalid checkpoint growth state")
                sim.steps, sim.rejected = metadata["steps"], metadata["rejected"]
                sim.compute_seconds, sim.events = metadata["compute_seconds"], metadata["events"]
                return sim
            except Exception:
                sim.close()
                raise

    def close(self):
        self.ops.close()
