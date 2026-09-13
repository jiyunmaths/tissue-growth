import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone


class Recorder:
    def __init__(self, directory, simulation):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.file = (self.directory / "diagnostics.csv").open("w", newline="")
        self.writer = None
        versions = {}
        for name in ("tissue-growth", "fenics-dolfinx", "petsc4py", "numpy", "scipy", "pyvista", "trame"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        provenance = {"created_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),
                      "python":sys.version,"packages":versions,"config":simulation.config.to_dict()}
        (self.directory / "run.json").write_text(json.dumps(provenance, indent=2))
        self.next_diag = simulation.t
        self.next_snapshot = simulation.t
        self.last_diag_step = None
        self.closed = False

    def record(self, sim, force=False):
        if (force or sim.t+1e-12 >= self.next_diag) and self.last_diag_step != (sim.steps, len(sim.events)):
            row = sim.diagnostics()
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=row.keys())
                self.writer.writeheader()
            self.writer.writerow(row)
            self.file.flush()
            self.last_diag_step = (sim.steps, len(sim.events))
            self.next_diag = sim.t+sim.config.diagnostic_every
        if force or sim.t+1e-12 >= self.next_snapshot:
            sim.checkpoint(self.directory / f"state_{sim.steps:08d}.npz")
            self.next_snapshot = sim.t+sim.config.snapshot_every
        (self.directory / "events.json").write_text(json.dumps(sim.events, indent=2)) if force else None

    def finish(self, sim):
        if self.closed:
            return
        try:
            self.record(sim, force=True)
            sim.checkpoint(self.directory / "checkpoint.npz")
            (self.directory / "summary.json").write_text(json.dumps(sim.diagnostics(), indent=2))
        finally:
            self.file.close()
            self.closed = True
