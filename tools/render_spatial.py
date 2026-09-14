"""Render front/back cell-composition maps for the two spatial Aim 1 presets."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from tissue_growth.config import Config
from tissue_growth.organization import Aim1Protocol
from tissue_growth.spatial_organization import SpatialOrganizationModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/aim1-spatial-patterns.png"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout="constrained")
    for row, name in enumerate(("niches", "bands")):
        protocol = Aim1Protocol.read(root/"configs"/f"aim1_{name}.json")
        config = Config(backend="scipy", geometry="sphere", n=protocol.n, dt=protocol.dt,
                        length=1, max_length=1, growth_rate=0)
        model = SpatialOrganizationModel(config, protocol.parameters, protocol.seed, "near_uniform")
        try:
            model.run_until(protocol.settle_time+protocol.turnover_time)
            values = model.state[0]/model.occupancy
            points, cells = model.ops.points, model.ops.triangles
            for col, side in enumerate((1, -1)):
                mesh = tri.Triangulation(side*points[:, 1], points[:, 2], cells)
                mesh.set_mask(side*points[cells, 0].mean(axis=1) < 0)
                artist = axes[row, col].tripcolor(mesh, values, vmin=0, vmax=1,
                                                shading="gouraud", cmap="viridis")
                axes[row, col].set_aspect("equal")
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{name.capitalize()} · {'front' if side == 1 else 'back'}")
            print(name, model.diagnostics(), flush=True)
        finally:
            model.close()
    fig.colorbar(artist, ax=axes, shrink=0.7, label="Renewing-cell fraction R / (R + D)")
    fig.suptitle("Spatial organization from near-uniform cells and 1% signal noise\n"
                 f"n={protocol.n}, dt={protocol.dt}, seed={protocol.seed}, t={model.t:g}", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
