"""Convert a saved state to VTU for ParaView; does not require FEniCSx.

Usage: python examples/export_checkpoint.py runs/NAME/checkpoint.npz field.vtu
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyvista as pv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new filename")
    with np.load(args.checkpoint, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta.get("schema") != 1:
            parser.error("Unsupported checkpoint schema")
        faces = np.column_stack([np.full(len(data["triangles"]), 3), data["triangles"]]).ravel()
        grid = pv.UnstructuredGrid(faces, np.full(len(data["triangles"]), pv.CellType.TRIANGLE), data["points"]*meta["length"])
        grid.point_data["activator_u"] = data["c"][0]
        grid.point_data["inhibitor_v"] = data["c"][1]
        grid.field_data["simulation_time"] = np.array([meta["time"]])
        grid.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
