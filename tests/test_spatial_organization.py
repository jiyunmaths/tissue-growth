from dataclasses import replace
import numpy as np
import pytest
from scipy.optimize import root

from tissue_growth.config import Config
from tissue_growth.organization import Aim1Protocol
from tissue_growth.organization_live import SpatialOrganizationSimulation
from tissue_growth.spatial_organization import SpatialOrganizationModel, SpatialParameters
from tissue_growth.spatial_protocol import organized


def config():
    return Config(backend="scipy", geometry="sphere", n=16, dt=0.04,
                  length=1, max_length=1, growth_rate=0)


@pytest.mark.parametrize("supply", [1.24, 1.65])
def test_spatial_instability_and_repair_from_near_uniform(supply):
    model = SpatialOrganizationModel(config(), replace(SpatialParameters(), substrate_supply=supply),
                                     initial_condition="near_uniform")
    try:
        initial = model.diagnostics()
        model.run_until(180)
        established = model.diagnostics()
        assert organized(established)
        assert established["std_renewing_fraction"] > 10*initial["std_renewing_fraction"]
        before = model.state.copy()
        model.perturb(fraction=0.8, radius=0.5, erase_signals=True)
        assert np.any(model.state[2] < before[2])
        assert np.any(model.state[3] < before[3])
        restored = model.run_until(300)
        assert organized(restored)
        assert abs(restored["mean_occupancy"]/established["mean_occupancy"]-1) < 0.1
        assert restored["cumulative_differentiated_loss"] > established["cumulative_differentiated_loss"]
    finally:
        model.close()


def test_local_equilibrium_stable_but_surface_modes_unstable():
    model = SpatialOrganizationModel(config())
    try:
        def reaction(z):
            return model.reaction(np.asarray(z)[:, None])[:, 0]
        equilibrium = root(reaction, [0.25, 0.5, 1.2, 0.8])
        assert equilibrium.success
        assert np.min(equilibrium.x) > 0
        delta = 1e-6*np.eye(4)
        jacobian = np.column_stack([(reaction(equilibrium.x+d)-reaction(equilibrium.x-d))/(2e-6) for d in delta])
        assert np.linalg.eigvals(jacobian).real.max() < 0
        growth = [np.linalg.eigvals(jacobian-l*(l+1)*np.diag(model.diffusivities)).real.max() for l in range(1, 30)]
        assert max(growth) > 0.01
        assert growth[-1] < 0
    finally:
        model.close()


def test_live_spatial_snapshot_and_reaction_balance(tmp_path):
    from pathlib import Path
    import json
    p = Aim1Protocol.read(Path(__file__).resolve().parents[1]/"configs"/"aim1_niches.json")
    live = SpatialOrganizationSimulation(config(), {"parameters": p.to_dict()["parameters"], "initial_condition": "near_uniform"})
    try:
        before = live.c @ live.ops.mass
        expected = before+live.config.dt*(live.reaction() @ live.ops.mass)
        live.step()
        np.testing.assert_allclose(live.c @ live.ops.mass, expected, rtol=1e-10, atol=1e-12)
        live.perturb(erase_signals=True)
        live.checkpoint(tmp_path/"state.npz")
        with np.load(tmp_path/"state.npz", allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            assert metadata["fields"] == ["renewing", "differentiated", "niche", "substrate"]
            assert metadata["model"] == "aim1-spatial"
            assert metadata["events"][-1]["erase_signals"]
    finally:
        live.close()
