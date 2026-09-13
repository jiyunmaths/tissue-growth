"""Closed-surface geometry, transport, intervention, and restart checks."""
import importlib.util
import json
import numpy as np
import pytest
from tissue_growth.config import Config
from tissue_growth.geometry import sphere_mesh
from tissue_growth.solver import Simulation


BACKENDS = ["scipy"] + (["fenicsx"] if importlib.util.find_spec("dolfinx") else [])


def integrate(sim):
    while sim.step():
        pass


def test_sphere_is_closed_and_oriented():
    points, cells = sphere_mesh(8)
    assert points.shape == (4*8**2+2, 3)
    assert cells.shape == (8*8**2, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1)
    edges = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    assert len(points)-len(unique)+len(cells) == 2
    xyz = points[cells]
    assert np.all(np.einsum('ij,ij->i', np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), xyz[:, 0]) > 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_surface_growth_conserves_nonuniform_amount(backend):
    sim = Simulation(Config(backend=backend, geometry="sphere", n=8, reaction_scale=0,
                            growth_rate=0.2, max_length=3.5, t_end=2, dt=0.04))
    try:
        before = sim.c @ sim.ops.mass * sim.length**2
        integrate(sim)
        np.testing.assert_allclose(sim.c @ sim.ops.mass * sim.length**2, before, rtol=1e-8)
        assert sim.length == pytest.approx(3.5)
        assert sim.balance_error < 1e-8
        assert sim.diagnostics()['area'] == pytest.approx(sim.length**2*sim.ops.mass.sum())
        assert sim.diagnostics()['area'] == pytest.approx(4*np.pi*sim.length**2, rel=0.02)
    finally:
        sim.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_surface_uniform_dilution(backend):
    sim = Simulation(Config(backend=backend, geometry="sphere", n=6, noise=0,
                            reaction_scale=0, growth_rate=0.1, dt=0.04, t_end=0.4))
    try:
        initial = sim.c.copy()
        integrate(sim)
        np.testing.assert_allclose(sim.c, initial*(sim.config.length/sim.length)**2, rtol=1e-8)
    finally:
        sim.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_spherical_harmonic_diffusion_converges(backend):
    # z is an l=1 spherical harmonic: Delta_S z = -2z on the unit sphere.
    errors, area_errors = [], []
    for n in (4, 8, 16):
        sim = Simulation(Config(backend=backend, geometry="sphere", n=n, noise=0,
                                length=1, max_length=1, growth_rate=0, reaction_scale=0,
                                du=0.2, dt=0.0005, t_end=0.05))
        try:
            z = sim.ops.points[:, 2]
            sim.c[0] = 1+0.1*z
            integrate(sim)
            exact = 1+0.1*np.exp(-2*sim.config.du*sim.t)*z
            errors.append(np.sqrt((sim.c[0]-exact)**2 @ sim.ops.mass))
            area_errors.append(abs(sim.ops.mass.sum()-4*np.pi))
        finally:
            sim.close()
    assert errors[1] < 0.5*errors[0], errors
    assert errors[2] < 0.5*errors[1], errors
    assert area_errors[2] < 0.3*area_errors[1] < 0.09*area_errors[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_surface_depletion_and_restart(tmp_path, backend):
    sim = Simulation(Config(backend=backend, geometry="sphere", n=8, noise=0, t_end=0.2))
    restored = None
    try:
        before = sim.c.copy()
        sim.perturb(x=0, y=0.5, radius=0.12, fraction=0.5)
        pos = np.argmax(sim.ops.points[:, 0])
        neg = np.argmin(sim.ops.points[:, 0])
        assert sim.c[0, pos] == pytest.approx(before[0, pos]*0.5)
        assert sim.c[0, neg] == pytest.approx(before[0, neg])
        np.testing.assert_array_equal(sim.c[1], before[1])
        depleted = sim.c.copy()
        sim.c = before.copy()
        sim.perturb(x=1, y=0.5, radius=0.12, fraction=0.5)
        np.testing.assert_allclose(sim.c, depleted, atol=1e-14)
        sim.set_growth_rate(0.05)
        sim.step()
        path = tmp_path/'surface.npz'
        sim.checkpoint(path)
        restored = Simulation.load(path)
        assert restored.config.geometry == 'sphere'
        integrate(sim)
        integrate(restored)
        np.testing.assert_allclose(restored.c, sim.c, rtol=1e-9, atol=1e-11)
        assert restored.events == sim.events
    finally:
        sim.close()
        if restored:
            restored.close()


def test_legacy_square_checkpoint(tmp_path):
    sim = Simulation(Config(backend='scipy', n=4, t_end=0.1))
    restored = None
    try:
        path = tmp_path/'legacy.npz'
        sim.checkpoint(path)
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        metadata = json.loads(str(data['metadata']))
        del metadata['config']['geometry']
        data['metadata'] = json.dumps(metadata)
        np.savez_compressed(path, **data)
        restored = Simulation.load(path)
        assert restored.config.geometry == 'square'
        np.testing.assert_array_equal(restored.c, sim.c)
    finally:
        sim.close()
        if restored:
            restored.close()


@pytest.mark.skipif('fenicsx' not in BACKENDS, reason='FEniCSx is not installed')
def test_surface_backends_agree():
    from scipy.spatial import cKDTree
    a = Simulation(Config(backend='scipy', geometry='sphere', n=8, t_end=0.1))
    b = Simulation(Config(backend='fenicsx', geometry='sphere', n=8, t_end=0.1))
    try:
        distance, order = cKDTree(a.ops.points).query(b.ops.points)
        assert distance.max() < 1e-12
        np.testing.assert_allclose(b.ops.mass, a.ops.mass[order], atol=1e-14)
        np.testing.assert_allclose(b.c, a.c[:, order], atol=1e-14)
        integrate(a)
        integrate(b)
        np.testing.assert_allclose(b.c, a.c[:, order], rtol=1e-8, atol=1e-10)
    finally:
        a.close()
        b.close()


@pytest.mark.skipif('fenicsx' not in BACKENDS, reason='FEniCSx is not installed')
@pytest.mark.parametrize('pc', ['gamg', 'auto'])
def test_reused_multigrid_solves_current_matrix(pc):
    """Check fresh, reused, rebuilt and zero-diffusion solves against sparse LU."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree
    from tissue_growth.operators import ScipyOperators, FenicsOperators
    reference = ScipyOperators(Config(backend='scipy', geometry='sphere', n=12))
    ops = FenicsOperators(Config(geometry='sphere', n=12, pc=pc))
    try:
        _, order = cKDTree(reference.points).query(ops.points)
        rng = np.random.default_rng(13)
        for alpha in (0.03, 0.0299, 0.028, 0.028, 0.01, 0.0, 0.0, 0.02, 0.02):
            for field in (0, 1):
                rhs = reference.mass*rng.uniform(0.2, 2, len(order))
                matrix = diags(reference.mass)+alpha*reference.stiffness
                expected = spsolve(matrix, rhs)
                actual = ops.solve(rhs[order], alpha, field)
                np.testing.assert_allclose(actual, expected[order], rtol=2e-8, atol=2e-10)
    finally:
        reference.close()
        ops.close()


@pytest.mark.skipif('fenicsx' not in BACKENDS, reason='FEniCSx is not installed')
@pytest.mark.parametrize('pc', ['gamg', 'auto'])
def test_multigrid_restart_after_growth_and_depletion(tmp_path, pc):
    sim = Simulation(Config(geometry='sphere', pc=pc, n=12, noise=0.1,
                            max_length=3.05, growth_rate=0.2, t_end=0.4))
    restored = None
    try:
        while sim.step(0.2):
            pass
        sim.perturb(fraction=0.7)
        path = tmp_path/'gamg.npz'
        sim.checkpoint(path)
        restored = Simulation.load(path)
        integrate(sim)
        integrate(restored)
        np.testing.assert_allclose(restored.c, sim.c, rtol=2e-8, atol=2e-10)
        assert sim.balance_error < 1e-8
    finally:
        sim.close()
        if restored:
            restored.close()
