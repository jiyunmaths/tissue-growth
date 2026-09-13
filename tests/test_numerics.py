from dataclasses import replace
import importlib.util
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from tissue_growth.config import Config
from tissue_growth.solver import Simulation

BACKENDS = ["scipy"] + (["fenicsx"] if importlib.util.find_spec("dolfinx") else [])


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def integrate(sim, end=None):
    while sim.step(end):
        pass


def test_growth_conserves_amount_and_dilutes_by_area(backend):
    cfg = Config(backend=backend,n=8,noise=0,reaction_scale=0,growth_rate=0.2,max_length=3.5,t_end=2,dt=0.04)
    sim = Simulation(cfg)
    try:
        before = sim.c.copy()*cfg.length**2
        integrate(sim)
        np.testing.assert_allclose(sim.c*sim.length**2,before,rtol=1e-8,atol=1e-10)
        assert sim.length == pytest.approx(cfg.max_length)
        assert sim.balance_error < 1e-8
    finally:
        sim.close()


def test_homogeneous_reaction_matches_independent_ode(backend):
    cfg = Config(backend=backend,n=6,noise=0,growth_rate=0,dt=0.0005,t_end=0.2)
    sim = Simulation(cfg)
    try:
        sim.c[:] = np.array([1.2,0.7])[:,None]
        def rhs(t,z):
            u,v=z
            return [cfg.a-u+u*u*v,cfg.b-u*u*v]
        reference = solve_ivp(rhs,[0,cfg.t_end],[1.2,0.7],rtol=1e-12,atol=1e-13).y[:,-1]
        integrate(sim)
        np.testing.assert_allclose(sim.c.mean(axis=1),reference,atol=8e-5)
    finally:
        sim.close()


def test_diffusion_cosine_mode_converges_with_mesh(backend):
    errors=[]
    for n in (8,16,32):
        cfg=Config(backend=backend,n=n,noise=0,growth_rate=0,length=1,max_length=1,reaction_scale=0,du=0.2,dv=0.4,dt=0.0002,t_end=0.04)
        sim=Simulation(cfg)
        try:
            mode=np.cos(np.pi*sim.ops.points[:,0])*np.cos(np.pi*sim.ops.points[:,1])
            sim.c[0]=1+0.1*mode
            integrate(sim)
            exact=1+0.1*np.exp(-2*np.pi**2*cfg.du*cfg.t_end)*mode
            errors.append(np.sqrt((sim.c[0]-exact)**2 @ sim.ops.mass))
        finally:
            sim.close()
    assert errors[1] < 0.5*errors[0], errors
    assert errors[2] < 0.65*errors[1], errors


def test_restart_reproduces_intervention_trajectory(tmp_path,backend):
    sim=Simulation(Config(backend=backend,n=8,t_end=0.6,dt=0.01))
    restored=None
    try:
        integrate(sim,0.2)
        sim.perturb(x=0.3,fraction=0.4)
        sim.set_growth_rate(0.08)
        checkpoint=tmp_path/"checkpoint.npz"
        sim.checkpoint(checkpoint)
        restored=Simulation.load(checkpoint)
        integrate(sim)
        integrate(restored)
        np.testing.assert_allclose(restored.c,sim.c,rtol=1e-9,atol=1e-11)
        assert restored.events == sim.events
        assert restored.length == pytest.approx(sim.length)
    finally:
        sim.close()
        if restored:
            restored.close()


def test_large_reaction_step_rejects_without_clipping(backend):
    sim=Simulation(Config(backend=backend,n=4,noise=0,dt=1,t_end=2))
    try:
        sim.c[:]=np.array([4.,1.])[:,None]
        sim.step()
        assert sim.rejected > 0
        assert sim.last_dt < 1
        assert sim.c.min() >= 0
    finally:
        sim.close()


def test_invalid_parameters_rejected():
    for update in ({"growth_rate":float("nan")},{"n":2},{"backend":"auto"},{"noise":1},{"dt":-1}):
        with pytest.raises(ValueError):
            replace(Config(),**update)


def test_operator_agreement():
    if "fenicsx" not in BACKENDS:
        pytest.skip("FEniCSx is not installed")
    from scipy.spatial import cKDTree
    a=Simulation(Config(backend="scipy",n=8))
    b=Simulation(Config(backend="fenicsx",n=8))
    try:
        distance,order=cKDTree(a.ops.points).query(b.ops.points)
        assert distance.max() < 1e-12
        # Right-triangle P1 stiffness equals the 5-point grid stencil for both diagonals.
        x=np.sin(a.ops.points[:,0])+a.ops.points[:,1]**2
        vector=b.ops.mass_vec.duplicate()
        out=b.ops.mass_vec.duplicate()
        try:
            vector.array[:]=x[order]
            b.ops.stiffness.mult(vector,out)
            np.testing.assert_allclose(out.array,(a.ops.stiffness @ x)[order],atol=1e-12)
            assert a.ops.mass.sum() == pytest.approx(b.ops.mass.sum())
        finally:
            vector.destroy()
            out.destroy()
    finally:
        a.close()
        b.close()
