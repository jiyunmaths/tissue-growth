# Mathematical model and discretization

## Domain and concentrations

The tissue is either a closed spherical surface `Gamma(t) = {x in R^3 : |x| = L(t)}` (`geometry="sphere"`) or the original square `Omega(t) = [0,L(t)]^2` (`geometry="square"`). The `length` parameter is radius on the sphere and side length on the square. Missing geometry keys in older configurations or checkpoints mean square. `configs/sphere.json` provides the recommended surface example. Growth is prescribed and isotropic:

\[
\dot L = gL,\quad L\le L_{max}.
\]

At the growth cap the size remains fixed. GUI interventions make `g` piecewise constant. Material coordinates are `y=x/L(t)` and tissue velocity before the cap is `w(x,t)=g x`. Both species are transported with this tissue velocity. On the square, diffusive boundary flux is zero. The sphere has no boundary; diffusion is tangential to the surface and no concentration field is defined inside it.

On the square, the Eulerian conservation equation is

\[
\partial_t c_i+\nabla_x\cdot(c_i w)=D_i\Delta_x c_i+R_i(c).
\]

On the sphere, material surface conservation is

\[
\partial_t^\bullet c_i+c_i\nabla_\Gamma\cdot w
=D_i\Delta_\Gamma c_i+R_i(c),\qquad
\nabla_\Gamma\cdot w=2\dot L/L.
\]

Here the material derivative follows a tissue point, and `Delta_Gamma` is the Laplace–Beltrami operator. On the fixed unit reference surface either equation becomes

\[
\partial_t c_i|_y=\frac{D_i}{L^2}\Delta_y c_i+R_i(c)-2\frac{\dot L}{L}c_i.
\]

For the sphere, `Delta_y` is the unit-sphere surface Laplacian. The factor **2** is essential: concentrations dilute with *surface area*, including on a surface embedded in 3D. There is no volume dilution term. The coordinate transformation is exact for uniform isotropic growth; planar triangles approximate the sphere's geometry.

## Surface mesh and operators

For `geometry="sphere"`, subdivide each octahedron edge into `n` segments, triangulate the faces, and project all vertices onto the unit sphere. Integer lattice keys weld shared vertices at edges and poles. The closed, outward-oriented mesh has `8*n^2` triangular faces, `4*n^2+2` vertices, and `8*n^2+4` scalar concentration unknowns. Physical node coordinates are `L(t)*y`; there is no remeshing. Refining `n` improves both geometry and field resolution.

Both backends use this same spherical connectivity. DOLFINx creates a mesh of topological dimension 2 and geometric dimension 3, using P1 coordinate and concentration elements. SciPy assembles the surface operators independently: for a triangle with oriented area vector `N=(p1-p0) cross (p2-p0)`, each basis gradient is `N cross e_opposite / |N|^2`, using cyclically oriented opposite edges. The local stiffness is `area * grad(phi_i) dot grad(phi_j)`, and each vertex receives lumped mass `area/3`. Thus gradients lie in the triangle tangent plane. Surface area is integrated on the polyhedral mesh; its reference area converges to `4*pi` under refinement.

The square still uses two right triangles per grid cell, with `2*(n+1)^2` scalar unknowns. The time integrator below applies to both geometries with their respective stiffness and mass matrices.

The current growing-sphere configuration uses `n=64` (32,768 triangles and 32,772 scalar unknowns); the earlier validation used `n=24`. Changing `n` refines the actual numerical mesh and requires a new run. Existing checkpoints retain their saved mesh. This refinement improves resolution without changing the continuum instability conditions.

## Reaction kinetics

Use the dimensionless Schnakenberg system as a testbed:

\[
R_u=\rho(a-u+u^2v),\qquad R_v=\rho(b-u^2v).
\]

The nongrowing homogeneous equilibrium is `u*=a+b`, `v*=b/(a+b)^2`. Initial conditions add seeded, bounded cosine perturbations to that equilibrium. Coordinate-based initialization makes the initial field independent of local DOF ordering. It is not spatial white noise and is not intended to represent measured biological fluctuations.

On the sphere, both concentrations are multiplied by `1 + noise*eta(y)`, where

\[
\eta(y)=\frac1{12}\sum_{r=1}^{12}A_r\cos(\pi k_r d_r\cdot y+\psi_r).
\]

For each term, NumPy `default_rng(seed)` draws three standard normals and normalizes them to get direction `d_r`, then draws an integer `k_r` from 1 through 8, a phase uniformly in `[0,2*pi)`, and an amplitude uniformly in `[-1,1)`. This smooth ambient-coordinate field avoids longitude seams and pole singularities, with `|eta| <= 1`. The square retains its product-cosine initialization described in the README.

For `a=0.1`, `b=0.9`, `Du=1`, `Dv=20`, and `rho=1`, the nongrowing linear instability band is approximately `0.07396 < k^2 < 0.67604`. On a square, Neumann modes have `k^2=pi^2(m^2+n^2)/L^2`, excluding `(0,0)`. The square's default initial side length 3 has no mode in this band, whereas larger sizes allow unstable modes. The sphere has different modes: spherical harmonics have `k^2=l*(l+1)/L^2`, so at radius 3, degrees 1 and 2 already fall in the static band. The sphere example therefore need not begin in a stable regime. During growth the homogeneous state and dilution change; this static band alone does **not** predict finite-time transitions. Pattern suppression, amplification, and mode competition are results to investigate, not imposed behavior.

For practical pattern selection, `configs/sphere_spots.json` and `configs/sphere_stripes.json` provide fixed-radius controls with different reaction parameters. The [README pattern guide](../README.md#resolving-and-obtaining-spots-or-stripes) gives their parameters, the spherical-mode Jacobian test, and convergence checks. They isolate reaction–diffusion dynamics from growth; a positive linear growth rate for a spatial mode predicts amplification, not a guaranteed nonlinear morphology or stripe orientation.

## Conservative time step

Write `J=L^2` and `q_i=J c_i`, so

\[
\partial_t q_i=\frac{D_i}{L^2}\Delta_y q_i+J R_i(q/J).
\]

Use P1 triangular finite elements, row-sum-lumped mass `M_l`, and reference stiffness `K`. For each step of size `h`, the implementation solves

\[
\left(M_l+h\frac{D_i}{L_{n+1}^2}K\right)q_i^{n+1}
=M_l\left[q_i^n+hJ_nR_i(c^n)\right].
\]

Finally `c_i^{n+1}=q_i^{n+1}/J_{n+1}`. Diffusion is implicit; reactions are explicit. This is **first order in time**. Growth updates use the exponential exactly, capped at `Lmax`. The stiffness and mass are assembled once on the reference mesh. Each species has a reusable matrix/solver; matrices are updated when the scalar diffusion coefficient changes.

FEniCSx CG starts from the previous solution. For multigrid, the preconditioner is reused while `alpha=h*D/L_new^2` stays within a factor 1.25 of its setup value; the system matrix itself always uses the current `alpha`. With `pc="auto"` (the current sphere presets), a repeated `alpha` switches to a cached sparse LU factorization, and a changed `alpha` switches back to CG/multigrid. Rejection, a changed step size, or renewed growth therefore cannot silently reuse an obsolete factorization. These changes accelerate linear algebra without altering the time discretization. Checkpoints save configuration and fields, not solver caches; restart reconstructs those caches and can differ within solve tolerance.

With reactions off, the discrete physical amount `1^T M_l q` is conserved to linear-solver tolerance. A spatially constant concentration therefore follows exactly the area dilution `c(t)=c(0)L(0)^2/L(t)^2`, up to roundoff and solve error. Reaction sources and logged depletion interventions legitimately change amounts.

Reaction RHS negativity, nonfinite values, or invalid resulting concentrations cause a trial step to be halved. The state is committed only after acceptance; no clipping is applied. Negative concentrations within `1e-12` roundoff tolerance are retained and reported. If halving reaches `min_dt`, the run fails visibly. Linear-solver nonconvergence is an error, not silently accepted.

**This rejection scheme is not an accuracy controller.** Positive solutions can still be temporally inaccurate or unstable. A reaction Jacobian-based step limiter, embedded error estimator, or fully implicit reaction integration may be necessary for later models. Every scientific comparison needs time-step refinement.

On the square's right-triangle mesh, mass lumping and the implicit diffusion operator support positivity. This reasoning must not be assumed for arbitrary distorted or obtuse surface meshes; the same acceptance checks remain active on the sphere. For the square, the independent backends may use opposite triangle diagonals and different boundary lumped weights. For the sphere they use the same triangles and agree to solve tolerance after coordinate remapping.

## Surface interventions

On the sphere, `perturb(x,y,radius,fraction)` interprets `x` as longitude divided by `2*pi` and `y` as colatitude divided by `pi`. The unit center is `(sin(pi*y)*cos(2*pi*x), sin(pi*y)*sin(2*pi*x), cos(pi*y))`. Let `theta=acos(clip(point dot center,-1,1))`. Activator is multiplied by `1-fraction*exp(-theta^2/(2*radius^2))`; inhibitor is unchanged. Width is angular distance in radians, giving physical arc width `L*radius`. Longitude is periodic, both poles are single mesh vertices, and the opposite hemisphere is far away along the surface. The default center `(0.5,0.5)` is on the negative x-axis. On the square the same inputs retain their original Cartesian interpretation.

Interventions log the normalized coordinates, width, fraction, time, and removed physical amount. Checkpoint configurations carry the geometry so events and `length` have an unambiguous interpretation on restart.

## Measurements and interpretation

Means and spatial standard deviations use reference lumped-mass weights. Physical species amount multiplies the weighted reference concentration integral by `L^2`. Reported `area` is `L^2*sum(mass)`, the actual triangulated surface area; it approaches `4*pi*L^2` on the sphere and equals `L^2` on the square. `balance_error` is the relative discrepancy between physical amount after the step and the amount predicted by the discrete reaction source. It excludes interventions between steps.

The displayed `sigma(u)` measures spatial variation. It is **not** a cancer score, a recovery-to-target error, a wavelength estimate, or proof of a Turing mechanism. A uniform state can be either healthy or abnormal in a biological model. Future recovery experiments need independently defined target states and separate growth-control and organization measurements.

## Deliberate limits

The surface geometry affects diffusion, but curvature does not feed back into reaction rates or growth. No cells, lineage transitions, force balance, boundaries determined by mechanics, topology changes, anisotropic growth, biological calibration, GPU execution, or distributed-memory solver are included. The sphere is prescribed and uniformly scaled; arbitrary imported surfaces and ellipsoids are not implemented. Add biological modules only after their conservation and coupling assumptions are explicit.

## Primary software references

- [DOLFINx 0.10 mesh creation, including embedded coordinates](https://docs.fenicsproject.org/dolfinx/v0.10.0/python/generated/dolfinx.mesh.html#dolfinx.mesh.create_mesh)
- [DOLFINx PETSc assembly and solver interfaces](https://docs.fenicsproject.org/dolfinx/v0.10.0.post2/python/generated/dolfinx.fem.petsc.html)
- [DOLFINx finite-element visualization](https://docs.fenicsproject.org/dolfinx/v0.10.0.post2/python/demos/demo_pyvista.html)
- [trame local and remote rendering](https://kitware.github.io/trame/guide/tutorial/vtk.html)
- [PETSc performance guidance](https://petsc.org/release/manual/performance/)
