# Validation record

## Solver acceleration — 2026-09-13

Profiling the original `n=64` growing sphere attributed about 98% of measured step time to diffusion solves. The optimized FEniCSx backend starts CG from its previous solution, reuses the GAMG preconditioner while the diffusion multiplier stays within a factor 1.25 of its setup value, and provides `pc="auto"` to switch to a reused sparse LU factorization when the multiplier repeats. A change in the multiplier switches back to CG/GAMG; the current system matrix is always used. The three sphere presets now select automatic mode. Geometry, time steps, and solver tolerances were unchanged.

For 200 measured warm steps on this workstation, the original Jacobi solver took 25.31 ms/step during growth and 15.04 ms/step at fixed radius 6. Automatic mode took 15.72 and 4.35 ms/step, respectively. These measurements use one BLAS/OpenMP thread, include Python profiling overhead, and exclude rendering and disk recording. They exclude the first solver setup but include automatic mode's subsequent LU setup. Optimized Jacobi alone took 18.72 and 9.72 ms/step; GAMG without the automatic LU switch took 15.54 and 13.08 ms/step. Multigrid alone was therefore not the fastest fixed-domain choice in this environment.

The complete growing-sphere run at `n=64`, `dt=0.02`, and final time 180 completed 9,000 steps with no rejections. Solver compute time fell from 153.93 s to 53.15 s (2.90× faster); the new batch loop including recording took 54.70 s. The final maximum absolute concentration difference from the original Jacobi trajectory was `2.02e-7`, and final `std_u` was 0.739147516. The final discrete balance residual was `5.12e-12`. Process peak RSS was approximately 182 MiB for the automatic run, compared with 139 MiB reported for the earlier Jacobi run. LU memory can become a limiting factor on larger meshes; these timings are machine- and configuration-specific.

The numerical/worker suite passed **28 tests**, with two browser cases skipped in that invocation. A subsequent full run with the FEniCSx browser tests enabled passed **all 30 tests**, including square and automatic-mode sphere controls. Added tests compare reused/rebuilt multigrid and automatic LU solves to independent SciPy sparse LU across repeated multipliers, abrupt changes, and zero diffusion, and verify restart after growth and depletion. Full-run and warm benchmark records are in [solver-speedup.json](solver-speedup.json). The earlier pattern images and provenance below were generated before these linear-solver optimizations.

## Finer mesh and pattern presets — 2026-09-13

The growing sphere was refined from `n=24` to `n=64`; the user-selected final time 180 was retained. The mesh now has 32,768 triangles, 16,386 vertices, and 32,772 scalar unknowns. The longest physical edge is approximately 0.230 at radius 6. No solver equations or numerical tolerances were changed.

Three complete FEniCSx runs used the configurations below with `dt=0.02`, `seed=42`, and `n=64`. Each reached time 180 in 9,000 accepted steps with zero rejected steps:

| Configuration | Final radius | Final `std_u` | Final minimum u / v | Final balance residual |
|---|---:|---:|---:|---:|
| `sphere.json` | 6 | 0.739148 | 0.242552 / 0.416564 | 3.24e-12 |
| `sphere_spots.json` | 12 | 0.846711 | 0.177591 / 0.349126 | 8.77e-13 |
| `sphere_stripes.json` | 12 | 0.610174 | 0.442266 / 0.373618 | 3.03e-13 |

The two pattern controls hold radius fixed and use the reaction parameters documented in the README. Visual inspection of front/back projections showed distinct spots in the spot case and connected stripe-like bands with some spot-like defects in the stripe case. [pattern-presets.png](pattern-presets.png) displays the final activator fields with separate color scales per row; inverse-distance interpolation of the three nearest surface nodes was used only for the display. [pattern-validation.json](pattern-validation.json) retains resolved configurations, software provenance, and final diagnostics.

These full-run checks verify operation at the new resolution and demonstrate the selected morphologies for seed 42. They do not establish time-step convergence, seed robustness, or unique pattern selection. The earlier automated numerical and browser checks below were not rerun for this configuration/documentation-only update.

## Spherical surface extension — 2026-09-13

The numerical and worker suite passed **24 tests**, with the two opt-in browser cases skipped. This includes both installed backends, FEniCSx and SciPy. Added checks cover closed outward-oriented sphere topology, uniform area dilution, nonuniform passive amount conservation through the growth cap, surface area convergence, decay of the degree-one spherical harmonic (`Delta_S z = -2z`), localized great-circle depletion, longitude periodicity, spherical checkpoint/intervention restart, legacy square checkpoints, and cross-backend surface trajectories. The two browser cases (square and sphere) also passed with FEniCSx workers and the installed Chrome binary, including run/pause, perturbation, growth rate, save, and reset controls.

The original `configs/sphere.json` batch run, before the later resolution and final-time changes, completed with FEniCSx at `n=24` (4,612 scalar unknowns), `dt=0.02`, and final time 80:

| Measurement | Result |
|---|---:|
| Initial / final radius | 3 / 6 |
| Accepted / rejected steps | 4,000 / 0 |
| Final triangulated surface area | 451.729461 |
| Final activator spatial standard deviation | 0.684773 |
| Final minimum activator / inhibitor | 0.157978 / 0.443700 |
| Final discrete balance residual | 1.44e-12 |

The smooth sphere's area at radius 6 is about 452.389342; the reported area integrates the actual planar triangles. Checkpoint-to-VTU export of the final spherical state completed successfully. These results establish a working example and specific numerical checks, not convergence of its selected pattern or onset time.

The numerical suite and batch run used the existing `tissue-growth` conda environment with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and `XDG_CACHE_HOME=/tmp/tissue-growth-cache`. MPI required execution outside the command sandbox. Native GLX rendering failed to create a context, and an EGL attempt failed in the local graphics driver. The working browser command used the already-installed virtual display and Chrome:

```bash
LIBGL_ALWAYS_SOFTWARE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkXOpenGLRenderWindow \
XDG_CACHE_HOME=/tmp/tissue-growth-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
TISSUE_BROWSER_TEST=1 TISSUE_TEST_BACKEND=fenicsx \
TISSUE_CHROMIUM=/opt/google/chrome/chrome \
xvfb-run -a python -m pytest -q -m browser
```

The spherical dashboard screenshot was inspected for the visible surface, radius metric, and angular depletion controls. No macOS validation was performed for this extension. The records below describe the earlier square implementation and retain their original scope.

## Original square validation — 2026-09-12

## Executed checks

| Check | Result |
|---|---|
| Numerical and worker suite with FEniCSx installed | **13 passed**, optional browser test skipped |
| Real Chromium browser test using the FEniCSx worker | **1 passed** |
| Numerical and worker suite without FEniCSx | **7 passed**, FEniCSx comparison and opt-in browser test skipped |
| Real Chromium browser test using the SciPy worker | **1 passed** |
| Long quick-demo run, FEniCSx | 4,000 accepted steps, no rejected steps |
| Long quick-demo run, SciPy | 4,000 accepted steps, no rejected steps |
| Checkpoint-to-VTU export | Completed successfully |

The numerical tests check:

1. Conserved physical amount and exact area dilution for homogeneous passive signals, including reaching the growth cap.
2. Homogeneous reaction dynamics against an independent high-accuracy `solve_ivp` solution.
3. Mesh convergence for a known Neumann cosine diffusion mode, at 8, 16, and 32 subdivisions per side.
4. Restart equivalence after growth-rate and depletion interventions.
5. Rejection of overly large reaction steps without concentration clipping.
6. Invalid parameter rejection.
7. Agreement of the independently assembled stiffness operators after coordinate remapping.
8. Separate-process run/pause controls, intervention logging, checkpoint saving, and shutdown.

The browser check launches the actual application and a Chromium headless browser. It verifies a visible canvas; local depletion; run and pause; no time advancement after pause; growth-rate changes; checkpoint creation; and reset to time zero. It also checks for JavaScript errors and visible application errors. `dashboard.png` is a screenshot of this actual FEniCSx test, not a mockup.

## Long example

Configuration: `configs/quick_demo.json`, FEniCSx, `n=40`, `dt=0.02`, final simulated time 80. There are 3,362 scalar unknowns.

- Initial side length: 6; final side length: 12.
- Accepted steps: 4,000; rejected steps: 0.
- Final activator spatial standard deviation: approximately 0.680791.
- Final minimum concentrations: approximately 0.229175 (`u`) and 0.442881 (`v`).
- Final relative discrete balance residual: approximately `1.71e-12`.
- Solver setup: approximately 0.42 seconds.
- Batch wall time after setup, including recording: approximately 6.87 seconds.

These are results from the development Linux environment. They are **not MacBook or NVIDIA-desktop benchmarks**, nor evidence of biological calibration. The independent reference run reached a very similar final amplitude (approximately 0.680790), but equality of nonlinear patterns is not required across different meshes or parameter regimes.

Machine-readable files: `fenicsx-demo-summary.json`, `fenicsx-demo-provenance.json`, and `reference-demo-summary.json`. The reference run preceded the Linux `/proc/self` memory-reporting fix; its invalid early PID-based current-RSS sample is explicitly recorded as null.

## Performance samples

The supplied benchmark JSON files used 30 warm steps at each size, one CPU process, and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. They measure different runs with shared host activity; **do not interpret their ratio as a controlled backend speed comparison**. Matrices change during growth. Initial form compilation and the first solve are outside the warm timer, and visualization/checkpoint output is excluded.

| FEniCSx subdivisions per side | Scalar DOFs | Warm milliseconds per step, approximately |
|---|---:|---:|
| 32 | 2,178 | 1.67 |
| 64 | 8,450 | 12.05 |
| 128 | 33,282 | 79.78 |

These short samples validate the benchmark command and provide a starting point for your own measurements. Longer runs, preconditioner tuning, mesh convergence, and representative coupled biology are needed for capacity planning. One early reference benchmark with unrestricted BLAS threading was interrupted because of poor throughput; this motivated the documented single-thread comparison.

## Environment details and limits

Core validated versions: Python 3.12.13, DOLFINx/Basix 0.10.0, FFCx 0.10.1, UFL 2025.2.1, PETSc 3.25.5, PyVista 0.49.0, VTK 9.7.0, trame 3.13.2, trame-vtk 2.11.16, and trame-vuetify 3.2.6. Browser rendering was exercised with Chromium 134 headless.

The restricted Linux execution environment needed two runtime adjustments for validation: `UCX_TLS=self` for this serial MPI worker, and preloading the system GNU OpenMP runtime to avoid an assertion in the available LLVM OpenMP build. The FEniCSx commands used:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 \
UCX_TLS=self OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -q
```

**These are Linux test-environment adjustments, not Mac installation instructions.** The default Mac setup is in the README. Earlier attempts to repair the first test environment caused incompatible dependency versions; that environment was discarded, and the reported FEniCSx validation used a fresh compatible 0.10 environment.

No macOS/Apple Silicon execution, GPU acceleration, MPI domain decomposition, remote desktop deployment, multiuser authentication, or checkpoint transfer between operating systems was tested. The code explicitly limits this release to one MPI rank. Live camera interaction was visually inspected; there is no automated pixel-level validation of scientific values. Numeric values are validated through the solver tests.

The model remains a two-species reaction–diffusion testbed on a prescribed growing square. Developmental lineage regulation, mechanics, and oncogenic perturbations are future model extensions. Passing the tests is not a substitute for convergence studies of each research experiment.
