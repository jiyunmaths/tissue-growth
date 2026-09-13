"""P1 lumped-mass diffusion on reference surfaces. Serial workers only."""
import numpy as np
from .geometry import sphere_mesh, square_mesh


class ScipyOperators:
    """Explicit reference backend; useful for tests and installations without PETSc."""
    def __init__(self, config):
        from scipy.sparse import coo_matrix
        self.points, self.triangles = (sphere_mesh(config.n) if config.geometry == "sphere"
                                       else square_mesh(config.n))
        xyz = self.points[self.triangles]
        normal = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
        twice_area = np.linalg.norm(normal, axis=1)
        area = twice_area / 2
        # Tangential P1 gradients on each triangle embedded in R^3.
        opposite_edges = np.stack([xyz[:, 2]-xyz[:, 1], xyz[:, 0]-xyz[:, 2],
                                   xyz[:, 1]-xyz[:, 0]], axis=1)
        grad = np.cross(normal[:, None, :], opposite_edges) / twice_area[:, None, None]**2
        ke = area[:,None,None] * np.einsum("tik,tjk->tij", grad, grad)
        rows = np.repeat(self.triangles, 3, axis=1).ravel()
        cols = np.tile(self.triangles, (1, 3)).ravel()
        size = len(self.points)
        self.stiffness = coo_matrix((ke.ravel(), (rows, cols)), shape=(size, size)).tocsr()
        self.mass = np.bincount(self.triangles.ravel(), weights=np.repeat(area/3, 3), minlength=size)
        self.rtol = config.ksp_rtol
        self._cache = {}
        self.iterations = 0

    def solve(self, rhs, alpha, field):
        from scipy.sparse import diags
        from scipy.sparse.linalg import cg, LinearOperator
        if field not in self._cache or self._cache[field][0] != alpha:
            matrix = diags(self.mass) + alpha * self.stiffness
            inv = 1 / matrix.diagonal()
            self._cache[field] = (alpha, matrix, LinearOperator(matrix.shape, matvec=lambda x: inv*x))
        _, matrix, pre = self._cache[field]
        self.iterations = 0
        def count(_):
            self.iterations += 1
        solution, info = cg(matrix, rhs, rtol=self.rtol, atol=0, M=pre, callback=count)
        if info != 0:
            raise RuntimeError(f"SciPy CG failed: info={info}")
        return solution

    def close(self):
        self._cache.clear()


class FenicsOperators:
    def __init__(self, config):
        try:
            import dolfinx
            from dolfinx import fem, mesh, plot
            from dolfinx.fem import petsc
            from mpi4py import MPI
            from petsc4py import PETSc
            import ufl
        except ImportError as exc:
            raise RuntimeError("FEniCSx is unavailable. Install environment.yml, or explicitly select --backend scipy for the reference backend.") from exc
        if MPI.COMM_WORLD.size != 1:
            raise RuntimeError("This prototype supports one MPI rank per worker. Do not launch it with mpirun.")
        if np.issubdtype(PETSc.ScalarType, np.complexfloating):
            raise RuntimeError("Use a real-valued PETSc build for this model")
        self.PETSc = PETSc
        if config.geometry == "sphere":
            from basix.ufl import element
            points, triangles = sphere_mesh(config.n)
            coordinate_element = element("Lagrange", "triangle", 1, shape=(3,))
            self.mesh = mesh.create_mesh(MPI.COMM_SELF, triangles, coordinate_element, points)
        else:
            self.mesh = mesh.create_unit_square(MPI.COMM_SELF, config.n, config.n, cell_type=mesh.CellType.triangle)
        self.space = fem.functionspace(self.mesh, ("Lagrange", 1))
        trial, test = ufl.TrialFunction(self.space), ufl.TestFunction(self.space)
        self.stiffness = petsc.assemble_matrix(fem.form(ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx))
        self.stiffness.assemble()
        self.mass_vec = petsc.assemble_vector(fem.form(test * ufl.dx))
        self.mass = self.mass_vec.array.copy()
        cells, types, points = plot.vtk_mesh(self.space)
        if not np.all(types == 5):
            raise RuntimeError("Expected P1 triangles")
        self.points = points.copy()
        self.triangles = cells.reshape(-1, 4)[:, 1:].copy()
        self._cache = {}
        self.config = config
        self.iterations = 0

    def solve(self, rhs, alpha, field):
        P = self.PETSc
        if field not in self._cache:
            matrix = self.stiffness.copy()
            b, x = self.mass_vec.duplicate(), self.mass_vec.duplicate()
            x.set(0)
            ksp = P.KSP().create(self.mesh.comm)
            ksp.setOptionsPrefix(f"tissue_{field}_")
            ksp.setType("cg")
            # Successive amount fields change little; retain the last solution
            # as the next initial guess. Explicit PETSc options can override it.
            ksp.setInitialGuessNonzero(True)
            ksp.getPC().setType("gamg" if self.config.pc == "auto" else self.config.pc)
            ksp.setTolerances(rtol=self.config.ksp_rtol, atol=1e-14, max_it=10000)
            ksp.setFromOptions()
            self._cache[field] = [None, matrix, b, x, ksp, None]
        entry = self._cache[field]
        previous, matrix, b, x, ksp, pc_alpha = entry
        if self.config.pc == "auto":
            # A repeated multiplier means identical h*D/L^2. Factor once in
            # fixed-domain segments; return to iterative solves if it changes.
            target = "lu" if previous == alpha else "gamg"
            if ksp.getPC().getType() != target:
                ksp.getPC().setReusePreconditioner(False)
                ksp.getPC().setType(target)
                ksp.setType("preonly" if target == "lu" else "cg")
                ksp.setInitialGuessNonzero(target != "lu")
                entry[5] = pc_alpha = None
        if previous != alpha:
            matrix.zeroEntries()
            matrix.axpy(alpha, self.stiffness, structure=P.Mat.Structure.SAME_NONZERO_PATTERN)
            matrix.setDiagonal(self.mass_vec, addv=P.InsertMode.ADD_VALUES)
            matrix.assemble()
            ksp.setOperators(matrix)
            if ksp.getPC().getType() == "gamg":
                # Always solve with the CURRENT matrix. Only lag the costly
                # multigrid preconditioner while alpha stays within a factor
                # 1.25 of its setup value. Large growth/step changes rebuild it.
                reuse = pc_alpha is not None and (
                    alpha == pc_alpha or (pc_alpha > 0 and 0.8 <= alpha/pc_alpha <= 1.25))
                ksp.getPC().setReusePreconditioner(reuse)
                if not reuse:
                    entry[5] = alpha
            entry[0] = alpha
        b.array[:] = rhs
        ksp.solve(b, x)
        self.iterations = ksp.getIterationNumber()
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"PETSc CG failed: reason={ksp.getConvergedReason()}")
        return x.array.copy()

    def close(self):
        for _, matrix, b, x, ksp, _ in self._cache.values():
            ksp.destroy()
            matrix.destroy()
            b.destroy()
            x.destroy()
        self._cache.clear()
        self.mass_vec.destroy()
        self.stiffness.destroy()


def make_operators(config):
    return FenicsOperators(config) if config.backend == "fenicsx" else ScipyOperators(config)
