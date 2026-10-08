import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float):
        self.L = L
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        D2 = self.p.D2(N=N, dx=self.L/N)
        # We do not strictly have to distinguish between D2x and D2y since N and L is the same for both x and y direction,
        # however it makes the code a bit more readable mathematically to have D2x and D2y.
        D2x = D2
        D2y = D2

        A = (sparse.kron(D2x, sparse.eye(N+1)) + sparse.kron(sparse.eye(N+1), D2y))
        return A

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector

        Note
        ----
        Compute the Kronecker product of the 1D Laplace operator with itself
        to create the 2D Laplace operator. Then, assemble the right-hand side
        vector b by evaluating the function f at the mesh points and applying
        Dirichlet boundary conditions using the exact solution ue.

        Note to note: We calculate the 2d Laplacian matrix in function laplace

        """
        # Create vectorized 2D Laplace matrix
        A = self.laplace(N)

        # Find the indices in the flatten u matrix that corresponds to the boundaries 
        bnds = self.get_boundary_indices(N)
        # Zero out every row A that has diagonal element corresponding to boundary in u
        A = A.tolil()
        for i in bnds:
            A[i, :] = 0
            A[i, i] = 1

        xij, yij = self.create_mesh(N)

        # Turn the sympy expression f into the value matrix F. (Could have used ue to find f, but here f is supplied)
        F = self.meshfunction(f, xij, yij)
        # Turn the sympy expression ue into the value matrix Ue
        Ue = self.meshfunction(ue, xij, yij)

        # Flatten F and Ue
        b = F.ravel()
        Ue_flat = Ue.ravel()

        # This is MMS at work where we pick a ue based on what boundary conditions we want (can be whatever we want),
        # substituting the values of Ue_flat that corresponds to the boundary points into the same spot in the b vector.
        b[bnds] = Ue_flat[bnds]
        
        return A.tocsr(), b

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """
        # Turn function u(x, y) into matrix with the grid values
        U = sp.lambdify((x, y), u)(xij, yij)
        return U

    def get_boundary_indices(self, N: int) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((N+1, N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        return bnds

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error

        """
        N = len(u) - 1
        xij, yij = self.create_mesh(N)
        Ue = self.meshfunction(ue, xij, yij)
        dx = self.L/N
        dy = self.L/N
        return np.sqrt(dx*dy*np.sum((u - Ue)**2))

    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(N, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, U: np.ndarray, x: float, y: float) -> float:
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """        
        from math import floor, ceil

        Nx = U.shape[0] - 1
        Ny = U.shape[1] - 1

        dx = self.L/Nx
        dy = self.L/Ny

        # Find the index (x,y) would have had given current grid step size
        i_in = x/dx
        j_in = y/dy

        # Find the index number of the grid points closest to, and, surrounding (x,y) 
        i_down1 = floor(i_in)
        i_up1 = ceil(i_in)
        j_down1 = floor(j_in)
        j_up1 = ceil(j_in)

        # Weight based on proximity to grid point
        i_down_weight = 1.0 - (x - i_down1*dx)/dx
        i_up_weight = 1.0 - i_down_weight
        j_down_weight = 1.0 - (y - j_down1*dy)/dy
        j_up_weight = 1.0 - j_down_weight
        
        U_interpolated = (U[i_down1, j_down1]*i_down_weight*j_down_weight
                      + U[i_down1, j_up1]*i_down_weight*j_up_weight
                      + U[i_up1, j_down1]*i_up_weight*j_down_weight
                      + U[i_up1, j_up1]*i_up_weight*j_up_weight)

        return U_interpolated


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h, y: 1 - h / 2}).n()) < 1e-3


if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")
