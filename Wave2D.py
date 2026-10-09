import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""

    def __init__(self):
        self.L = 1

    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        xyij = np.linspace(0, 1, N + 1)
        xij, yij = np.meshgrid(xyij, xyij, indexing="ij", sparse=sparse)
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        D = sparse.diags([1., -2., 1.], [-1, 0, 1], (N + 1, N + 1), format="lil")
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        omega = self.c*np.sqrt((self.mx*np.pi)**2 + (self.my*np.pi)**2)
        return omega

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        t0 = 0
        xij, yij = self.create_mesh(N)
        ue = self.ue(mx=mx, my=my)

        u0 = sp.lambdify((t, x, y), ue)(t0, xij, yij)
        return u0

    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl*self.h/self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        # u is a 2d array of a given time point
        Un = u

        ue = self.ue(mx=self.mx, my=self.my)
        xij, yij = self.create_mesh(self.N)
        Un_e = sp.lambdify((t, x, y), ue)(t0, xij, yij)
        
        en_ij = Un - Un_e

        return np.sqrt(self.h**2 * np.sum(en_ij**2))

    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[0, :] = 0
        u[-1, :] = 0 
        u[:, 0] = 0
        u[:, -1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        # Must turn mx and my into explicit instance variables to be able to reach the ue method,
        # as they will not be passed to the l2_error method further down, that again relies on the ue method. 
        self.mx = mx
        self.my = my

        self.N = N

        dx = self.L/N
        self.h = dx

        # Need for the dt property
        self.cfl = cfl
        self.c = c

        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))

        u0 = self.initialize(N=N, mx=mx, my=my)
        Unm1[:] = u0
        self.apply_bcs(Unm1)

        D = self.D2(N)/dx**2
        
        Un[:] = Unm1 + 0.5*(c*self.dt)**2*(D @ Unm1 + Unm1 @ D.T)
        self.apply_bcs(Un)

        plotdata = {0: Unm1.copy()}
        if store_data == 1:
            plotdata[1] = Un.copy()

        for n in range(1, Nt):
            Unp1[:] = 2*Un - Unm1 + (c*self.dt)**2*(D @ Un + Un @ D.T)

            self.apply_bcs(Unp1)

            Unm1[:] = Un
            Un[:] = Unp1

            if n % store_data == 0:
                plotdata[n] = Unm1.copy()   # Unm1 is now swapped to Un

        if store_data > 0:
            return plotdata

        elif store_data == -1:
            final_time = Nt*self.dt
            l2_err = self.l2_error(Un, final_time)  # use current time point (remember loop is done)
            return (self.h, l2_err)

        else:
            raise ValueError("store_data should be an integer, either positive or -1")

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        D = sparse.diags([1., -2., 1.], [-1, 0, 1], (N + 1, N + 1), format="lil")
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self, u: np.ndarray):
        pass


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


# def test_exact_wave2d():
#     raise NotImplementedError("The test_exact_wave2d function is not implemented yet.")

