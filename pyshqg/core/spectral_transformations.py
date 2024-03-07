r"""Submodule dedicated to the Poisson solver."""

import numpy as np
import pyshtools as pysh

class SpectralTransformations:
    r"""Class for the spectral transformations.

    The goal of this class is to define the transformations
    $x \mapsto \hat{x} \mapsto x$ where $x$ is a variable
    in grid space and $\hat{x}$ is the spectral transform of $x$.

    The spectral space has shape $(2, T+1, T+1)$ where $T$ is
    the spectral truncature. The first index corresponds to the
    real/imaginary part, the second index corresponds to the spectral
    index $l$ and the second index corresponds to the spectral index
    $m\leq l$. With this latter constraint, the number of non-zero
    coefficients is $(T+1)(T+2)$.

    The grid space has shape $(T_{\mathsf{g}}+1, 2T_{\mathsf{g}}+2)$
    where $T_{\mathsf{g}}$ is the grid truncature. The first
    index corresponds to the latitude and the second to the longitude.

    The transformations implemented in this class only work when
    $T_{\mathsf{g}}\geq T$. Note that the number of degrees of freedom
    in spectral space is $(T+1)(T+2)$ while in grid space it is
    $2(T_{\mathsf{g}+1)^2$. This means that even in the case where 
    $T_{\mathsf{g}}=T$, there are more (approximately double) degrees
    of freedom in grid space. Consequently, the transformation from
    grid space to spectral space is the inverse of the transformation
    from spectral space to grid space only on the image of 
    the transformation from spectral to grid space.

    Attributes
    ----------
    backend : pyshqg.backend.Backend object
        The backend.
    T : int
        The internal spectral truncature (i.e. the data truncature in spectral space).
    T_grid : int
        The grid spectral truncature (i.e. the data truncature in grid space),
        which detrmines the size of the grid: `Nlat = T_grid+1` and `Nlon=2*Nlat`.
    planet_radius : float
        Planet radius.
    planet_omega : float
        Planet rotation speed.
    lats : np.ndarray, shape (Nlat,)
        Latitude nodes.
    lons : np.ndarray, shape (Nlon,)
        Longitude nodes.
    legendre : dict of str to backend array, shape (Nlat, T+1, T+1)
        Pre-computed Legendre coefficients for the spectral transformations.
    laplacian_spectrum : backend array, shape (T+1,)
        Pre-computed eigen spectrum of the Laplacian operator.
    qp : backend array, shape (Nlat, Nlon)
        Pre-computed planet vorticity in grid space.
    """
    
    def __init__(
        self,
        backend,
        T,
        T_grid,
        planet_radius=1,
        planet_omega=1,
    ):
        r"""Constructor for the spectral transformations.

        Parameters
        ----------
        backend : pyshqg.backend.Backend object
            The backend.
        T : int
            The internal spectral truncature.
        T_grid : int
            The grid spectral truncature.
        planet_radius : float
            Planet radius.
        planet_omega : float
            Planet rotation speed.
        """
        self.backend = backend
        self.T = T
        self.T_grid = T_grid
        self.planet_radius = planet_radius
        self.planet_omega = planet_omega

        self.lats, self.lons = self.precompute_grid_nodes()
        self.legendre = self.precompute_legendre_coefficients()
        self.laplacian_spectrum = self.precompute_laplacian_spectrum()
        self.qp = self.precompute_planet_vorticity()

    def precompute_grid_nodes(self):
        r"""Pre-computes the grid nodes.

        Returns
        -------
        lats, lons : np.ndarray, shape (Nlat,) and (Nlon,)
            Latitude and longitude nodes in degrees.
        """
        lats, _ = pysh.expand.GLQGridCoord(self.T_grid)
        lats = lats[::-1]
        lons = np.mod(np.linspace(
            0,
            360,
            2*self.T_grid+2,
            endpoint=False,
        ) - 180, 360)
        return lats, lons

    def precompute_legendre_coefficients(self):
        r"""Pre-computes the Legendre coefficients.

        Several variants are computed:

        - `'PLM'`: for the standard transformation from spectral space to grid space;
        - `'PPLM'`: for the transformation from spectral space to grid space with derivative with respect to longitude;
        - `'ALM'`: for the transformation from spectral space to grid space with derivative with respect to latitude;
        - `'PW'`: for the transformation from grid space to spectral space.

        Note that all these coefficients are pre-computed
        using `pyshtools`.

        Returns
        -------
        legendre : dict of str to backend array, shape (Nlat, T+1, T+1)
            Pre-computed Legendre coefficients for the spectral transformations.
        """
        # Gauss--Legendre weights
        cost, w = pysh.expand.SHGLQ(self.T_grid)
        cosl = np.cos(self.lats*np.pi/180)

        # pre-compute the coefficients for the spectral transformations
        PLM = np.zeros((self.T_grid+1, self.T+1, self.T+1))
        PPLM = np.zeros((self.T_grid+1, self.T+1, self.T+1))
        ALM = np.zeros((self.T_grid+1, self.T+1, self.T+1))
        PW = np.zeros((self.T_grid+1, self.T+1, self.T+1))
        for i in range(self.T_grid+1):
            p, a = pysh.legendre.PlmBar_d1(self.T, cost[i], cnorm=1, csphase=1)
            for l in range(self.T+1):
                for m in range(l+1):
                    ind = ( l * (l+1) ) // 2 + m
                    PLM[i, m, l] = p[ind]
                    PPLM[i, m, l] = p[ind] / (cosl[i] * self.planet_radius)
                    ALM[i, m, l] = a[ind] * cosl[i] / self.planet_radius
                    PW[i, m, l] = 0.5 * p[ind] * w[i]

        return dict(
            PLM=self.backend.from_numpy(PLM),
            PPLM=self.backend.from_numpy(PPLM),
            ALM=self.backend.from_numpy(ALM),
            PW=self.backend.from_numpy(PW),
        )

    def precompute_laplacian_spectrum(self):
        r"""Pre-computes the Laplacian eigen spectrum.

        Returns
        -------
        laplacian_spectrum : backend array, shape (T+1,)
           Pre-computed eigen spectrum of the Laplacian operator. 
        """
        laplacian_spectrum = np.zeros(self.T+1)
        for l in range(self.T+1):
            laplacian_spectrum[l] = -l*(l+1) / self.planet_radius**2
        return self.backend.from_numpy(laplacian_spectrum)

    def precompute_planet_vorticity(self):
        r"""Pre-computes the planet vorticity $q^{\mathsf{p}}$.
        
        Using the $\beta$-plane approximation, the
        planet vorticity $q^{\mathsf{p}}$ is defined as...

        This method is used by the Poisson solver
        to define the transformation

        Returns
        -------
        qp : backend array, shape (Nlat, Nlon)
            Planet vorticity $q^{\mathsf{p}}$ in grid space.
        """
        spec_qp = np.zeros((2, self.T+1, self.T+1))
        spec_qp[0, 1, 0] = 2 * self.planet_omega / np.sqrt(3)
        spec_qp = self.backend.from_numpy(spec_qp)
        return self.spec_to_grid(spec_qp)
        
    def spec_to_grid_generic(self, spec_x, plm):
        r"""Applies the generic transformation from spectral space to grid space.

        Use `plm=PLM` for a regular spectral to grid transformation,
        `plm=PPLM` for a transformation with gradients with respect
        to longitude, and `plm=ALM` for a transformation with gradients 
        with respect to latitude.

        Parameters
        ----------
        spec_x : backend array, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.
        plm : backend array, shape (Nlat, T+1, T+1)
            One of `PLM`, `PPLM`, `ALM`.

        Returns
        -------
        x : backend array, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        leg_x = self.backend.einsum('...jm,imj->...im', spec_x, plm)
        return self.backend.apply_fft(self.T, self.T_grid, leg_x)
        
    def spec_to_grid(self, spec_x):
        r"""Applies the regular transformation from spectral space to grid space.

        The `spec_to_grid_generic` method is used with
        `plm=PLM` to implement this transformation.

        Parameters
        ----------
        spec_x : backend array, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.

        Returns
        -------
        x : backend array, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        return self.spec_to_grid_generic(spec_x, self.legendre['PLM'])
    
    def spec_to_grid_grad_theta(self, spec_x):
        r"""Applies the transformation from spectral space to grid space
        with gradients with respect to latitude ($\theta$).

        The `spec_to_grid_generic` method is used with
        `plm=ALM` to implement this transformation.

        Parameters
        ----------
        spec_x : backend array, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.

        Returns
        -------
        x : backend array, shape (..., Nlat, Nlon)
            Gradient of variable $x$ with respect to latitude in grid space.
        """
        return self.spec_to_grid_generic(spec_x, self.legendre['ALM'])
    
    def spec_to_grid_grad_phi(self, spec_x):
        r"""Applies the transformation from spectral space to grid space
        with gradients with respect to longitude ($\phi$).

        The `spec_to_grid_generic` method is used with
        `plm=PPLM` to implement this transformation.

        Parameters
        ----------
        spec_x : backend array, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.

        Returns
        -------
        x : backend array, shape (..., Nlat, Nlon)
            Gradient of variable $x$ with respect to longitude in grid space.
        """
        spec_dx_real = - self.backend.range(self.T+1) * spec_x[..., 1, :, :]
        spec_dx_real = self.backend.expand_dims(spec_dx_real, axis=-3)
        spec_dx_imag = self.backend.range(self.T+1) * spec_x[..., 0, :, :]
        spec_dx_imag = self.backend.expand_dims(spec_dx_imag, axis=-3)
        spec_dx = self.backend.concatenate([spec_dx_real, spec_dx_imag], axis=-3)
        return self.spec_to_grid_generic(spec_dx, self.legendre['PPLM'])

    def grid_to_spec(self, x):
        r"""Applies the transformation from grid space to spectral space.

        Parameters
        ----------
        x : backend array, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.

        Returns
        -------
        spec_x : backend array, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.
        """
        leg_x = self.backend.apply_ifft(self.T, self.T_grid, x)
        return self.backend.einsum('...im,iml->...lm', leg_x, self.legendre['PW'])

