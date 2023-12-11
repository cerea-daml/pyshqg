r"""Submodule dedicated to the Poisson solver."""

import numpy as np
import pyshtools as pysh
import scipy.fftpack

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
    T : int
        The internal spectral truncature (i.e. the data truncature in spectral space).
    T_grid : int
        The grid spectral truncature (i.e. the data truncature in grid space),
        which detrmines the size of the grid: `Nlat = T_grid+1` and `Nlon=2*Nlat`.
    PLM : np.ndarray, shape (Nlat, T+1, T+1)
        Coefficients for the transformation from spectral space to grid space.
    PPLM : np.ndarray, shape (Nlat, T+1, T+1)
        Coefficients for the transformation from spectral space to grid space
        with derivative with respect to longitude.
    ALM : np.ndarray, shape (Nlat, T+1, T+1)
        Coefficients for the transformation from spectral space to grid space
        with derivative with respect to latitude.
    PW : np.ndarray, shape (Nlat, T+1, T+1)
        Coefficients for the transformation from grid space to spectral space.
    laplacian_spectrum : np.ndarray, shape (T+1,)
        Eigen spectrum of the Laplacian operator.
    planet_radius : float
        Planet radius.
    planet_omega : float
        Planet rotation speed.
    lats : np.ndarray, shape (Nlat,)
        Latitude nodes.
    lons : np.ndarray, shape (Nlon,)
        Longitude nodes.
    """
    
    def __init__(
        self,
        T,
        T_grid,
        planet_radius=1,
        planet_omega=1,
    ):
        r"""Constructor for the spectral transformations.

        All the coefficients for the spectral transformations are pre-computed
        using `pyshtools`.

        Parameters
        ----------
        T : int
            The internal spectral truncature.
        T_grid : int
            The grid spectral truncature.
        planet_radius : float
            Planet radius.
        planet_omega : float
            Planet rotation speed.
        """
        self.T = T
        self.T_grid = T_grid
        self.planet_radius = planet_radius
        self.planet_omega = planet_omega

        # latitude and longitude nodes
        lats, _ = pysh.expand.GLQGridCoord(T_grid)
        self.lats = lats[::-1]
        self.lons = np.mod(np.linspace(
            0,
            360,
            2*T_grid+2,
            endpoint=False,
        ) - 180, 360)

        # Gauss--Legendre weights
        cost, w = pysh.expand.SHGLQ(T_grid)
        cosl = np.cos(lats*np.pi/180)

        # pre-compute the coefficients for the spectral transformations
        PLM = np.zeros((T_grid+1, T+1, T+1))
        PPLM = np.zeros((T_grid+1, T+1, T+1))
        ALM = np.zeros((T_grid+1, T+1, T+1))
        PW = np.zeros((T_grid+1, T+1, T+1))
        for i in range(T_grid+1):
            p, a = pysh.legendre.PlmBar_d1(T, cost[i], cnorm=1, csphase=1)
            for l in range(T+1):
                for m in range(l+1):
                    ind = ( l * (l+1) ) // 2 + m
                    PLM[i, m, l] = p[ind]
                    PPLM[i, m, l] = p[ind] / (cosl[i] * planet_radius)
                    ALM[i, m, l] = a[ind] * cosl[i] / planet_radius
                    PW[i, m, l] = 0.5 * p[ind] * w[i]
        self.PLM = PLM
        self.PPLM = PPLM
        self.ALM = ALM
        self.PW = PW

        # pre-compute the Eigen spectrum of the Laplacian operator
        laplacian_spectrum = np.zeros(T+1)
        for l in range(T+1):
            laplacian_spectrum[l] = -l*(l+1) / planet_radius**2
        self.laplacian_spectrum = laplacian_spectrum

        # select appropriate versions for the DFT
        self.apply_fft = self.apply_fft_v2
        self.apply_ifft = self.apply_ifft_v2

    def apply_fft_v1(self, leg_x):
        r"""Applies the forward DFT, first version.

        This method relies on `numpy.fft.fft`.

        Parameters
        ----------
        leg_x : np.ndarray, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.

        Returns
        -------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        shape = list(leg_x.shape)
        batch_shape = shape[:-3]
        augm_leg_x = np.zeros(
            tuple(batch_shape+[self.T_grid+1, 2*self.T_grid+2]),
            dtype=np.complex128,
        )
        augm_leg_x[..., :self.T+1] = leg_x[..., 0, :, :] + 1j*leg_x[..., 1, :, :]
        augm_leg_x[..., -self.T:] = leg_x[..., 0, :, 1:][..., ::-1] - 1j*leg_x[..., 1, :, 1:][..., ::-1]
        return np.fft.fft(augm_leg_x, axis=-1).real

    def apply_fft_v2(self, leg_x):
        r"""Applies the forward DFT, second version.

        This method relies on `numpy.fft.hfft`.

        Parameters
        ----------
        leg_x : np.ndarray, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.

        Returns
        -------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        shape = list(leg_x.shape)
        batch_shape = shape[:-3]
        augm_leg_x = np.zeros(
            tuple(batch_shape+[self.T_grid+1, self.T_grid+2]),
            dtype=np.complex128,
        )
        augm_leg_x[..., :self.T+1] = leg_x[..., 0, :, :] + 1j*leg_x[..., 1, :, :]
        return np.fft.hfft(augm_leg_x, axis=-1)

    def apply_fft_v3(self, leg_x):
        r"""Applies the forward DFT, third version.

        This method relies on `scipy.fftpack.dct`
        and `scipy.fftpack.dst`.

        Parameters
        ----------
        leg_x : np.ndarray, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.

        Returns
        -------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        shape = list(leg_x.shape)
        batch_shape = shape[:-3]
        augm_real_leg_x = np.zeros(tuple(batch_shape+[self.T_grid+1, 2*self.T_grid+3]))
        augm_real_leg_x[..., ::2][..., :self.T+1] = leg_x[..., 0, :, :]
        augm_real_leg_x[..., :, 0] = leg_x[..., 0, :, 0]
        x_real = scipy.fftpack.dct(augm_real_leg_x, type=1, axis=-1)[..., :-1]        
        augm_imag_leg_x = np.zeros(tuple(batch_shape+[self.T_grid+1, 2*self.T_grid+1]))
        augm_imag_leg_x[..., 1::2][..., :self.T] = leg_x[..., 1, :, 1:]
        x_imag = scipy.fftpack.dst(augm_imag_leg_x, type=1, axis=-1)
        x_real[..., 1:] += x_imag
        return x_real

    def apply_ifft_v1(self, x):
        r"""Applies the inverse DFT, first version.

        This method relies on `numpy.fft.ifft`.

        Parameters
        ----------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.

        Returns
        -------
        leg_x : np.ndarray, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.
        """
        leg_x = np.fft.ifft(x, axis=-1)[..., :self.T+1]
        return np.stack([leg_x.real, leg_x.imag], axis=-3)

    def apply_ifft_v2(self, x):
        r"""Applies the inverse DFT, second version.

        This method relies on `numpy.fft.ihfft`.

        Parameters
        ----------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.

        Returns
        -------
        leg_x : np.ndarray, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.
        """
        leg_x = np.fft.ihfft(x, axis=-1)[..., :self.T+1]
        return np.stack([leg_x.real, leg_x.imag], axis=-3)

    def spec_to_grid_generic(self, spec_x, plm):
        r"""Applies the generic transformation from spectral space to grid space.

        Use `plm=PLM` for a regular spectral to grid transformation,
        `plm=PPLM` for a transformation with gradients with respect
        to longitude, and `plm=ALM` for a transformation with gradients 
        with respect to latitude.

        Parameters
        ----------
        spec_x : np.ndarray, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.
        plm : np.ndarray, shape (Nlat, T+1, T+1)
            One of `PLM`, `PPLM`, `ALM`.

        Returns
        -------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        leg_x = np.einsum('...jm,imj->...im', spec_x, plm)
        return self.apply_fft(leg_x)
        
    def spec_to_grid(self, spec_x):
        r"""Applies the regular transformation from spectral space to grid space.

        The `spec_to_grid_generic` method is used with
        `plm=PLM` to implement this transformation.

        Parameters
        ----------
        spec_x : np.ndarray, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.

        Returns
        -------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        return self.spec_to_grid_generic(spec_x, self.PLM)
    
    def spec_to_grid_grad_theta(self, spec_x):
        r"""Applies the transformation from spectral space to grid space
        with gradients with respect to latitude ($\theta$).

        The `spec_to_grid_generic` method is used with
        `plm=ALM` to implement this transformation.

        Parameters
        ----------
        spec_x : np.ndarray, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.

        Returns
        -------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Gradient of variable $x$ with respect to latitude in grid space.
        """
        return self.spec_to_grid_generic(spec_x, self.ALM)
    
    def spec_to_grid_grad_phi(self, spec_x):
        r"""Applies the transformation from spectral space to grid space
        with gradients with respect to longitude ($\phi$).

        The `spec_to_grid_generic` method is used with
        `plm=PPLM` to implement this transformation.

        Parameters
        ----------
        spec_x : np.ndarray, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.

        Returns
        -------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Gradient of variable $x$ with respect to longitude in grid space.
        """
        spec_dx = np.zeros(spec_x.shape)
        spec_dx[..., 0, :, :] = - np.arange(self.T+1) * spec_x[..., 1, :, :]
        spec_dx[..., 1, :, :] = np.arange(self.T+1) * spec_x[..., 0, :, :]
        return self.spec_to_grid_generic(spec_dx, self.PPLM)

    def grid_to_spec(self, x):
        r"""Applies the transformation from grid space to spectral space.

        Parameters
        ----------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.

        Returns
        -------
        spec_x : np.ndarray, shape (..., 2, T+1, T+1)
            Variable $\hat{x}$ in spectral space.
        """
        leg_x = self.apply_ifft(x)
        return np.einsum('...im,iml->...lm', leg_x, self.PW)

    def precompute_planet_vorticity(self):
        r"""Pre-computes the planet vorticity $q^{\mathsf{p}}$.
        
        Using the $\beta$-plane approximation, the
        planet vorticity $q^{\mathsf{p}}$ is defined as...

        This method is used by the Poisson solver
        to define the transformation

        Returns
        -------
        qp : np.ndarray, shape (Nlat, Nlon)
            Planet vorticity $q^{\mathsf{p}}$ in grid space.
        """
        spec_qp = np.zeros((2, self.T+1, self.T+1))
        spec_qp[0, 1, 0] = 2 * self.planet_omega / np.sqrt(3)
        return self.spec_to_grid(spec_qp)
        
