r"""Submodule dedicated to numpy backend."""

import numpy as np

import pyshqg.backend.abstract

class NumpyBackend(pyshqg.backend.abstract.Backend):
    r"""Numpy backend."""

    def from_numpy(self, array):
        r"""Converts an array from `numpy` into backend format.

        Here the backend is `numpy`.
        Therefore, there is nothing to do except potentially change the data type.

        Parameters
        ----------
        array : numpy.ndarray
            Array in `numpy` format.

        Returns
        -------
        array : numpy.ndarray
            Array in backend format.
        """
        return array.astype(self.floatx)

    @staticmethod
    def to_numpy(array):
        r"""Converts an array from backend into `numpy` format.

        Here the backend is `numpy`.
        Therefore, there is nothing to do.

        Parameters
        ----------
        array : numpy.ndarray
            Array in backend format.

        Returns
        -------
        array : numpy.ndarray
            Array in `numpy` format.
        """
        return array

    @staticmethod
    def expand_dims(*args, **kwargs):
        r"""Wrapper around `numpy.expand_dims`."""
        return np.expand_dims(*args, **kwargs)

    @staticmethod
    def pad(*args, **kwargs):
        r"""Wrapper around `numpy.pad`."""
        return np.pad(*args, **kwargs)

    @staticmethod
    def einsum(*args, **kwargs):
        r"""Wrapper around `numpy.einsum`."""
        return np.einsum(*args, **kwargs)

    def range(self, *args, **kwargs):
        r"""Wrapper around `numpy.arange` using real numbers."""
        return np.arange(*args, **kwargs, dtype=self.floatx)

    @staticmethod
    def concatenate(*args, **kwargs):
        r"""Wrapper around `numpy.concatenate`."""
        return np.concatenate(*args, **kwargs)

    @staticmethod
    def repeat(*args, **kwargs):
        r"""Wrapper around `numpy.repeat`."""
        return np.repeat(*args, **kwargs)

    @staticmethod
    def apply_fft(T, T_grid, leg_x):
        r"""Forward Fourier transformation for the spectral harmonics.

        Notes
        -----
        For the Gauss--Legendre grid used here, we have
        $N_{\mathsf{lat}}=T_{\mathsf{grid}}+1$ and
        $N_{\mathsf{lon}}=2N_{\mathsf{lat}}=2\times(T_{\mathsf{grid}}+1)$.

        Parameters
        ----------
        T : int
            Truncature of the data in spectral space.
        T_grid : int
            Truncature of the Gauss--Legendre grid.
        leg_x : np.ndarray, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.

        Returns
        -------
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        leg_x = leg_x[..., 0, :, :] + 1j*leg_x[..., 1, :, :]
        rank = len(leg_x.shape)
        paddings = [[0, 0] for _ in range(rank-1)] + [[0, T_grid+1-T]]
        leg_x = np.pad(leg_x, paddings, mode='constant', constant_values=0)
        return np.fft.hfft(leg_x, axis=-1)

    @staticmethod
    def apply_ifft(T, T_grid, x):
        r"""Inverse Fourier transformation for the spectral harmonics.

        Notes
        -----
        For the Gauss--Legendre grid used here, we have
        $N_{\mathsf{lat}}=T_{\mathsf{grid}}+1$ and
        $N_{\mathsf{lon}}=2N_{\mathsf{lat}}=2\times(T_{\mathsf{grid}}+1)$.

        Parameters
        ----------
        T : int
            Truncature of the data in spectral space.
        T_grid : int
            Truncature of the Gauss--Legendre grid.
        x : np.ndarray, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.

        Returns
        -------
        leg_x : np.ndarray, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.
        """
        leg_x = np.fft.ihfft(x, axis=-1)[..., :T+1]
        return np.stack([leg_x.real, leg_x.imag], axis=-3)

