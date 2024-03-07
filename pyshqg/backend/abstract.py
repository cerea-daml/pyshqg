r"""Submodule dedicated to abstract backend."""

import abc


class Backend(abc.ABC):
    r"""Abstract backend.

    Attributes
    ----------
    floatx : str
        Precision for real numbers.
    """

    def __init__(self, floatx):
        r"""Constructor for the backend.

        Parameters
        ----------
        floatx : str
            Precision for real numbers.
        """
        self.floatx = floatx

    @abc.abstractmethod
    def from_numpy(self, array):
        r"""Converts an array from `numpy` into backend format.

        Parameters
        ----------
        array : numpy.ndarray
            Array in `numpy` format.

        Returns
        -------
        array : backend array
            Array in backend format.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def to_numpy(array):
        r"""Converts an array from backend into `numpy` format.

        Parameters
        ----------
        array : backend array
            Array in backend format.

        Returns
        -------
        array : numpy.ndarray
            Array in `numpy` format.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def expand_dims(*args, **kwargs):
        r"""Backend equivalent of `numpy.expand_dims`."""
        pass

    @staticmethod
    @abc.abstractmethod
    def pad(*args, **kwargs):
        r"""Backend equivalent of `numpy.pad`."""
        pass

    @staticmethod
    @abc.abstractmethod
    def einsum(*args, **kwargs):
        r"""Backend equivalent of `numpy.einsum`."""
        pass

    @abc.abstractmethod
    def range(self, *args, **kwargs):
        r"""Backend equivalent of `numpy.arange` using real numbers."""
        pass

    @staticmethod
    @abc.abstractmethod
    def concatenate(*args, **kwargs):
        r"""Backend equivalent of `numpy.concatenate`."""
        pass

    @staticmethod
    @abc.abstractmethod
    def repeat(*args, **kwargs):
        r"""Backend equivalent of `numpy.repeat`."""
        pass

    @staticmethod
    @abc.abstractmethod
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
        leg_x : backend array, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.

        Returns
        -------
        x : backend array, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        pass

    @staticmethod
    @abc.abstractmethod
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
        x : backend array, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.

        Returns
        -------
        leg_x : backend array, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.
        """
        pass

