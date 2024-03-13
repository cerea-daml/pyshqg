r"""Submodule dedicated to tensorflow backend."""

import tensorflow as tf

import pyshqg.backend.abstract


class TensorflowBackend(pyshqg.backend.abstract.Backend):
    r"""Tensorflow backend."""

    def from_numpy(self, array):
        r"""Converts an array from `numpy` into backend format.

        Parameters
        ----------
        array : numpy.ndarray
            Array in `numpy` format.

        Returns
        -------
        tensor : tensorflow.Tensor
            Array in backend format.
        """
        return tf.convert_to_tensor(array, dtype=self.floatx)

    @staticmethod
    def to_numpy(tensor):
        r"""Converts an array from backend into `numpy` format.

        Parameters
        ----------
        tensor : tensorflow.Tensor
            Array in backend format.

        Returns
        -------
        array : numpy.ndarray
            Array in `numpy` format.
        """
        return tensor.numpy()

    @staticmethod
    def expand_dims(*args, **kwargs):
        r"""Wrapper around `tensorflow.expand_dims`."""
        return tf.expand_dims(*args, **kwargs)

    @staticmethod
    def pad(*args, mode='constant', **kwargs):
        r"""Wrapper around `tensorflow.pad`.

        Notes
        -----
        The kwarg `mode` is uppercased before
        being sent to `tensorflow.pad` in order
        to match `numpy` syntax.
        """
        return tf.pad(*args, mode=mode.upper(), **kwargs)

    @staticmethod
    def einsum(*args, **kwargs):
        r"""Wrapper around `tensorflow.einsum`."""
        return tf.einsum(*args, **kwargs)

    def range(self, *args, **kwargs):
        r"""Wrapper around `tensorflow.range` using real numbers."""
        return tf.range(*args, **kwargs, dtype=self.floatx)

    @staticmethod
    def concatenate(*args, **kwargs):
        r"""Wrapper around `tensorflow.concat`."""
        return tf.concat(*args, **kwargs)

    @staticmethod
    def repeat(*args, **kwargs):
        r"""Wrapper around `tensorflow.repeat`."""
        return tf.repeat(*args, **kwargs)

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
        leg_x : tensorflow.Tensor, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.

        Returns
        -------
        x : tensorflow.Tensor, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.
        """
        leg_x = tf.dtypes.complex(leg_x[..., 0, :, :], -leg_x[..., 1, :, :])
        rank = len(leg_x.shape)
        paddings = [[0, 0] for _ in range(rank-1)] + [[0, T_grid+1-T]]
        leg_x = tf.pad(leg_x, paddings, mode='CONSTANT', constant_values=0)
        return tf.signal.irfft(leg_x) * (2*T_grid+2)

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
        x : tensorflow.Tensor, shape (..., Nlat, Nlon)
            Variable $x$ in grid space.

        Returns
        -------
        leg_x : tensorflow.Tensor, shape (..., 2, Nlat, T+1)
            Legendre transform of $\hat{x}$.
        """
        leg_x = tf.signal.rfft(x)[..., :T+1] / (2*T_grid+2)
        return tf.stack([tf.math.real(leg_x), -tf.math.imag(leg_x)], axis=-3)

