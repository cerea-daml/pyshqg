r"""Submodule dedicated to sources."""

import tensorflow as tf

class QGForcing:
    r"""Class for a standard, constant forcing.

    Attributes
    ----------
    forcing : tf.Tensor, shape (Nlevel, Nlat, Nlon)
        Forcing coefficients in grid space.
    """

    def __init__(self, forcing):
        r"""Constructor for the forcing.

        Parameters
        ----------
        forcing : numpy.ndarray, shape (Nlevel, Nlat, Nlon)
            Forcing coefficients in grid space.
        """
        self.forcing = tf.convert_to_tensor(forcing)

    def compute_forcing(self):
        r"""Computes the forcing

        Note that the forcing coefficients have been
        pre-computed in the grid.

        Returns
        -------
        forcing : tf.Tensor, shape (Nlevel, Nlat, Nlon)
            Forcing coefficients in grid space.
        """
        return self.forcing

