r"""Submodule dedicated to sources."""

import dataclasses

@dataclasses.dataclass
class QGForcing:
    r"""Class for a standard, constant forcing.

    Attributes
    ----------
    forcing : np.ndarray, shape (Nlevel, Nlat, Nlon)
        Forcing coefficients in grid space.
    """
    forcing : 'numpy.ndarray'

    def compute_forcing(self):
        r"""Computes the forcing

        Note that the forcing coefficients have been
        pre-computed in the grid.

        Returns
        -------
        forcing : np.ndarray, shape (Nlevel, Nlat, Nlon)
            Forcing coefficients in grid space.
        """
        return self.forcing

