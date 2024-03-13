r"""Submodule dedicated to sources."""


class QGForcing:
    r"""Class for a standard, constant forcing.

    Attributes
    ----------
    forcing : backend array, shape (Nlevel, Nlat, Nlon)
        Forcing coefficients in grid space.
    """

    def __init__(self, backend, forcing):
        r"""Constructor for the forcing.

        Parameters
        ----------
        backend : pyshqg.backend.Backend object
            The backend.
        forcing : numpy.ndarray, shape (Nlevel, Nlat, Nlon)
            Forcing coefficients in grid space.
        """
        self.forcing = backend.from_numpy(forcing)

    def compute_forcing(self):
        r"""Computes the forcing

        Note that the forcing coefficients have been
        pre-computed in the grid.

        Returns
        -------
        forcing : backend array, shape (Nlevel, Nlat, Nlon)
            Forcing coefficients in grid space.
        """
        return self.forcing

