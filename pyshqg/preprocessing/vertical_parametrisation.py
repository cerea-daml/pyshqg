r"""Submodule dedicated to the vertical parametrisation."""

import numpy as np


class VerticalParametrisation:
    r"""Container class for the vertical parametrisation.

    Attributes
    ----------
    rossby_radius_list : list of float
        List of Rossby radius for each level interface.
    num_levels : int
        Number of vertical levels.
    """

    def __init__(self, rossby_radius_list):
        r"""Constructs the vertical parametrisation.

        Parameters
        ----------
        rossby_radius_list : list of float
            List of Rossby radius for each level interface.
        """
        self.rossby_radius_list = rossby_radius_list
        self.num_levels = len(rossby_radius_list) + 1

    def precompute_coupling_matrix(self, scaling=1):
        r"""Pre-computes the vertical coupling matrix.

        Parameters
        ----------
        scaling : float
            Scaling of the matrix.

        Returns
        -------
        coupling_matrix : np.ndarray, shape (Nlevel, Nlevel)
            The vertical coupling matrix.
        """
        coupling_matrix = np.zeros((self.num_levels, self.num_levels))
        for (z, rossby_radius) in enumerate(self.rossby_radius_list):
            coupling = 1 / rossby_radius**2
            coupling_matrix[z:z+2, z:z+2] -= coupling * np.array([[1, -1], [-1, 1]])
        return scaling*coupling_matrix
