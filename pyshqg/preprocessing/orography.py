r"""Submodule dedicated to the orography."""

import dataclasses

import numpy as np


@dataclasses.dataclass
class Orography:
    r"""Container class for the orography.

    The purpose of this class is to implement
    pre-processing functions related to
    the orography.

    Parameters
    ----------
    land_sea_mask : np.ndarray, shape (Nlat, Nlon)
        The land/sea mask.
    orography : np.ndarray, shape (Nlat, Nlon)
        The orography.
    """
    land_sea_mask: 'numpy.ndarray'
    orography: 'numpy.ndarray'

    def precorrect_planet_vorticity(
        self,
        qp,
        orography_scale,
    ):
        r"""Corrects the precomputed planet vorticity.

        Parameters
        ----------
        qp : np.ndarray, shape (Nlevel, Nlat, Nlon)
            The pre-computed planet vorticity.
        orography_scale : float
            The orography vertical length scale for the correction.

        Returns
        -------
        qp : np.ndarray, shape (Nlevel, Nlat, Nlon)
            The corrected pre-computed planet vorticity.
        """
        qp[-1] *= 1 + self.orography / orography_scale
        return qp

    def precompute_ekman_friction(
        self,
        weight_land_sea_mask,
        weight_orography,
        orography_scale,
        tau,
    ):
        r"""Pre-computes the Ekman friction coefficient.

        Parameters
        ----------
        weight_land_sea_mask : float
            Weight of the land/sea mask contribution.
        weight_orography : float
            Weight of the orography mask contribution.
        orography_scale : float
            The orography vertical length scale.
        tau : float
            The time scale.

        Returns
        -------
        mu : np.ndarray, shape (Nlat, Nlon)
            Friction coefficient.
        """
        return ( 
            1 + 
            weight_land_sea_mask * self.land_sea_mask +
            weight_orography * ( 
                1 -
                np.exp(-np.maximum(
                    0, 
                    self.orography/orography_scale,
                ))
            )
        )/tau
