r"""Submodule dedicated to dissipation processes."""
# TODO:
# - ekman: replace np.zeros_like by np.pad?

import dataclasses

import numpy as np


class QGEkmanDissipation:
    r"""Class for the Ekman dissipation processes.

    The Ekman dissipation term is defined as...

    Attributes
    ----------
    mu : np.ndarray, shape (Nlat, Nlon)
        Friction coefficient $\mu$ defined on the grid.
    dmu_dphi : np.ndarray, shape (Nlat, Nlon)
        Gradient of $\mu$ with respect to longitude ($\phi$).
    dmu_dtheta : np.ndarray, shape (Nlat, Nlon)
        Gradient of $\mu$ with respect to latitude ($\theta$).
    """

    def __init__(
        self,
        spectral_transformations,
        orography,
        tau,
        weight_land_sea_mask,
        weight_orography,
        orography_scale,
    ):
        r"""Constructor for the Ekman dissipation.

        First, the `orography` object is used to pre-compute the friction coefficient
        $\mu$ in the grid. Then, the `spectral_transformations` object is used
        to pre-compute the spatial gradients of $\mu$ in the grid.

        Parameters
        ----------
        spectral_transformations : pyshqg.core_numpy.spectral_transformations.SpectralTransformations
            The object encapsulating the spectral transformations.
        orography : pyshqg.preprocessing.orography.Orography
            The object encapsulating the orography in the grid.
        tau : float
            The parameter $\tau$ of the friction coefficient $\mu$.
        weight_land_sea_mask : float
            The weight of the land/sea mask contribution in $\mu$.
        weight_orography : float
            The weight of the orography contribution in $\mu$.
        orography_scale : float
            The vertical length scale for the orography in $\mu$.
        """
        # pre-compute friction field
        self.mu = orography.precompute_ekman_friction(
            weight_land_sea_mask,
            weight_orography,
            orography_scale,
            tau,
        )

        # pre-compute spatial gradients of friction field
        spec_mu = spectral_transformations.grid_to_spec(self.mu)
        self.dmu_dphi = spectral_transformations.spec_to_grid_grad_phi(
            spec_mu
        )
        self.dmu_dtheta = spectral_transformations.spec_to_grid_grad_theta(
            spec_mu
        )   

    def compute_ekman_dissipation(self, zeta, dpsi_dtheta, dpsi_dphi):
        r"""Computes the Ekman dissipation term in the QG model tendencies.

        The Ekman dissipation term is computed in the grid, 
        at the lowest level (indexed by -1).
        The output is padded with zeros to get a contribution for all
        levels.

        Parameters
        ----------
        zeta : np.ndarray, shape (..., Nlat, Nlon)
            The drag $\zeta$ in grid space.
        dpsi_dtheta : np.ndarray, shape (..., Nlevel, Nlat, Nlon)
            The gradient of the stream function $\psi$ with respect 
            to latitude ($\theta$) in grid space.
        dpsi_dphi : np.ndarray, shape (..., Nlevel, Nlat, Nlon)
            The gradient of the stream function $\psi$ with respect 
            to longitude ($\phi$) in grid space.

        Returns
        -------
        ekman : np.ndarray, shape (..., Nlevel, Nlat, Nlon)
            The Ekman dissipation term in the grid.
        """
        ekman_1 = zeta * self.mu
        ekman_2 = ( 
            self.dmu_dtheta * dpsi_dtheta[..., -1, :, :] +
            self.dmu_dphi * dpsi_dphi[..., -1, :, :]
        )
        ekman = np.zeros_like(dpsi_dtheta)
        ekman[..., -1, :, :] = ekman_1 + ekman_2
        return ekman


class QGSelectiveDissipation:
    r"""Class for the selective dissipation processes.

    The selective dissipation term is defined as...

    Attributes
    ----------
    spectrum : np.ndarray, shape (T+1,)
        Eignen spectrum in spectral space, with respect to `l`
        (the first spectral index), of the selective dissipation.
    """

    def __init__(
        self,
        spectral_transformations,
        tau,
    ):
        r"""Constructor for the selective dissipation.

        The Eigen spectrum of the Laplacian operator, defined in the
        `spectral_transformations` object is used to pre-compute the Eigen spectrum 
        of the selective dissipation.

        Parameters
        ----------
        spectral_transformations : pyshqg.core_numpy.spectral_transformations.SpectralTransformations
            The object encapsulating the spectral transformations.
        tau : float
            The parameter $\tau$ of the selective dissipation.
        """
        T = spectral_transformations.T
        self.spectrum = (
            spectral_transformations.planet_radius**2 *
            spectral_transformations.laplacian_spectrum / (
            T * (T+1)
        ))**4 / tau

    def compute_selective_dissipation(self, spec_total_q):
        r"""Computes the selective dissipation term in the QG model tendencies.

        The selective dissipation term is computed in spectral space,
        using the Eigen spectrum.

        Parameters
        ----------
        spec_total_q : np.ndarray, shape (..., Nlevel, 2, T+1, T+1)
            The total vorticity in spectral space $\hat{q}^{\mathsf{t}}$.

        Returns
        -------
        selective : np.ndarray, shape (..., Nlevel, 2, T+1, T+1)
            The selective dissipation term in spectral space.
        """
        return np.einsum(
            '...lm,l->...lm',
            spec_total_q,
            self.spectrum,
        )


class QGThermalDissipation:
    r"""Class for the thermal dissipation processes.

    The thermal dissipation term is defined as...

    Attributes
    ----------
    thermal_coupling : np.ndarray, shape (Nlevel, Nlevel)
        Vertical coupling matrix for the thermal dissipation.
    """

    def __init__(
        self,
        vertical_parametrisation,
        tau,
    ):
        r"""Constructor for the thermal dissipation.

        The vertical coupling matrix is pre-computed using 
        the `vertical_parametrisation` object.

        Parameters
        ----------
        vertical_parametrisation : pyshqg.preprocessing.vertical_parametrisation.VerticalParametrisation
            The object encapsulating the vertical parametrisation.
        tau : float
            The parameter $\tau$ of the thermal dissipation.
        """
        self.thermal_coupling = vertical_parametrisation.precompute_coupling_matrix(
            scaling=1/tau
        )

    def compute_thermal_dissipation(self, spec_psi):
        r"""Computes the thermal dissipation term in the QG model tendencies.

        The thermal dissipation term is computed in spectral space,
        using a coupling matrix between vertical levels.

        Parameters
        ----------
        spec_psi : np.ndarray, shape (..., Nlevel, 2, T+1, T+1)
            The stream function in spectral space $\hat{\psi}$.

        Returns
        -------
        thermal : np.ndarray, shape (..., Nlevel, 2, T+1, T+1)
            The thermal dissipation term in spectral space.
        """
        return np.einsum(
            '...jklm,ij->...iklm',
            spec_psi,
            self.thermal_coupling,
        )


@dataclasses.dataclass
class QGDissipation: 
    r"""Container class for all dissipation processes.

    Parameters
    ----------
    ekman: pyshqg.core_numpy.dissipation.QGEkmanDissipation
        The Ekman dissipation object.
    selective: pyshqg.core_numpy.dissipation.QGSelectiveDissipation
        The selective dissipation object.
    thermal: pyshqg.core_numpy.dissipation.QGThermalDissipation
        The thermal dissipation object.
    """
    ekman: 'pyshqg.core_numpy.dissipation.QGEkmanDissipation'
    selective: 'pyshqg.core_numpy.dissipation.QGSelectiveDissipation'
    thermal: 'pyshqg.core_numpy.dissipation.QGThermalDissipation'

