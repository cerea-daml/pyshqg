r"""Submodule dedicated to the Poisson solver."""

import numpy as np


class QGPoissonSolver:
    r"""Class for the Poisson solver.

    The goal of this class is to define the transformations
    $\hat{q} \mapsto \hat{q}^{\mathsf{t}} \mapsto \hat{\psi} \mapsto \zeta$.
    The (relative) vorticity $\hat{q}$ is the control variable of the QG model.
    The total vorticity $\hat{q}^{\mathsf{t}}$ is related to $\hat{q}$ by...
    Then, $\hat{q}^{\mathsf{t}}$ and the stream function $\hat{\psi}$ are related 
    by the following Poisson-like equation...
    Finally, the drag $\zeta$ is defined as...

    Attributes
    ----------
    spectral_transformations : pyshqg.core_numpy.spectral_transformations.SpectralTransformations
        The object encapsulating the spectral transformations.
    spec_qp : np.ndarray, shape (Nlevel, 2, T+1, T+1)
        The planet vorticity, corrected by the orography, in spectral space $\hat{q}^{\mathsf{p}}$.
    q_to_psi_coupling : np.ndarray, shape (Nlevel, Nlevel, T+1)
        The vertical coupling coefficients for the
        $\hat{q}^{\mathsf{t}} \mapsto \hat{\psi}$ transformation.
    """

    def __init__(
        self,
        spectral_transformations,
        vertical_parametrisation,
        orography,
        orography_scale,
    ):
        r"""Constructor for the Poisson solver.

        First, the planet voticity in spectral space $\hat{q}^{\mathsf{p}}$
        is pre-computed using the `precompute_planet_vorticity` method, then the 
        vertical coupling coefficients are pre-computed using
        the `precompute_q_to_psi_coupling` method.

        Parameters
        ----------
        spectral_transformations : pyshqg.core_numpy.spectral_transformations.SpectralTransformations
            The object encapsulating the spectral transformations.
        vertical_parametrisation : pyshqg.preprocessing.vertical_parametrisation.VerticalParametrisation
            The object encapsulating the vertical parametrisation.
        orography : pyshqg.preprocessing.orography.Orography
            The object encapsulating the orography in the grid.
        orography_scale : float
            The vertical length scale for the orography.
        """
        self.spectral_transformations = spectral_transformations
        
        # pre-compute the corrected planet vorticity
        self.spec_qp = self.precompute_planet_vorticity(
            vertical_parametrisation,
            orography,
            orography_scale,
        )
        
        # pre-compute the coefficients for the q to psi transformation
        self.q_to_psi_coupling = self.precompute_q_to_psi_coupling(
            vertical_parametrisation,
        )

    def precompute_planet_vorticity(
        self,
        vertical_parametrisation,
        orography,
        orography_scale,
    ):
        r"""Pre-computes the planet vorticity in spectral space $\hat{q}^{\mathsf{p}}$.

        First, the planet voticity is pre-computed in grid space
        using the `spectral_transformations` object. Then,
        the orography correction is added in grid space by the
        `orography` object. Finally, the corrected planet
        vorticity is sent to spectral space using 
        `spectral_transformations` object.

        Parameters
        ----------
        vertical_parametrisation : pyshqg.preprocessing.vertical_parametrisation.VerticalParametrisation
            The object encapsulating the vertical parametrisation.
        orography : pyshqg.preprocessing.orography.Orography
            The object encapsulating the orography in the grid.
        orography_scale : float
            The vertical length scale for the orography.

        Returns
        -------
        spec_qp : np.ndarray, shape (Nlevel, 2, T+1, T+1)
            The corrected planet vorticity in spectral space $\hat{q}^{\mathsf{p}}$.
        """
        # planet rotation (beta plane approximation)
        qp = self.spectral_transformations.precompute_planet_vorticity()
        qp = np.repeat(
            qp[np.newaxis, :, :], 
            vertical_parametrisation.num_levels,
            axis=0,
        )

        # orography correction
        qp = orography.precorrect_planet_vorticity(qp, orography_scale)

        # total f in spectral space
        return self.spectral_transformations.grid_to_spec(qp)

    def precompute_q_to_psi_coupling(
        self,
        vertical_parametrisation,
    ):
        r"""Pre-computes the vertical coupling coefficients.

        First, the inverse coupling matrix is computed using
        the `vertical_parametrisation` object. Then, the
        inverse coupling matrix is corrected using Eigen
        spectrum of the Laplacian operator, defined in the
        `spectral_transformations` object, and inversed
        using `numpy.linalg.inv`.

        Note that this method distinguished the special
        case l=0 from the general case.

        Parameters
        ----------
        vertical_parametrisation : pyshqg.preprocessing.vertical_parametrisation.VerticalParametrisation
            The object encapsulating the vertical parametrisation.

        Returns
        -------
        q_to_psi_coupling : np.ndarray, shape (Nlevel, Nlevel, T+1)
            The vertical coupling coefficients.
        """
        # first compute the common part of the coupling matrix
        coupling_matrix = vertical_parametrisation.precompute_coupling_matrix()

        # then compute the coefficients
        q_to_psi_coupling = np.zeros((
            vertical_parametrisation.num_levels,
            vertical_parametrisation.num_levels,
            self.spectral_transformations.T+1,
        ))

        # special case for l = 0
        q_to_psi_coupling[1:, 1:, 0] = np.linalg.inv(coupling_matrix[1:, 1:])

        # general case for l > 0
        for l in range(1, self.spectral_transformations.T+1):
            q_to_psi_coupling[..., l] = np.linalg.inv(
                self.spectral_transformations.laplacian_spectrum[l] *
                np.eye(vertical_parametrisation.num_levels) + 
                coupling_matrix
            )

        return q_to_psi_coupling

    def q_to_total_q(self, spec_q):
        r"""Computes the $\hat{q} \mapsto \hat{q}^{\mathsf{t}}$ transformation.

        To do this, the planet vorticity in spectral space
        $\hat{q}^{\mathsf{p}}$ is added to the relative vorticity
        in spectral space $\hat{q}$.

        Parameters
        ----------
        spec_q : np.ndarray, shape (..., Nlevel, 2, T+1, T+1)
            The relative vorticity in spectral space $\hat{q}$.

        Returns
        -------
        spec_total_q : np.ndarray, shape (..., Nlevel, 2, T+1, T+1)
            The total vorticity in spectral space $\hat{q}^{\mathsf{t}}$.
        """
        return spec_q + self.spec_qp

    def total_q_to_psi(self, spec_total_q):
        r"""Computes the $\hat{q}^{\mathsf{t}} \mapsto \hat{\psi}$ transformation.

        To do this, $\hat{q}^{\mathsf{t}}$ is multiplied by the 
        vertical coupling coefficients.

        Parameters
        ----------
        spec_total_q : np.ndarray, shape (..., Nlevel, 2, T+1, T+1)
            The total vorticity in spectral space $\hat{q}^{\mathsf{t}}$.

        Returns
        -------
        spec_psi : np.ndarray, shape (..., Nlevel, 2, T+1, T+1)
            The stream function in spectral space $\hat{\psi}$.
        """
        return np.einsum(
            'ijl,...jklm->...iklm',
            self.q_to_psi_coupling,
            spec_total_q,
        )

    def psi_to_zeta(self, spec_psi):
        r"""Computes the $\hat{psi} \mapsto \zeta$ transformation.

        To do this, the lowest level of $\hat{psi}$ (indexed by -1)
        is multiplied by the Eigen
        values of the Laplacian operator to get $\hat{\zeta}$.
        Then, $\hat{\zeta}$ is sent to grid space to get $\zeta$.

        Parameters
        ----------
        spec_psi : np.ndarray, shape (..., Nlevel, 2, T+1, T+1)
            The stream function in spectral space $\hat{\psi}$.

        Returns
        -------
        zeta : np.ndarray, shape (..., Nlat, Nlon)
            The drag in grid space $\zeta$.
        """
        spec_zeta = np.einsum(
            '...lm,l->...lm',
            spec_psi[..., -1, :, :, :],
            self.spectral_transformations.laplacian_spectrum,
        )
        return self.spectral_transformations.spec_to_grid(spec_zeta)
            
