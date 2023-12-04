import numpy as np

class QGPoissonSolver:

    def __init__(
        self,
        spectral_transformations,
        vertical_parametrisation,
        orography,
        orography_scale,
    ):
        self.spectral_transformations = spectral_transformations
        
        # compute the corrected planet vorticity
        self.spec_planet_vorticity = self.precompute_planet_vorticity(
            vertical_parametrisation,
            orography,
            orography_scale,
        )
        
        # compute the coefficients for the q to psi transformation
        self.q_to_psi_coupling = self.precompute_q_to_psi_coupling(
            vertical_parametrisation,
        )

    def precompute_planet_vorticity(
        self,
        vertical_parametrisation,
        orography,
        orography_scale,
    ):
        # planet rotation (beta plane approximation)
        f = self.spectral_transformations.precompute_planet_vorticity()
        f = np.repeat(
            f[np.newaxis, :, :], 
            vertical_parametrisation.num_levels,
            axis=0,
        )

        # orography correction
        f = orography.precorrect_planet_vorticity(f, orography_scale)

        # total f in spectral space
        return self.spectral_transformations.grid_to_spec(f)

    def precompute_q_to_psi_coupling(
        self,
        vertical_parametrisation,
    ):
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
        return spec_q + self.spec_planet_vorticity

    def total_q_to_psi(self, spec_total_q):
        return np.einsum(
            'ijl,...jklm->...iklm',
            self.q_to_psi_coupling,
            spec_total_q,
        )

    def psi_to_zeta(self, spec_psi):
        spec_zeta = np.einsum(
            '...lm,l->...lm',
            spec_psi[..., -1, :, :, :],
            self.spectral_transformations.laplacian_spectrum,
        )
        return self.spectral_transformations.spec_to_grid(spec_zeta)
            
