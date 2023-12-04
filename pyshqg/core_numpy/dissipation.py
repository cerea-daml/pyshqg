import numpy as np

class QGEkmanDissipation:

    def __init__(
        self,
        spectral_transformations,
        orography,
        tau,
        weight_land_sea_mask,
        weight_orography,
        orography_scale,
    ):
        # compute friction field
        self.mu = orography.precompute_ekman_friction(
            weight_land_sea_mask,
            weight_orography,
            orography_scale,
            tau,
        )

        # compute spatial gradients of friction field
        spec_mu = spectral_transformations.grid_to_spec(self.mu)
        self.d_mu_d_phi = spectral_transformations.spec_to_grid_grad_phi(
            spec_mu
        )
        self.d_mu_d_theta = spectral_transformations.spec_to_grid_grad_theta(
            spec_mu
        )   

    def compute_ekman_dissipation(self, zeta, gradients):
        ekman_1 = zeta*self.mu
        ekman_2 = ( 
            self.d_mu_d_theta * gradients['d_psi_d_theta'][..., -1, :, :] +
            self.d_mu_d_phi * gradients['d_psi_d_phi'][..., -1, :, :]
        )
        # TODO: replace with np.pad?
        ekman = np.zeros_like(gradients['d_psi_d_theta'])
        ekman[..., -1, :, :] = ekman_1 + ekman_2
        return ekman

class QGSelectiveDissipation:

    def __init__(
        self,
        spectral_transformations,
        tau,
    ):
        # compute spectrum
        T = spectral_transformations.T
        self.spectrum = (
            spectral_transformations.planet_radius**2 *
            spectral_transformations.laplacian_spectrum / (
            T * (T+1)
        ))**4 / tau

    def compute_selective_dissipation(self, spec_total_q):
        return np.einsum(
            '...lm,l->...lm',
            spec_total_q,
            self.spectrum,
        )

class QGThermalDissipation:

    def __init__(
        self,
        vertical_parametrisation,
        tau,
    ):
        self.thermal_coupling = vertical_parametrisation.precompute_coupling_matrix(
            scaling=1/tau
        )

    def compute_thermal_dissipation(self, spec_psi):
        return np.einsum(
            '...jklm,ij->...iklm',
            spec_psi,
            self.thermal_coupling,
        )

class QGDissipation:

    def __init__(
        self,
        ekman,
        selective,
        thermal,
    ):
        self.ekman = ekman
        self.selective = selective
        self.thermal = thermal
        
