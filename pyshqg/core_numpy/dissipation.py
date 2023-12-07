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
        self.dmu_dphi = spectral_transformations.spec_to_grid_grad_phi(
            spec_mu
        )
        self.dmu_dtheta = spectral_transformations.spec_to_grid_grad_theta(
            spec_mu
        )   

    def compute_ekman_dissipation(self, state):
        ekman_1 = state['zeta']*self.mu
        ekman_2 = ( 
            self.dmu_dtheta * state['dpsi_dtheta'][..., -1, :, :] +
            self.dmu_dphi * state['dpsi_dphi'][..., -1, :, :]
        )
        # TODO: replace with np.pad?
        ekman = np.zeros_like(state['dpsi_dtheta'])
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

    def compute_selective_dissipation(self, state):
        return np.einsum(
            '...lm,l->...lm',
            state['spec_total_q'],
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

    def compute_thermal_dissipation(self, state):
        return np.einsum(
            '...jklm,ij->...iklm',
            state['spec_psi'],
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
        
