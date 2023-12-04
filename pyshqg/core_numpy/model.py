
class QGModel:

    def __init__(
        self,
        spectral_transformations,
        poisson_solver,
        dissipation,
        forcing,
    ):
        self.spectral_transformations = spectral_transformations
        self.poisson_solver = poisson_solver
        self.dissipation = dissipation
        self.forcing = forcing

    def compute_model_tendencies(self, spec_q):
        # compute total q
        spec_total_q = self.poisson_solver.q_to_total_q(spec_q)
        
        # compute psi
        spec_psi = self.poisson_solver.total_q_to_psi(spec_total_q)

        # compute zeta
        zeta =self.poisson_solver.psi_to_zeta(spec_psi)

        # compute spatial gradients
        gradients = dict(
            d_q_d_theta=self.spectral_transformations.spec_to_grid_grad_theta(spec_q),
            d_q_d_phi=self.spectral_transformations.spec_to_grid_grad_phi(spec_q),
            d_psi_d_theta=self.spectral_transformations.spec_to_grid_grad_theta(spec_psi),
            d_psi_d_phi=self.spectral_transformations.spec_to_grid_grad_phi(spec_psi),
        )

        # compute Jacobian
        jacobian = (
            gradients['d_q_d_phi'] * gradients['d_psi_d_theta'] - 
            gradients['d_q_d_theta'] * gradients['d_psi_d_phi']
        )

        # compute forcing
        forcing = self.forcing.compute_forcing()

        # compute Ekman dissipation
        dissipation_ekman = self.dissipation.ekman.compute_ekman_dissipation(zeta, gradients)

        # compute selective dissipation
        spec_dissipation_selective = self.dissipation.selective.compute_selective_dissipation(spec_total_q)

        # compute thermal dissipation
        spec_dissipation_thermal = self.dissipation.thermal.compute_thermal_dissipation(spec_psi)

        # aggregate contributions in grid space
        tendencies = jacobian + forcing - dissipation_ekman

        # aggregate contributions in spectral space
        spec_tendencies = -spec_dissipation_selective - spec_dissipation_thermal

        # return all contributions in spectral space
        return spec_tendencies + self.spectral_transformations.grid_to_spec(tendencies)