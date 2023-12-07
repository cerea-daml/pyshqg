import numpy as np
import xarray as xr

QG_CORE_DIMENSIONS = dict(
    spec_q=['level', 'c', 'l', 'm'],
    spec_total_q=['level', 'c', 'l', 'm'],
    spec_psi=['level', 'c', 'l', 'm'],
    zeta=['lat', 'lon'],
    q=['level', 'lat', 'lon'],
    psi=['level', 'lat', 'lon'],
)

class QGModelTrajectory:

    def __init__(self, model, variables):
        self.model = model
        self.time = []
        self.variables = {
            variable: []
            for variable in QG_CORE_DIMENSIONS if variable in variables
        }

    def append(self, time, state):
        self.time.append(time)
        state = self.model.compute_dependent_variables(
            state,
            list(self.variables),
        )
        for variable in self.variables:
            self.variables[variable].append(state[variable])
        return state

    def to_numpy(self):
        return {
            variable: np.array(self.variables[variable])
            for variable in self.variables
        }

    def num_batch_dimensions(self):
        for variable in QG_CORE_DIMENSIONS:
            if variable in self.variables:
                if len(self.variables[variable]) == 0:
                    return 0
                return len(self.variables[variable][0].shape) - len(QG_CORE_DIMENSIONS[variable])

    def to_xarray(self):
        batch_dimensions = [f'batch_{i}' for i in range(self.num_batch_dimensions())]
        data_vars = {
            variable: (
                ['time']+batch_dimensions+QG_CORE_DIMENSIONS[variable],
                np.array(self.variables[variable])
            ) for variable in self.variables
        }
        coords = dict(
            time=('time', np.array(self.time)),
            lat=('lat', self.model.spectral_transformations.lats),
            lon=('lon', self.model.spectral_transformations.lons),
        )
        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
        )

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

    @staticmethod
    def model_state(spec_q):
        return dict(spec_q=spec_q)

    def model_trajectory(self, variables):
        return QGModelTrajectory(self, variables)

    def compute_dependent_variable(self, state, variable):

        if variable in state:
            return state

        elif variable == 'spec_total_q':
            spec_total_q = self.poisson_solver.q_to_total_q(state['spec_q'])
            return state | dict(spec_total_q=spec_total_q)

        elif variable == 'spec_psi':
            state = self.compute_dependent_variable(state, 'spec_total_q')
            spec_psi = self.poisson_solver.total_q_to_psi(state['spec_total_q'])
            return state | dict(spec_psi=spec_psi)

        elif variable == 'zeta':
            state = self.compute_dependent_variable(state, 'spec_psi')
            zeta = self.poisson_solver.psi_to_zeta(state['spec_psi'])
            return state | dict(zeta=zeta)

        elif variable == 'dq_dtheta':
            dq_dtheta = self.spectral_transformations.spec_to_grid_grad_theta(state['spec_q'])
            return state | dict(dq_dtheta=dq_dtheta)

        elif variable == 'dq_dphi':
            dq_dphi = self.spectral_transformations.spec_to_grid_grad_phi(state['spec_q'])
            return state | dict(dq_dphi=dq_dphi)

        elif variable == 'dpsi_dtheta':
            state = self.compute_dependent_variable(state, 'spec_psi')
            dpsi_dtheta = self.spectral_transformations.spec_to_grid_grad_theta(state['spec_psi'])
            return state | dict(dpsi_dtheta=dpsi_dtheta)

        elif variable == 'dpsi_dphi':
            state = self.compute_dependent_variable(state, 'spec_psi')
            dpsi_dphi = self.spectral_transformations.spec_to_grid_grad_phi(state['spec_psi'])
            return state | dict(dpsi_dphi=dpsi_dphi)

        elif variable == 'q':
            q = self.spectral_transformations.spec_to_grid(state['spec_q'])
            return state | dict(q=q)

        elif variable == 'psi':
            state = self.compute_dependent_variable(state, 'spec_psi')
            psi = self.spectral_transformations.spec_to_grid(state['spec_psi'])
            return state | dict(psi=psi)

    def compute_dependent_variables(self, state, variables):
        for variable in variables:
            state = self.compute_dependent_variable(state, variable)
        return state

    def compute_model_tendencies(self, state):
        # compute dependent variables
        state = self.compute_dependent_variables(state, (
            'spec_total_q',
            'spec_psi',
            'zeta',
            'dq_dtheta',
            'dq_dphi',
            'dpsi_dtheta',
            'dpsi_dphi',
        ))

        # compute Jacobian
        jacobian = (
            state['dq_dphi'] * state['dpsi_dtheta'] - 
            state['dq_dtheta'] * state['dpsi_dphi']
        )

        # compute forcing
        forcing = self.forcing.compute_forcing()

        # compute Ekman dissipation
        dissipation_ekman = self.dissipation.ekman.compute_ekman_dissipation(state)

        # compute selective dissipation
        spec_dissipation_selective = self.dissipation.selective.compute_selective_dissipation(state)

        # compute thermal dissipation
        spec_dissipation_thermal = self.dissipation.thermal.compute_thermal_dissipation(state)

        # aggregate contributions in grid space
        tendencies = jacobian + forcing - dissipation_ekman

        # aggregate contributions in spectral space
        spec_tendencies = -spec_dissipation_selective - spec_dissipation_thermal

        # return all contributions in spectral space
        spec_tendencies += self.spectral_transformations.grid_to_spec(tendencies)
        return self.model_state(spec_tendencies)

    def apply_euler_step(self, state, tendencies, step):
        return self.model_state(
            state['spec_q'] + step * tendencies['spec_q']
        )


