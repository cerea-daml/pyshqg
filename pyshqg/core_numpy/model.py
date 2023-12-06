import numpy as np
import xarray as xr

class QGModelState:

    def __init__(self, spec_q):
        self.spec_q = spec_q
        self.spec_total_q = None
        self.spec_psi = None
        self.zeta = None
        self.gradients = None
        self.has_dependent_variables = False
        self.extra_variables = dict()

    def __add__(self, other):
        if isinstance(other, QGModelState):
            return QGModelState(self.spec_q+other.spec_q)
        else:
            return QGModelState(self.spec_q+other)

    def __rmul__(self, other):
        return QGModelState(self.spec_q*other)

    def __mul__(self, other):
        return QGModelState(self.spec_q*other)

class QGModelTrajectory:

    def __init__(self, model, variables):
        self.model = model
        self.time = []
        self.variables = {
            var: []
            for var in (
                'spec_q',
                'spec_total_q',
                'spec_psi',
                'zeta',
                'q',
                'psi',
            ) if var in variables
        }
        self.core_dimensions = dict(
            spec_q=['level', 'c', 'l', 'm'],
            spec_total_q=['level', 'c', 'l', 'm'],
            spec_psi=['level', 'c', 'l', 'm'],
            zeta=['lat', 'lon'],
            q=['level', 'lat', 'lon'],
            psi=['level', 'lat', 'lon'],
        )

    def append(self, time, state):
        self.time.append(time)
        for var in self.variables:
            if var == 'spec_q':
                self.variables[var].append(state.spec_q)
            elif var in (
                'spec_total_q',
                'spec_psi',
                'zeta',
            ):
                self.model.compute_dependent_variables(state)
                self.variables[var].append(
                    state.__getattribute__(var)
                )
            elif var in (
                'q',
                'psi',
            ):
                self.model.compute_extra_variable(state, var)
                self.variables[var].append(
                    state.extra_variables[var]
                )

    def to_numpy(self):
        return {
            var: np.array(self.variables[var])
            for var in self.variables
        }

    def num_batch_dimensions(self):
        for var in (
            'spec_q',
            'spec_total_q',
            'spec_psi',
            'zeta',
        ):
            if var in self.variables:
                if len(self.variables[var]) == 0:
                    return 0
                return len(self.variables[var][0].shape) - len(self.core_dimensions[var])

    def to_xarray(self):
        batch_dimensions = [f'batch_{i}' for i in range(self.num_batch_dimensions())]
        data_vars = {
            var: (
                ['time']+batch_dimensions+self.core_dimensions[var],
                np.array(self.variables[var])
            ) for var in self.variables
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
        return QGModelState(spec_q)

    def model_trajectory(self, variables):
        return QGModelTrajectory(self, variables)

    def compute_dependent_variables(self, state):
        if state.has_dependent_variables:
            return

        # compute total q
        state.spec_total_q = self.poisson_solver.q_to_total_q(state.spec_q)
        
        # compute psi
        state.spec_psi = self.poisson_solver.total_q_to_psi(state.spec_total_q)

        # compute zeta
        state.zeta = self.poisson_solver.psi_to_zeta(state.spec_psi)

        # compute spatial gradients
        state.gradients = dict(
            dq_dtheta=self.spectral_transformations.spec_to_grid_grad_theta(state.spec_q),
            dq_dphi=self.spectral_transformations.spec_to_grid_grad_phi(state.spec_q),
            dpsi_dtheta=self.spectral_transformations.spec_to_grid_grad_theta(state.spec_psi),
            dpsi_dphi=self.spectral_transformations.spec_to_grid_grad_phi(state.spec_psi),
        )

        state.has_dependent_variables = True

    def compute_extra_variable(self, state, var):
        if var in state.extra_variables:
            return

        if var == 'q':
            q = self.spectral_transformations.spec_to_grid(state.spec_q)
            state.extra_variables['q'] = q

        elif var == 'psi':
            self.compute_dependent_variables(state)
            psi = self.spectral_transformations.spec_to_grid(state.spec_psi)
            state.extra_variables['psi'] = psi

    def compute_model_tendencies(self, state):
        # compute dependent variables
        self.compute_dependent_variables(state)

        # compute Jacobian
        jacobian = (
            state.gradients['dq_dphi'] * state.gradients['dpsi_dtheta'] - 
            state.gradients['dq_dtheta'] * state.gradients['dpsi_dphi']
        )

        # compute forcing
        forcing = self.forcing.compute_forcing()

        # compute Ekman dissipation
        dissipation_ekman = self.dissipation.ekman.compute_ekman_dissipation(
            state.zeta,
            state.gradients,
        )

        # compute selective dissipation
        spec_dissipation_selective = self.dissipation.selective.compute_selective_dissipation(
            state.spec_total_q
        )

        # compute thermal dissipation
        spec_dissipation_thermal = self.dissipation.thermal.compute_thermal_dissipation(
            state.spec_psi
        )

        # aggregate contributions in grid space
        tendencies = jacobian + forcing - dissipation_ekman

        # aggregate contributions in spectral space
        spec_tendencies = -spec_dissipation_selective - spec_dissipation_thermal

        # return all contributions in spectral space
        return QGModelState(spec_tendencies + self.spectral_transformations.grid_to_spec(tendencies))

