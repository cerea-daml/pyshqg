r"""Submodule dedicated to the QG model."""

import dataclasses

import numpy as np
import xarray as xr


class QGModelTrajectory:
    r"""Container class for a QG model trajectory.

    The purpose of this class is to make a container class
    for a model trajectory. In practice, for each recorded
    variable, a list is created. Each time a snapshot is 
    appended to the trajectory, using the `append` method,
    all the recorded variables are appended to the appropriate lists. 

    At any time, the trajectory can be exported to 'numpy' format
    (i.e. as a dictionnary of `str` to `numpy.ndarray`, with one entry 
    for each recorded variable) using the `to_numpy` method 
    or to 'xarray' format (i.e. as an `xarray.Dataset`, with one 
    `xarray.DataArray` for each recorded variable) using the
    `to_xarray` method.

    Attributes
    ----------
    core_dimensions : dict of str to list of str
        Mapping from potential variable name to list of core dimension names.
    model : pyshqg.core.model.QGModel
        Reference to the QG model instance.
    time : list of float
        List of the snapshot times.
    variables : dict of str to list of numpy.ndarray
        Mapping from recorded variable name to list of snapshots.
    """

    def __init__(self, model, variables):
        r"""Constructor for the QG model trajectory.

        Parameters
        ----------
        model : pyshqg.core.model.QGModel
            Reference to the QG model instance.
        variables : list of str
            List of variables to record.
        """
        self.core_dimensions = dict(
            spec_q=['level', 'c', 'l', 'm'],
            spec_total_q=['level', 'c', 'l', 'm'],
            spec_psi=['level', 'c', 'l', 'm'],
            zeta=['lat', 'lon'],
            q=['level', 'lat', 'lon'],
            psi=['level', 'lat', 'lon'],
        )
        self.model = model
        self.time = []
        self.variables = {
            variable: []
            for variable in self.core_dimensions if variable in variables
        }

    def append(self, time, state):
        r"""Creates a snapshot and adds it to the trajectory.

        If some variables to record are not yet available,
        they are computed and added to the state.
        The present method returns the state, so that the added
        variables can be used later on.

        Parameters
        ----------
        time : float
            Snapshot time.
        state : dict of str to backend array
            QG model state to record.

        Returns
        -------
        state : dict of str to backend array
            Recorded QG model state.
        """
        self.time.append(time)
        state = self.model.compute_dependent_variables(
            state,
            list(self.variables),
        )
        for variable in self.variables:
            self.variables[variable].append(
                self.model.backend.to_numpy(
                    state[variable]
                )
            )
        return state

    def to_numpy(self):
        r"""Exports the trajectory to 'numpy' format."""
        return {
            variable: np.array(self.variables[variable])
            for variable in self.variables
        }

    def _num_batch_dimensions(self):
        r"""Computes the number of batch dimensions."""
        for variable in self.core_dimensions:
            if variable in self.variables:
                if len(self.variables[variable]) == 0:
                    return 0
                return len(self.variables[variable][0].shape) - len(self.core_dimensions[variable])

    def to_xarray(self):
        r"""Exports the trajectory to 'xarray' format."""
        num_batch_dimensions = self._num_batch_dimensions()
        if num_batch_dimensions == 1:
            batch_dimensions = ['batch']
        else:
            batch_dimensions = [f'batch_{i}' for i in range(num_batch_dimensions)]
        data_vars = {
            variable: (
                ['time']+batch_dimensions+self.core_dimensions[variable],
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


@dataclasses.dataclass
class QGModel:
    r"""QG model class.

    The purpose of this class is to gather all
    the necessary pieces to implement the QG model.
    The main method of this class is `compute_model_tendencies`,
    which can be used to compute the tendencies of the QG model.

    Parameters
    ----------
    backend : pyshqg.backend.Backend object
        The backend.
    spectral_transformations : pyshqg.core.spectral_transformations.SpectralTransformations
        Object encapsulating the spectral transformations.
    poisson_solver : pyshqg.core.poisson.QGPoissonSolver
        Object encapsulating the Poisson solver.
    dissipation : pyshqg.core.dissipation.QGDissipation
        Object encapsulating the dissipation processes.
    forcing : pyshqg.core.source.QGForcing
        Object encapsulating the forcing therm.
    """
    backend: 'pyshqg.backend.Backend'
    spectral_transformations: 'pyshqg.core.spectral_transformations.SpectralTransformations'
    poisson_solver: 'pyshqg.core.poisson.QGPoissonSolver'
    dissipation: 'pyshqg.core.dissipation.QGDissipation'
    forcing: 'pyshqg.core.source.QGForcing'

    @staticmethod
    def model_state(spec_q):
        r"""Constructs a model state.

        Parameters
        ----------
        spec_q : backend array, shape (..., 2, T+1, T+1)
            Relative vorticity in spectral space $\hat{q}$.

        Returns
        -------
        state : dict of str to backend array
            QG model state as dictionary of variables.
        """
        return dict(spec_q=spec_q)

    def model_trajectory(self, variables):
        r"""Constructs and initialises a model trajectory.

        Parameters
        ----------
        variables : list of str
            List of variables to record.

        Returns
        -------
        trajectory : pyshqg.core.model.QGModelTrajectory
            Initialised trajectory object.
        """
        return QGModelTrajectory(self, variables)

    def compute_dependent_variable(self, state, variable):
        r"""Computes a dependent variable.

        The requested variable is computed and added
        to the state to be used later on. If the requested
        variable require another variable which is unavailable,
        the latter unavailable variable will be computed first.
        Also note that if the requested variable is 
        already available, then nothing happens.

        The present method currently supports the following variables:

        - the total vorticity in spectral space $\hat{q}^{\mathsf{t}}$;
        - the stream function in spectral space $\hat{\psi}$;
        - the drag in real space $\zeta$;
        - the spatial gradient of $\psi$ and $q$ in real space;
        - the relative vorticity in real space $q$;
        - the stream function in real space $\psi$.

        Parameters
        ----------
        state : dict of str to backend array
            Current QG model state.
        variable : str
            Requested variable.

        Returns
        -------
        state : dict of str to backend array
            Updated QG model state.
        """

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
        r"""Computes a list of dependent variable.

        The present method uses the `compute_dependent_variable`
        method to compute each requested variable in the list.

        Parameters
        ----------
        state : dict of str to backend array
            Current QG model state.
        variables : list of str
            List of requested variable.

        Returns
        -------
        state : dict of str to backend array
            Updated QG model state.
        """
        for variable in variables:
            state = self.compute_dependent_variable(state, variable)
        return state

    def compute_model_tendencies(self, state):
        r"""Computes the model tendencies.

        The model tendencies are computed only for the
        control variable of the QG model, i.e. the relative
        vorticity in spectral space $\hat{q}$. They are
        encapsulated in a model state dictionnary before
        they are returned.

        Parameters
        ----------
        state : dict of str to backend array
            Current QG model state.

        Returns
        -------
        tendencies : dict of str to backend array
            Tendencies associated to the QG model state.
        """
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
        dissipation_ekman = self.dissipation.ekman.compute_ekman_dissipation(
            state['zeta'],
            state['dpsi_dtheta'],
            state['dpsi_dphi'],
        )

        # compute selective dissipation
        spec_dissipation_selective = self.dissipation.selective.compute_selective_dissipation(
            state['spec_total_q'],
        )

        # compute thermal dissipation
        spec_dissipation_thermal = self.dissipation.thermal.compute_thermal_dissipation(
            state['spec_psi'],
        )

        # aggregate contributions in grid space
        tendencies = jacobian + forcing - dissipation_ekman

        # aggregate contributions in spectral space
        spec_tendencies = -spec_dissipation_selective - spec_dissipation_thermal

        # return all contributions in spectral space
        spec_tendencies += self.spectral_transformations.grid_to_spec(tendencies)
        return self.model_state(spec_tendencies)

    def apply_euler_step(self, state, tendencies, step):
        r"""Applies an explicit Euler integration step.

        The integration step is applied only to the
        control variable of the QG model, i.e. the relative
        vorticity in spectral space $\hat{q}$. 

        Parameters
        ----------
        state : dict of str to backend array
            Current QG model state.
        tendencies : dict of str to backend array
            QG model tendencies to apply.
        step : float
            Step size.

        Returns
        -------
        state : dict of str to backend array
            Updated QG model state.

        Notes
        -----
        This algorithmic step is at the core of every
        explicit integration scheme, e.g. the Runge--Kutta
        schemes as used in 
        `pyshqg.core.integration.RungeKuttaModelIntegrator`.
        """
        return self.model_state(
            state['spec_q'] + step * tendencies['spec_q']
        )


