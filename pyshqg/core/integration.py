r"""Submodule dedicated to integration schemes."""

import numpy as np


class RungeKuttaModelIntegrator:
    r"""Runge--Kutta integration class.

    The purpose of this class is to implement
    Runge--Kutta integration schemes.

    Attributes
    ----------
    model : pyshqg.core_numpy.model.QGModel
        The model to integrate.
    dt : float
        The integration time step.
    steps : list of float
        The scheme integration sub time steps.
    weights : list of float
        The weights of each integration sub time step.
    """

    def __init__(
        self,
        model,
        dt,
        method,
    ):
        r"""Constructor for the Runge--Kutta integration class.

        The list of integration sub time steps and the associated weights
        are constructed for each available schemes. The list of
        currently available schemes is:
        - RK4;
        - RK2;
        - ABM;
        - EE.

        Parameters
        ----------
        model : pyshqg.core_numpy.model.QGModel
            The model to integrate.
        dt : float
            The integration time step.
        method : str
            The method name.
        """
        self.model = model
        self.dt = dt
        match method.lower():
            case 'rk4':
                self.steps = [0, 0.5*dt, 0.5*dt, dt]
                self.weights = [w/6 for w in [1, 2, 2, 1]]
            case 'rk2':
                self.steps = [0, 0.5*dt]
                self.weights = [0, 1]
            case 'abm':
                self.steps = [0, dt]
                self.weights = [0.5, 0.5]
            case 'ee':
                self.steps = [0]
                self.weights = [1]

    def forward(
        self,
        state,
    ):
        r"""Applies an integration step forward in time.

        Parameters
        ----------
        state : dict of str to numpy.ndarray
            The current QG model state.

        Returns
        -------
        state : dict of str to numpy.ndarray
            The integrated QG model state.
        """
        averaged_tendencies = self.model.model_state(0)
        current_tendencies = self.model.model_state(0)
        for (w, step) in zip(self.weights, self.steps):
            current_state = self.model.apply_euler_step(
                state,
                current_tendencies,
                step,
            )
            current_tendencies = self.model.compute_model_tendencies(current_state)
            averaged_tendencies = self.model.apply_euler_step(
                averaged_tendencies,
                current_tendencies,
                w,
            )
        return self.model.apply_euler_step(
            state,
            averaged_tendencies,
            self.dt,
        )

    def run(
        self,
        state,
        t_start,
        num_snapshots,
        num_steps_per_snapshot,
        variables,
        use_tqdm=True,
    ):
        r"""Computes a full trajectory.

        Parameters
        ----------
        state : dict of str to numpy.ndarray
            The initial QG model state.
        t_start : float
            The time of the initial state.
        num_snapshots : int
            Number of snapshots to compute.
        num_steps_per_snapshot : int
            Number of integration steps between snapshots.
        variables : list of str
            List of variables to record.
        use_tqdm : bool
            Whether to use `tqdm` for the main integration loop.

        Returns
        -------
        trajectory : pyshqg.core_numpy.model.QGModelTrajectory
            The model trajectory.
        """
        if use_tqdm:
            import tqdm.auto as tqdm
            main_range = tqdm.trange(
                num_snapshots, 
                desc='model integration',
            ) 
        else:
            main_range = range(num_snapshots)
        time = t_start + np.arange(num_snapshots+1) * self.dt * num_steps_per_snapshot
        trajectory = self.model.model_trajectory(variables)
        for t in main_range:
            state = trajectory.append(time[t], state)
            for _ in range(num_steps_per_snapshot):
                state = self.forward(state)
        state = trajectory.append(time[-1], state)
        return trajectory

