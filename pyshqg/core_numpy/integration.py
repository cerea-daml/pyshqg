
import numpy as np

# use steps=[0, 0.5, 0.5, 1] and w=[1, 2, 2, 1] for RK4
# use steps=[0] and w=[1] for EE
# use steps=[0, 0.5] and w=[0, 1] for RK2
# use steps=[0, 1] and w=[0.5, 0.5] for ABM
# note that steps[0] is always 0
class RungeKuttaModelIntegrator:

    def __init__(
        self,
        model,
        dt,
        steps,
        weights,
    ):
        self.model = model
        self.dt = dt
        self.steps = [dt*step for step in steps]
        sum_w = sum(weights)
        self.weights = [w/sum_w for w in weights]

    def forward(
        self,
        state,
    ):
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
        output='xarray',
    ):
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
        if output == 'numpy':
            return trajectory.to_numpy()
        elif output == 'xarray':
            return trajectory.to_xarray()
        else:
            return trajectory

