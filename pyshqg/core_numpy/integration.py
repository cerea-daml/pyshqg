
import numpy as np

# use steps=[0, 0.5, 0.5, 1] and w=[1, 2, 2, 1] for RK4
# use steps=[0] and w=[1] for EE
# use steps=[0, 0.5] and w=[0, 1] for RK2
# use steps=[0, 1] and w=[0.5, 0.5] for ABM
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
        self.steps = steps
        sum_w = sum(weights)
        self.weights = [w/sum_w for w in weights]

    def forward(
        self,
        state,
    ):
        averaged_tendencies = 0
        current_tendencies = 0
        for (w, step) in zip(self.weights, self.steps):
            current_state = state + current_tendencies * self.dt * step
            current_tendencies = self.model.compute_model_tendencies(current_state)
            averaged_tendencies += w * current_tendencies
        return state + averaged_tendencies * self.dt

