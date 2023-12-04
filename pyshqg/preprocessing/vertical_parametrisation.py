import numpy as np

class VerticalParametrisation:

    def __init__(self, rossby_radius_list):
        self.rossby_radius_list = rossby_radius_list
        self.num_levels = len(rossby_radius_list) + 1

    def precompute_coupling_matrix(self, scaling=1):
        coupling_matrix = np.zeros((self.num_levels, self.num_levels))
        for (z, rossby_radius) in enumerate(self.rossby_radius_list):
            coupling = 1 / rossby_radius**2
            coupling_matrix[z:z+2, z:z+2] -= coupling * np.array([[1, -1], [-1, 1]])
        return scaling*coupling_matrix