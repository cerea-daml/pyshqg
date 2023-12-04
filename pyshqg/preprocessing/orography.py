import numpy as np

class Orography:

    def __init__(
        self,
        land_sea_mask,
        orography,
    ):
        self.land_sea_mask = land_sea_mask
        self.orography = orography

    def precorrect_planet_vorticity(
        self,
        f,
        orography_scale,
    ):
        f[-1] *= 1 + self.orography / orography_scale
        return f

    def precompute_ekman_friction(
        self,
        weight_land_sea_mask,
        weight_orography,
        orography_scale,
        tau,
    ):
        return ( 
            1 + 
            weight_land_sea_mask * self.land_sea_mask +
            weight_orography * ( 
                1 -
                np.exp(-np.maximum(
                    0, 
                    self.orography/orography_scale,
                ))
            )
        )/tau
