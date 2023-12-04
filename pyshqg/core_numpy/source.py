class QGForcing:

    def __init__(
        self,
        forcing,
    ):
        self.forcing = forcing

    def compute_forcing(self):
        return self.forcing