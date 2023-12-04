from pyshqg.preprocessing.reference_data import load_reference_data, interpolate_data
from pyshqg.preprocessing.reference_data import load_test_data
from pyshqg.preprocessing.vertical_parametrisation import VerticalParametrisation
from pyshqg.preprocessing.orography import Orography
from pyshqg.core_numpy.spectral_transformations import SpectralTransformations
from pyshqg.core_numpy.poisson import QGPoissonSolver
from pyshqg.core_numpy.dissipation import QGEkmanDissipation, QGSelectiveDissipation
from pyshqg.core_numpy.dissipation import QGThermalDissipation, QGDissipation
from pyshqg.core_numpy.source import QGForcing
from pyshqg.core_numpy.model import QGModel

def construct_model(config):
    # spectral transformations
    spectral_transformations = SpectralTransformations(**config['spectral_transformations'])

    # data
    ds_reference = load_reference_data(**config['reference_data'])
    ds_interpolated = interpolate_data(
        ds=ds_reference,
        lat=spectral_transformations.lats,
        lon=spectral_transformations.lons,
        methods=config['data_interpolation'],
    )

    # vertical parametrisation
    vertical_parametrisation = VerticalParametrisation(**config['vertical_parametrisation'])

    # orography
    orography = Orography(
        land_sea_mask=ds_interpolated.land_sea_mask.to_numpy(),
        orography=ds_interpolated.orography.to_numpy(),
    )

    # poisson solver
    poisson_solver = QGPoissonSolver(
        spectral_transformations=spectral_transformations,
        vertical_parametrisation=vertical_parametrisation,
        orography=orography,
        **config['poisson_solver'],
    )

    # dissipation
    dissipation = QGDissipation(
        ekman=QGEkmanDissipation(
            spectral_transformations=spectral_transformations,
            orography=orography,
            **config['dissipation']['ekman'],
        ),
        selective=QGSelectiveDissipation(
            spectral_transformations=spectral_transformations,
            **config['dissipation']['selective'],
        ),
        thermal=QGThermalDissipation(
            vertical_parametrisation=vertical_parametrisation,
            **config['dissipation']['thermal'],
        ),
    )

    # forcing
    forcing = QGForcing(
        forcing=ds_interpolated.forcing.to_numpy(),
    )

    # model
    model = QGModel(
        spectral_transformations=spectral_transformations,
        poisson_solver=poisson_solver,
        dissipation=dissipation,
        forcing=forcing,
    )

    return model


