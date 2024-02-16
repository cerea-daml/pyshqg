r"""Submodule dedicated to constructors."""

from pyshqg.preprocessing.reference_data import load_reference_data, interpolate_data
from pyshqg.preprocessing.reference_data import load_test_data
from pyshqg.preprocessing.vertical_parametrisation import VerticalParametrisation
from pyshqg.preprocessing.orography import Orography
from pyshqg.core_tensorflow.spectral_transformations import SpectralTransformations
from pyshqg.core_tensorflow.poisson import QGPoissonSolver
from pyshqg.core_tensorflow.dissipation import QGEkmanDissipation, QGSelectiveDissipation
from pyshqg.core_tensorflow.dissipation import QGThermalDissipation, QGDissipation
from pyshqg.core_tensorflow.source import QGForcing
from pyshqg.core_tensorflow.model import QGModel
from pyshqg.core_tensorflow.integration import RungeKuttaModelIntegrator

def construct_model(config):
    r"""Constructs a model from a given configuration.

    In practice, this function constructs:
    1. the spectral transformations (section 'spectral_transformations');
    2. the vertical parametrisation (section 'vertical_parametrisation');
    3. the orography;
    4. the Poisson solver (section 'poisson_solver');
    5. the dissipation processes (section 'dissipation');
    6. the forcing processes (section 'forcing').
    The orography and forcing processes require input data,
    which is taken from the reference dataset as specified in section
    'reference_data', and then interpolated as specified in section
    'data_interpolation'.

    Parameters
    ----------
    config : dict
        The configuration.

    Returns
    -------
    model : pyshqg.core_tensorflow.model.QGModel
        The constructed model.
    """
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

def construct_integrator(config, model):
    r"""Constructs a Runge--Kutta integrator for a given model.

    Parameters
    ----------
    config : dict
        The integrator parametrisation.
    model : pyshqg.core_tensorflow.model.QGModel
        The model to integrate.

    Returns
    -------
    integrator : pyshqg.core_tensorflow.integration.RungeKuttaModelIntegrator
        The constructed integrator.
    """
    return RungeKuttaModelIntegrator(
        model,
        dt=config['dt'],
        method=config['method'],
    )

