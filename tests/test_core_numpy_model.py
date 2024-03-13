import numpy as np
import pytest

from pyshqg.preprocessing.reference_data import load_test_data
from pyshqg.backend.numpy_backend import NumpyBackend as Backend
from pyshqg.core.constructors import construct_model, construct_integrator

def assert_close(
    x, 
    y, 
    all_atol=0,
    all_rtol=1e-7,
    rms_atol=0, 
    rms_rtol=1e-7,
):
    # point-wise statistics
    all_ref = abs(y).max()
    all_abs = abs(x-y).max()
    np.seterr(invalid='ignore')
    all_rel = np.nanmax(np.divide(abs(x-y), abs(y)))
    print('point-wise statistics')
    print('reference:          ', all_ref)
    print('asbolute difference:', all_abs)
    print('relative difference:', all_rel)
    # aggregated statistics
    rms_ref = np.sqrt(np.mean(np.square(y)))
    rms_abs = np.sqrt(np.mean(np.square(x-y)))
    rms_rel = rms_abs / rms_ref
    print('root-mean-squared statistics')
    print('reference:          ', rms_ref)
    print('asbolute difference:', rms_abs)
    print('relative difference:', rms_rel)
    # assertions
    if all_atol is not None and all_rtol is not None:
        np.testing.assert_allclose(x, y, atol=all_atol, rtol=all_rtol)
    if rms_atol is not None and rms_rtol is not None:
        assert rms_abs < rms_atol + rms_rtol * rms_ref

@pytest.fixture(params=[
    dict(
        internal_truncature=21,
        grid_truncature=31,
    ),
    dict(
        internal_truncature=42,
        grid_truncature=63,
    ),
])
def config(request):
    return request.param

@pytest.fixture
def ds_test(config):
    return load_test_data(**config)

@pytest.fixture
def backend():
    return Backend('float64')

@pytest.fixture
def model(backend, ds_test):
    return construct_model(backend, ds_test.config)

@pytest.fixture(params=['ee', 'abm', 'rk2', 'rk4'])
def integrator_name(request):
    return request.param

@pytest.fixture
def integrator(ds_test, model, integrator_name):
    section = f'{integrator_name}_integration'
    return construct_integrator(ds_test.config[section], model)

def test_config(config):
    print(config)

def test_total_q(backend, ds_test, model):
    spec_q = backend.from_numpy(ds_test.spec_q.to_numpy())
    spec_total_q = model.poisson_solver.q_to_total_q(spec_q)
    assert_close(
        backend.to_numpy(spec_total_q),
        ds_test.spec_total_q.to_numpy(),
    )

def test_psi(backend, ds_test, model):
    spec_total_q = backend.from_numpy(ds_test.spec_total_q.to_numpy())
    spec_psi = model.poisson_solver.total_q_to_psi(spec_total_q)
    assert_close(
        backend.to_numpy(spec_psi),
        ds_test.spec_psi.to_numpy(),
    )

def test_zeta(backend, ds_test, model):
    spec_psi = backend.from_numpy(ds_test.spec_psi.to_numpy())
    zeta = model.poisson_solver.psi_to_zeta(spec_psi)
    assert_close(
        backend.to_numpy(zeta),
        ds_test.zeta.to_numpy(),
    )

def test_jacobian(backend, ds_test, model):
    spec_q = backend.from_numpy(ds_test.spec_q.to_numpy())
    spec_psi = backend.from_numpy(ds_test.spec_psi.to_numpy())
    d_q_d_theta = model.spectral_transformations.spec_to_grid_grad_theta(spec_q)
    d_q_d_phi = model.spectral_transformations.spec_to_grid_grad_phi(spec_q)
    d_psi_d_theta = model.spectral_transformations.spec_to_grid_grad_theta(spec_psi)
    d_psi_d_phi = model.spectral_transformations.spec_to_grid_grad_phi(spec_psi)
    jacobian = (
        d_q_d_phi * d_psi_d_theta - 
        d_q_d_theta * d_psi_d_phi
    )
    assert_close(
        backend.to_numpy(jacobian),
        ds_test.jacobian.to_numpy(),
    )

# this test accuracy is not very good :(
def test_forcing(backend, ds_test, model):
    forcing = model.forcing.compute_forcing()
    assert_close(
        backend.to_numpy(forcing),
        ds_test.forcing.to_numpy(),
        all_rtol=None,
        rms_rtol=1e-6,
    )

def test_dissipation_ekman(backend, ds_test, model):
    zeta = backend.from_numpy(ds_test.zeta.to_numpy())
    spec_q = backend.from_numpy(ds_test.spec_q.to_numpy())
    spec_psi = backend.from_numpy(ds_test.spec_psi.to_numpy())
    dq_dtheta = model.spectral_transformations.spec_to_grid_grad_theta(spec_q)
    dq_dphi = model.spectral_transformations.spec_to_grid_grad_phi(spec_q)
    dpsi_dtheta = model.spectral_transformations.spec_to_grid_grad_theta(spec_psi)
    dpsi_dphi = model.spectral_transformations.spec_to_grid_grad_phi(spec_psi)
    dissipation_ekman = model.dissipation.ekman.compute_ekman_dissipation(
        zeta,
        dpsi_dtheta,
        dpsi_dphi,
    )
    assert_close(
        backend.to_numpy(dissipation_ekman),
        ds_test.dissip_ekman.to_numpy(),
    )

def test_dissipation_selective(backend, ds_test, model):
    spec_total_q = backend.from_numpy(ds_test.spec_total_q.to_numpy())
    spec_dissipation_selective = model.dissipation.selective.compute_selective_dissipation(
        spec_total_q,
    )
    assert_close(
        backend.to_numpy(spec_dissipation_selective),
        ds_test.spec_dissip_sel.to_numpy(),
    )

# this test accuracy is not very good :(
def test_dissipation_thermal(backend, ds_test, model):
    spec_psi = backend.from_numpy(ds_test.spec_psi.to_numpy())
    spec_dissipation_thermal = model.dissipation.thermal.compute_thermal_dissipation(
        spec_psi,
    )
    assert_close(
        backend.to_numpy(spec_dissipation_thermal),
        ds_test.spec_dissip_therm.to_numpy(),
        all_rtol=None,
    )

def test_grid_dq_dt(backend, ds_test, model):
    # 1: jacobian
    spec_q = backend.from_numpy(ds_test.spec_q.to_numpy())
    spec_psi = backend.from_numpy(ds_test.spec_psi.to_numpy())
    zeta = backend.from_numpy(ds_test.zeta.to_numpy())
    dq_dtheta = model.spectral_transformations.spec_to_grid_grad_theta(spec_q)
    dq_dphi = model.spectral_transformations.spec_to_grid_grad_phi(spec_q)
    dpsi_dtheta = model.spectral_transformations.spec_to_grid_grad_theta(spec_psi)
    dpsi_dphi = model.spectral_transformations.spec_to_grid_grad_phi(spec_psi)
    jacobian = (
        dq_dphi * dpsi_dtheta - 
        dq_dtheta * dpsi_dphi
    )

    # 2: forcing (replace by test data forcing)
    # otherwise, accuracy is too low...
    forcing = model.forcing.compute_forcing()
    forcing = backend.from_numpy(ds_test.forcing.to_numpy())

    # 3: Ekman dissipation
    dissipation_ekman = model.dissipation.ekman.compute_ekman_dissipation(
        zeta,
        dpsi_dtheta,
        dpsi_dphi,
    )
    
    # aggregate
    grid_dq_dt = jacobian + forcing - dissipation_ekman
    assert_close(
        backend.to_numpy(grid_dq_dt),
        ds_test.grid_dq_dt.to_numpy(),
    )

def test_grid_to_spec_dq_dt(backend, ds_test, model):
    grid_dq_dt = backend.from_numpy(ds_test.grid_dq_dt.to_numpy())
    grid_to_spec_dq_dt = model.spectral_transformations.grid_to_spec(grid_dq_dt)
    assert_close(
        backend.to_numpy(grid_to_spec_dq_dt),
        ds_test.grid_to_spec_dq_dt.to_numpy(),
    )

def test_spec_dq_dt(backend, ds_test, model):
    # 1: selective dissipation
    spec_total_q = backend.from_numpy(ds_test.spec_total_q.to_numpy())
    spec_dissipation_selective = model.dissipation.selective.compute_selective_dissipation(
        spec_total_q,
    )

    # 2: thermal dissipation (accuracy is low...)
    spec_psi = backend.from_numpy(ds_test.spec_psi.to_numpy())
    spec_dissipation_thermal = model.dissipation.thermal.compute_thermal_dissipation(
        spec_psi,
    )

    # aggregate
    spec_dq_dt = -spec_dissipation_selective - spec_dissipation_thermal
    assert_close(
        backend.to_numpy(spec_dq_dt),
        ds_test.spec_dq_dt.to_numpy(),
        all_rtol=None,
    )

def test_tendencies(backend, ds_test, model):
    spec_q = backend.from_numpy(ds_test.spec_q.to_numpy())
    spec_tendencies = model.compute_model_tendencies(model.model_state(spec_q))
    assert_close(
        backend.to_numpy(spec_tendencies['spec_q']),
        ds_test.spec_tendencies.to_numpy(),
        all_rtol=None,
    )

def test_integration(backend, ds_test, model, integrator_name, integrator):
    spec_q = backend.from_numpy(ds_test.spec_q.to_numpy())
    spec_integrated = integrator.forward(model.model_state(spec_q))
    assert_close(
        backend.to_numpy(spec_integrated['spec_q']),
        ds_test[f'spec_{integrator_name}'].to_numpy(),
        all_rtol=None,
    )

