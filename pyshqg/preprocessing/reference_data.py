r"""Submodule dedicated to the reference data."""

import pathlib

import numpy as np
import xarray as xr


def _data_file(name):
    r"""Returns the full data file name.

    Parameters
    ----------
    name : str
        The name of the file.

    Returns
    -------
    full_name : pathlib.Path
        The full name of the file.
    """
    this_file = pathlib.Path(__file__).absolute()
    return this_file.parent.parent / f'data/{name}.zarr'

def _open_zarr_data(name, load):
    r"""Opens the zarr archive.

    Parameters
    ----------
    name : str
        The name of the file.
    load : bool
        Whether to load data in memory.

    Returns
    -------
    ds : xarray.Dataset
        The opened dataset.
    """
    ds = xr.open_zarr(_data_file(name))
    if load:
        ds = ds.load()
    return ds
    
def _add_padding(x):
    r"""Adds padding to a numpy array.

    Parameters
    ----------
    x : np.ndarray, shape (..., Nlat, Nlon)
        Array defined in the grid.

    Returns
    -------
    x : np.ndarray, shape (..., Nlat+2, Nlon)
        Padded array in the grid.
    """
    shape = list(x.shape)
    shape[-2] +=2
    augmented_x = np.zeros(shape)
    
    mean = x[..., 0, :].mean(axis=-1)
    mean = np.repeat(mean[..., np.newaxis], shape[-1], axis=-1)
    augmented_x[..., 0, :] = mean
    
    augmented_x[..., 1:-1, :] = x 

    mean = x[..., -1, :].mean(axis=-1)
    mean = np.repeat(mean[..., np.newaxis], shape[-1], axis=-1)
    augmented_x[..., -1, :] = mean
    return augmented_x

def load_reference_data(
    grid_truncature, 
    padding=True,
    load=True,
):
    r"""Loads the reference data.

    Parameters
    ----------
    grid_truncature : int
        The grid truncature of the data to load.
    padding : bool
        Whether to add padding to the data.
    load : bool
        Whether to load data into memory.

    Returns
    -------
    ds : xarray.Dataset
        The reference data.
    """
    ds = _open_zarr_data(
        f'data_t{grid_truncature}',
        load,
    )
    if padding:
        augmented_lat = np.array([-90]+list(ds.lat.to_numpy())+[90])
        ds = xr.apply_ufunc(
            _add_padding,
            ds,
            input_core_dims=(('lat', 'lon'),),
            output_core_dims=(('augmented_lat', 'lon'),),
            dask='parallelized'
        ).assign_coords(
            augmented_lat=augmented_lat,
        ).rename(dict(
            augmented_lat='lat',
        ))
    return ds


def interpolate_data(ds, lat, lon, methods):
    r"""Interpolates the reference data.

    Parameters
    ----------
    ds : xarray.Dataset
        The reference data.
    lat : numpy.ndarray, shape (Nlat,)
        The latitude nodes.
    lon : numpy.ndarray, shape (Nlon,)
        The longitude nodes.
    methods : dict of str to str
        For each variable, the interpolation method to use.

    Returns
    -------
    ds : xarray.Dataset
        The interpolated reference data.
    """
    interpolated = []
    for var in methods:
        if methods[var] == 'linear':
            interpolated.append(
                ds[var].interp(
                    lon=lon, 
                    lat=lat, 
                    method='linear',
                )
            )
        if methods[var] == 'quintic':
            interpolated.append(
                ds[var].interp(
                    lon=lon, 
                    lat=lat, 
                    method='polynomial',
                    kwargs=dict(
                        order=5,
                    ),
                )
            )
    return xr.merge(interpolated)

def load_test_data(
	internal_truncature,
    grid_truncature,
    load=True,
):
    r"""Loads the test data.

    Parameters
    ----------
    internal_truncature : int
        The internal truncature of the test data.
    grid_truncature : int
        The grid truncature of the test data.
    load : bool
        Whether to load data into memory.

    Returns
    -------
    ds : xarray.Dataset
        The test data.
    """
    return _open_zarr_data(
        f'test_t{internal_truncature}_t{grid_truncature}',
        load,
    )
    
