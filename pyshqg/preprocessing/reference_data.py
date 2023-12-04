import numpy as np
import xarray as xr
import pathlib

def data_file(name):
    this_file = pathlib.Path(__file__).absolute()
    return this_file.parent.parent / f'data/{name}.zarr'

def open_zarr_data(name, load):
    ds = xr.open_zarr(data_file(name))
    if load:
        ds = ds.load()
    return ds
    
def add_padding(x):
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
    load=True
):
    ds = open_zarr_data(
        f'data_t{grid_truncature}',
        load,
    )
    if padding:
        augmented_lat = np.array([-90]+list(ds.lat.to_numpy())+[90])
        ds = xr.apply_ufunc(
            add_padding,
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
    return open_zarr_data(
        f'test_t{internal_truncature}_t{grid_truncature}',
        load,
    )
    