# 2019-pyshqg

`pyshqg` is a python solver for the Marshal and Molteni (1993) quasi-geostrophic model.

## Installation

Install using pip:

    $ pip install pyshqg

## Usage

    >>> import numpy as np
    >>>
    >>> from pyshqg.core_numpy.constructors import construct_model
    >>> from pyshqg.preprocessing.reference_data import load_test_data
    >>>
    >>> ds_test = load_test_data(internal_truncature=21, grid_truncature=31)
    >>> model = construct_model(ds_test.config)
    >>>
    >>> spec_tendencies = model.compute_model_tendencies(ds_test.spec_q.to_numpy())
    >>> rms_ref = np.sqrt(np.mean(np.square(ds_test.spec_tendencies.to_numpy())))
    >>> rms_diff = np.sqrt(np.mean(np.square(
    >>>     spec_tendencies,
    >>>     ds_test.spec_tendencies.to_numpy(),
    >>> )))
    >>> assert rms_diff < 1e-6 * rms_ref

## Aknowledgements

This python package is based on an original software produced at LMD.

## Todo-list

    - write docstrings
    - write documentation with sphinx + readthedocs
    - make a few examples notebooks
    - write visualisation module
    - write tensorflow/pytorch/jax modules

