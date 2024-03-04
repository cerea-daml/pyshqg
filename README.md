
![Animated example](http://cerea.enpc.fr/HomePages/farchia/pyshqg-map.gif)

# pyshqg

[![Documentation Status](https://readthedocs.org/projects/pyshqg/badge/?version=latest)](https://pyshqg.readthedocs.io/en/latest/?badge=latest)

`pyshqg` is a python solver for the [Marshal and Molteni (1993) quasi-geostrophic (QG) model](https://doi.org/10.1175/1520-0469(1993)050%3C1792:TADUOP%3E2.0.CO;2).
QG models express the conservation of potential vorticity over time and are
meant to describe the large-scale circulation of the atmosphere under specific hypotheses.
This QG model is very special, because it is expressed in spherical harmonics and
because it encapsulates complex physical processes.

- [Documentation](https://pyshqg.readthedocs.io)
- [Source code](https://github.com/cerea-daml/pyshqg)
- [Issue tracker](https://github.com/cerea-daml/pyshqg/issues)

## Installation

Install using pip:

    $ pip install pyshqg

More details can be found on this [page](https://pyshqg.readthedocs.io/en/latest/pages/installation.html).

## Usage

Here is a sneak peak at how to use the package:

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

More details can be found on this [page](https://pyshqg.readthedocs.io/en/latest/pages/examples.html).

## Aknowledgements

This python package is based on an original implementation of the model
in Fortran written at LMD by XXX.

## Todo-list

    - make more example notebooks
    - fill in the documentation sections
    - choose tensorflow precision
    - fix poetry things
    - publish first version
    - write pytorch/jax modules

