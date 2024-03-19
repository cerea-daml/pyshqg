.. _installation:

Installation
============

Requirements
------------

``pyshqg`` provides two implementations of the QG model: 
the first based on NumPy_ and the second based on TensorFlow_.

The NumPy_-based variant depends on:

- Python;
- NumPy_: for array support;
- pyshtools_: for the spherical harmonics coefficients;
- Xarray_: for dataset support;
- Zarr_: for dataset storage;
- tqdm_: for the progress bar.

The TensorFlow_-based variant additionally depends on

- TensorFlow_: for tensor support.

Instructions
------------

For the NumPy-based variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install ``pyshqg`` is to use ``pip``.
This will also install all the dependencies for the NumPy_-based variant.

.. code-block:: bash

    $ pip install pyshqg

Alternatively, ``pyshqg`` is available on conda-forge and hence
can be installed using ``conda`` or ``mamba``:

.. code-block:: bash

    $ conda install conda-forge::pyshqg

For the TensorFlow-based variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Iy you need support for the TensorFlow_-based variant,
you just need to have TensorFlow_ available on top of what
``pip`` has installed. For example, you can follow the instructions
`here <https://www.tensorflow.org/install>`_.

.. _NumPy: http://www.numpy.org
.. _pyshtools: https://shtools.github.io/SHTOOLS/index.html
.. _Xarray: https://xarray.dev
.. _Zarr: https://zarr.readthedocs.io
.. _tqdm: https://tqdm.github.io
.. _TensorFlow: https://www.tensorflow.org

