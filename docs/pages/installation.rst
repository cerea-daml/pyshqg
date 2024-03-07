.. _installation:

Installation
============

Requirements
------------

``pyshqg`` provides two implementations of the QG model: 
the first based on ``numpy`` and the second based on ``tensorflow``.

The ``numpy``-based variant depends on:

- Python;
- `NumPy <http://www.numpy.org>`_ for array support;
- `pyshtools <https://shtools.github.io/SHTOOLS/index.html>`_ for the spherical harmonics coefficients;
- `Xarray <https://xarray.dev>`_ for dataset support;
- `Zarr <https://zarr.readthedocs.io>`_ for dataset storage;
- `tqdm <https://tqdm.github.io>`_ for the progress bar.

The ``tensorflow``-based variant additionally depends on

- `TensorFlow <https://www.tensorflow.org>`_ for tensor support.

Instructions
------------

For the ``numpy``-based variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install ``pyshqg`` is to use ``pip``.
This will also install all the dependencies for the ``numpy``-based variant.

.. code-block:: bash

    $ pip install pyshqg

For the ``tensorflow``-based variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Iy you need support for the ``tensorflow``-based variant,
you just need to have ``tensorflow`` available on top of what
``pip`` has installed. For example, you can follow the instructions
`here <https://www.tensorflow.org/install>`_.

