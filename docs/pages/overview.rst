.. _overview:

Package overview
================

The goal of ``pyshqg`` is to provide an implementation of the QG model.
It is divided into three sub-packages:

1. :ref:`pyshqg.preprocessing <api.preprocessing>` provides functionalities related to preprocessing;
2. :ref:`pyshqg.core <api.core>` provides a high-level implementation of the QG model;
3. :ref:`pyshqg.backend <api.backend>` provides the interface between the core model and the low-level implementation.

In practice, the core model implementation is highly generic and only depends on a 
so-called backend. The backend should provide support for multidimensional arrays
similar to ``numpy.ndarray`` that can be created and can be converted to ``numpy.ndarray``
as well as support for specific array manipulation functions such as ``einsum()``.
All these functionalities are gathered into a single backend object. For convenience,
we provide the abstract base class :py:class:`pyshqg.backend.abstract.Backend` which
list all the functionalities that should be provided by the backend.

Two backends are already provided:

1. :py:class:`pyshqg.backend.numpy_backend.NumpyBackend` in which backend arrays are simply implemented as ``numpy.ndarray``;
2. :py:class:`pyshqg.backend.tensorflow_backend.TensorflowBackend` in which backend arrays are implemented as ``tensorflow.Tensor``.

More details about ``pyshqg`` are provided in the :ref:`API <api>`.

