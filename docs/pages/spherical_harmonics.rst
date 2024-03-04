.. _spherical_harmonics:

Spherical harmonics
===================

In ``pyshqg``, the QG model is expressed in spherical harmonics.
In this page, we give a brief technical overview of spherical harmonics.

Spherical harmonics functions
-----------------------------

Let :math:`P_{l}, l\in\mathbb{N}` be Legendre polynomials, defined by

.. math::

   P_{l}(x) \triangleq \frac{1}{2^{l}l!} \frac{\mathrm{d}^{l}(x^{2}-1)^{l}}{\mathrm{d}x^{l}}.

The associate Legendre functions :math:`\bar{P}_{l,m}, l\in\mathbb{N}, m\in\{0, \ldots, l\}` are defined by

.. math::

   \bar{P}_{l,m}(x) \triangleq (-1)^{m}(1-x^{2})^{m/2}\frac{\mathrm{d}^{m}P_{l}}{\mathrm{d}x^{m}}(x),

and the spherical harmonics functions :math:`Y_{l,m}, l\in\mathbb{N}, m\in\{-l, \ldots, l\}`, defined on the sphere, are given by

.. math::

   Y_{l,m}(\theta, \phi) \triangleq \begin{cases}
   \bar{P}_{l,|m|}(\mathrm{cos}\theta)\mathrm{cos}(|m|\phi), & \text{if $m\geq0$}, \\
   \bar{P}_{l,|m|}(\mathrm{cos}\theta)\mathrm{sin}(|m|\phi), & \text{if $m<0$}.
   \end{cases}

where :math:`\theta` and :math:`\phi` are the co-latitude and longitude, respectively.

The spherical harmonics functions form an orthonormal basis of the real, square-integrable
functions on the sphere, so that any such function :math:`f` can be expressed as the series

.. math::

    f(\theta, \phi) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l} f_{l,m}Y_{l,m}(\theta, \phi),

where the spherical harmonics coefficients :math:`f_{l,m}` are given by

.. math::
   f_{l,m} = \int_{0}^{2\pi}\int_{0}^{\pi} f(\theta, \phi) Y_{l,m}(\theta, \phi) \mathrm{sin}(\theta)\mathrm{d}\theta\mathrm{d}\phi.

Spectral truncation
-------------------

In the QG model, we use a so-called triangular spectral trunctation, in which
all the spectral modes are kept up to a selected spectral degree :math:`T`.
Consequently, the series expansion of :math:`f` becomes

.. math::

    f(\theta, \phi) = \sum_{l=0}^{T}\sum_{m=-l}^{l} f_{l,m}Y_{l,m}(\theta, \phi).

Using this truncature, the number of degrees of freedom is :math:`(T+1)^{2}`.

In practice, the spectral coefficients :math:`f_{l,m}` are usually stored
in an array or tensor with shape :math:`(2, T+1, T+1)`, where the second
dimension represents the spectral degree :math:`l`, the third dimension
represents the absolute value of the spectral mode :math:`m`, and the first
dimension represents the sign of the spectral mode :math:`m`. This
representation is suboptimal in terms of memory (because of the constraint
:math:`|m|\leq l`, half of the array or tensor is unused) but turns out to
be well suited for spectral transformations.

Quadrature
----------

In order to numerically evaluate the spectral coefficients :math:`f_{l,m}`,
the most convenient approach is to use a predefined quadrature, where the
integrand is evaluated at a given number of points on the sphere.

In the QG model, we choose to use the Gauss--Legendre quadrature, in which
integrands are evaluated on a rectangular grid with :math:`N_{\mathsf{lat}}`
co-latitudes :math:`\theta_{i}, i\in\{1, \ldots, N_{\mathsf{lat}}\}` and 
:math:`N_{\mathsf{lon}}` longitudes :math:`\phi_{j}, j\in\{1, \ldots, N_{\mathsf{lon}}\}`.
The co-latitudes :math:`\theta_{i}` are chosen as the zeros of the Legendre
Polynomial :math:`P_{N_{\mathsf{lat}}}`, while the longitudes are evenly
sampled over :math:`[0, 2\pi]`. Using a rectangular quadrature means that 
polar areas are over-sampled, which is compensated by weighting the grid
nodes with a factor :math:`w_{i}` that depends only on latitude, called Gauss--Legendre
weight. 

As long as the number of co-latitudes :math:`N_{\mathsf{lat}}` is larger
than :math:`T+1` and the number of longitudes :math:`N_{\mathsf{lon}}` is larger
than :math:`2T+1`, the Gauss--Legendre quadrature is exact.
In particular, the spectral coefficients :math:`f_{l,m}` are then given by

.. math::

    f_{l,m} = \sum_{i=1}^{N_{\mathsf{lat}}} \sum_{j=1}^{N_{\mathsf{lon}}}
    f(\theta_{i}, \phi_{j}) \bar{P}_{l, m}(\mathrm{cos}\theta)\mathrm{cos}(m\phi_{j}) w_{i}

for :math:`m\geq0` and

.. math::

    f_{l,m} = - \sum_{i=1}^{N_{\mathsf{lat}}} \sum_{j=1}^{N_{\mathsf{lon}}}
    f(\theta_{i}, \phi_{j}) \bar{P}_{l, m}(\mathrm{cos}\theta)\mathrm{sin}(m\phi_{j}) w_{i}

for :math:`m<0`.

For simplicity, in the QG model we always choose to use 
:math:`N_{\mathsf{lon}}=2N_{\mathsf{lat}}`, but this constraint is not
mandatory. In practice, the values of the function :math:`f` are stored
in an array or tensor with shape :math:`(N_{\mathsf{lat}}, N_{\mathsf{lon}})`.
Note that, even in the case where :math:`N_{\mathsf{lat}}=T+1` and
:math:`N_{\mathsf{lon}}=2T+1` (which is the smallest possible grid for a given
truncature :math:`T`), the number of degrees of freedom in this grid space
is :math:`(T+1)(2T+1)`, which is almost double the number of degrees
of freedom in spectral space, :math:`(T+1)^{2}`.


Spectral transformation
-----------------------

The first sum, over :math:`i`, is usually referred to as Legendre transformation,
and is typically computed using functions such as ``numpy.einsum``. Note that
in that operation, the :math:`\bar{P}_{l, m}(\mathrm{cos}\theta)w_{i}` factors
do not depend on :math:`f` and hence can be precomputed once the grid is fixed.

The second sum, over :math:`j`, is usually referred to as Fourier transformation,
and can be computed using specific implementations of the discrete Fourier
transform since it has been assumed that the longitudes are evenly distributed.

More bla bla to come

Longitudinal derivative
-----------------------

More bla bla to come

Latitudinal derivative
----------------------

More bla bla to come

