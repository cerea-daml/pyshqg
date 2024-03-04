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

    f(\theta, \phi) = \sum_{l=0}^{\infty}\sum_{m=-l}{l} f_{l,m}Y_{l,m}(\theta, \phi),

where the spherical harmonics coefficients :math:`f_{l,m}` are given by

.. math::
   f_{l,m} = \int_{0}^{2\pi}\int_{0}^{\pi} f(\theta, \phi) Y_{l,m}(\theta, \phi) \mathrm{sin}(\theta}\mathrm{d}\theta\mathrm{d}\phi.

Spectral truncation
-------------------

In the QG model, we use a so-called triangular spectral trunctation, in which
all the spectral modes are kept up to a selected spectral degree :math:`T`.
Consequently, the series expansion of :math:`f` becomes

.. math::

    f(\theta, \phi) = \sum_{l=0}^{T}\sum_{m=-l}{l} f_{l,m}Y_{l,m}(\theta, \phi).

Using this truncature, the number of degrees of freedom is :math:`(T+1)^{2}`.

Quadrature
----------

In order to numerically evaluate the spectral coefficients :math:`f_{l,m}`,
the most convenient approach is to use a predefined quadrature, where the
integrand is evaluated at a given number of points on the sphere.

In the QG model, we choose to use the Gauss--Legendre quadrature.
