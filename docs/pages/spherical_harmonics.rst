.. _spherical_harmonics:

Spherical harmonics
===================

In ``pyshqg``, the QG model is expressed in spherical harmonics.
In this page, we give a brief technical overview of spherical harmonics.

Any real quare-integrable function :math:`f` on the sphere can be expressed as a series:

.. math::

    f(\theta, \phi) = \sum_{l=0}^{\infty}\sum_{m=-l}{l} f_{l,m}Y_{l,m}(\theta, \phi),

where :math:`\theta` and :math:`\phi` are the latitude and longitude, 
:math:`f_{l,m}` is the spherical harmonics coefficients for degree :math:`l` and 
order :math:`m`, and :math:`Y_{l,m}` is the corresponding spherical harmonics function,
define by

.. math::

   Y_{l,m}(\theta, \phi) = \begin{cases}
   \bar{P}_{l,m}(\mathrm{cos}\theta)\mathrm{cos}m\phi, & \text{if $m\geq0$}, \\
   -\bar{P}_{l,-m}(\mathrm{cos}\theta)\mathrm{sin}m\phi, & \text{if $m<0$}.
   \end{cases}

