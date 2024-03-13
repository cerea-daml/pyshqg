.. _model:

Model
=====

The QG model expresses the conservation of potential vorticity on the sphere.

Notations
---------

=================================== =================================
Symbol                              Description
=================================== =================================
:math:`\theta`                      co-latitude
:math:`\phi`                        longitude
:math:`q(\theta, \phi)`             potential vorticity on the sphere
:math:`\psi(\theta, \phi)`          stream function on the sphere
:math:`f'`                          bla bla
:math:`f_0`                         bla bla
:math:`\sigma`                      bla bla
:math:`p`                           bla bla
=================================== =================================

Model equations
---------------

The QG model equation reads

.. math::

    \frac{\partial q}{\partial t} = - \mathrm{J}(\psi, q) - D + S,

where :math:`\mathrm{J}` is the Jacobian:

.. math::

    \mathrm{J}(\psi, q) \triangleq \frac{1}{R^{2}\mathrm{sin}\theta}
    \Bigg[
    \frac{\partial\psi}{\partial\theta}\frac{\partial q}{\partial\phi} 
    - \frac{\partial\psi}{\partial\phi}\frac{\partial q}{\partial\theta}
    \Bigg],

and where :math:`D` and :math:`S` encapsulate the dissipative 
and source processes, respectively. Furthermore, the potential vorticity
:math:`q` and the stream function :math:`\psi` are related by the
following Poisson-like equation:

.. math::

    q = \Delta \psi + f' + \frac{\partial}{\partial p} \bigg( 
    \frac{f_{0}^{2}}{\sigma}\frac{\partial \psi}{\partial p}\bigg).

More details here about this equation + the orography correction

Dissipation processes
^^^^^^^^^^^^^^^^^^^^^

Three dissipation processes are included in the model: (i) thermal
relaxation, (ii) Ekman friction, and (iii) hyper-diffusion. The
diffusion term :math:`D` aggregates the contribution of all three
processes.

Thermal relaxation
******************

Bla bla.

Ekman friction
**************

Bla bla.

Hyper-diffusion
***************

Bla bla.

Source processes
^^^^^^^^^^^^^^^^

Bla bla.

Implementation with spherical harmonics
---------------------------------------

Jacobian term
^^^^^^^^^^^^^

Thermal relaxation term
^^^^^^^^^^^^^^^^^^^^^^^

Bla bla.

Ekman friction term
^^^^^^^^^^^^^^^^^^^

Bla bla.

Hyper-diffusion term
^^^^^^^^^^^^^^^^^^^^

Bla bla.

Source term
^^^^^^^^^^^

Bla bla.

