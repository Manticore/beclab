BEC-specific wrappers
=====================

.. py:module:: beclab

.. autoclass:: System

.. autofunction:: box_for_tf


Grids
-----

.. autoclass:: UniformGrid
    :members:
    :show-inheritance:

.. autoclass:: beclab.grid.Grid
    :members:


Potential
---------

.. autoclass:: beclab.bec.Potential
    :members:

.. autoclass:: HarmonicPotential
    :show-inheritance:


Cutoffs
-------

.. autoclass:: WavelengthCutoff
    :members:
    :show-inheritance:

.. autoclass:: beclab.cutoff.Cutoff
    :members:


Ground states
-------------

.. autoclass:: ThomasFermiGroundState
    :members: __call__

.. autoclass:: ImaginaryTimeGroundState
    :members: __call__


Integration
-----------

.. autoclass:: Integrator
    :members:


Wavefunctions
-------------

.. autodata:: REPR_CLASSICAL

.. autodata:: REPR_WIGNER

.. autodata:: REPR_POSITIVE_P

.. autoclass:: beclab.wavefunction.WavefunctionSetMetadata

.. autoclass:: WavefunctionSet
    :show-inheritance:


Beam splitter
-------------

.. autoclass:: BeamSplitter
