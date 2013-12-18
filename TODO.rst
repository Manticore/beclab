Features
========

- Drift and diffusion modules for BEC (V, g, losses)
- Ground state calculation
- Meters: energy, total number
- Classical to Wigner transformation
- Beam splitter


Core
====

- Add an integrator that runs single and double steps simultaneously and compares them during each sample.
  Will be required for the ground state calculation.
- The integrator above will be able to adjust the time step after each sample depending on the convergence.
- The integrator above will require some ``linalg.norm`` computation, numpy will be too slow.


Tests
=====

- Run tests for complex64
