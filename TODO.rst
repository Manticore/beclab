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


Tests for thesis
================

- Comparison of integrators: [strong convergence, weak convergence (density, N, some 4-th order correlation), difference(N - exact_N), time] for 4-5 time step sizes. t=1s for better estimations, may test with t=0.1s and 1/10 of time steps first.
- Noise behavior: 1D, Psi(0)=0, see what happens to noise over time
- Visibility for t=1.3 with varying grid/box sizes
