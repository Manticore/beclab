Core
====

- Steppers API needs to be updated to support per-step error estimation (see Numeric Recipes)
- If we really need samplers that return several values, we can declare them as
  samplers=dict{('N_mean', 'N_err'): n_sampler} and split the return value of __call__.
  For the time being the resulting list will just be the minor dimension of the full resulting array.
- How do we specify stateful samplers in XMDS-like interface? Pass them the list of results of their previous invocations?
- Need to display the full errors in the end of adaptive step integration.
  Sum errors over sample steps (don't forget they're relative)?
- Kinetic coefficient parameter for steppers should support per-component coefficients


Examples and tests
==================

- Application to experiments: Swin's (Ramsey, spin echo), Riedel (spatial separation), Widera (1D)
- Noise balance test: check that if we start with 0 atoms, we will still have 0 atoms even with enabled losses
  (try different loss sources, different cutoffs etc)
- Correlations (atom bunching): check that <Psi1^+ Psi1^+ Psi1 Psi1> / <Psi1+ Psi1>^2 ~ 1 in condensate.
  May also check <Psi1^+ Psi2^+ Psi1 Psi2> / (<Psi1+ Psi1> <Psi2+ Psi2>) (should be ~1 too in condensate).
- Check correlations for thermal states (should be >1). See if we can reach theoretical predictions (3! for 3-body, 2! for 2-body)
- Gradual decrease of cutoff (results should not diverge even when we get down to 2-mode)
- Compare uniform/harmonic grid
- chapters/wigner-bec/fpe-bec: it should be possible to proof that the formula for d<Psi+ Psi>/dt agrees with the classical one up to the $1/N$ order, same as we do for particular cases later on.

