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
