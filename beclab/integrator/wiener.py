import numpy

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.core import Computation, Parameter, Annotation, Type, Transformation
from reikna.cbrng import CBRNG


def combine(arr_t):
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input1', Annotation(arr_t, 'i')),
        Parameter('input2', Annotation(arr_t, 'i'))],
        """
        ${input1.ctype} dW1 = ${input1.load_same};
        ${input2.ctype} dW2 = ${input2.load_same};
        ${output.store_same}(dW1 + dW2);
        """)


def scale_sqrt_param(arr_t, coeff_dtype):
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i')),
        Parameter('param', Annotation(coeff_dtype))],
        "${output.store_same}(${mul}(${input.load_same}, sqrt(${param})));",
        render_kwds=dict(mul=functions.mul(arr_t.dtype, coeff_dtype, out_dtype=arr_t.dtype)))


class Wiener(Computation):
    """
    Generates differentials of a Wiener process W(x,t) on a regular grid, that is

        W(x,t) = \sum_{n \in B} \phi_n w_n(t),

    where B is the full basis set, \phi_n are basis modes and w_n(t)
    are single-mode Wiener processes.

    The generated noise has the correlation <dW(x, t) dW^*(x', t)> = C delta[x, x'] dt.
    """

    def __init__(self, noise_arr, normalization, seed=None):

        if dtypes.is_complex(noise_arr.dtype):
            real_dtype = dtypes.real_for(noise_arr.dtype)
        else:
            real_dtype = noise_arr.dtype

        self._normalization = normalization
        self._seed = seed

        trf = scale_sqrt_param(noise_arr, real_dtype)
        self._cbrng = CBRNG.normal_bm(
            noise_arr, len(noise_arr.shape),
            sampler_kwds=dict(mean=0, std=numpy.sqrt(normalization)), seed=seed)
        self._cbrng.parameter.randoms.connect(trf, trf.input, dW=trf.output, dt=trf.param)

        Computation.__init__(self, [
            Parameter('state', Annotation(self._cbrng.parameter.counters, 'io')),
            Parameter('dW', Annotation(noise_arr, 'o')),
            Parameter('dt', Annotation(real_dtype))])

    def _build_plan(self, plan_factory, device_params, state, dW, dt):

        plan = plan_factory()
        plan.computation_call(self._cbrng, state, dW, dt)
        return plan

    def double_step(self):
        return WienerDouble(self.parameter.dW, self._normalization, seed=self._seed)

        self.wiener = wiener.compile(thr)
        self.wiener2 = wiener.combine_with()
        print(self.wiener2.signature)


class WienerDouble(Computation):

    def __init__(self, noise_arr, normalization, seed=None):

        self._wiener1 = Wiener(noise_arr, normalization, seed=seed)

        self._wiener2 = Wiener(noise_arr, normalization, seed=seed)
        trf = combine(noise_arr)
        self._wiener2.parameter.dW.connect(trf, trf.input2, dW_combined=trf.output, dW1=trf.input1)

        Computation.__init__(self, list(self._wiener1.signature.parameters.values()))

    def _build_plan(self, plan_factory, device_params, state, dW, dt):

        plan = plan_factory()

        dW1 = plan.temp_array_like(dW)
        plan.computation_call(self._wiener1, state, dW1, dt)
        plan.computation_call(self._wiener2, state, dW, dW1, dt)

        return plan
