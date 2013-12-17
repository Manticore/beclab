import numpy

from reikna.helpers import product
from reikna.cluda import dtypes, Module, Snippet, functions
from reikna.core import Computation, Parameter, Annotation, Transformation, Type

from reikna.fft import FFT
from reikna.pureparallel import PureParallel


def get_ksquared(shape, box):
    ks = [
        2 * numpy.pi * numpy.fft.fftfreq(size, length / size)
        for size, length in zip(shape, box)]

    if len(shape) > 1:
        full_ks = numpy.meshgrid(*ks, indexing='ij')
    else:
        full_ks = ks

    return sum([full_k ** 2 for full_k in full_ks])


def get_nonlinear_wrapper(c_dtype, nonlinear_module, dt):
    s_dtype = dtypes.real_for(c_dtype)
    return Module.create(
        """
        %for comp in (0, 1):
        INLINE WITHIN_KERNEL ${c_ctype} ${prefix}${comp}(
            ${c_ctype} psi0, ${c_ctype} psi1, ${s_ctype} t)
        {
            ${c_ctype} nonlinear = ${nonlinear}${comp}(psi0, psi1, t);
            return ${mul}(
                COMPLEX_CTR(${c_ctype})(0, -${dt}),
                nonlinear);
        }
        %endfor
        """,
        render_kwds=dict(
            c_ctype=dtypes.ctype(c_dtype),
            s_ctype=dtypes.ctype(s_dtype),
            mul=functions.mul(c_dtype, c_dtype),
            dt=dtypes.c_constant(dt, s_dtype),
            nonlinear=nonlinear_module))


def get_nonlinear1(state_arr, real_dtype, nonlinear_module):
    # output = N(input)
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('t', Annotation(real_dtype))],
        """
        <%
            all_indices = ', '.join(idxs)
        %>
        ${output.ctype} psi0 = ${input.load_idx}(0, ${all_indices});
        ${output.ctype} psi1 = ${input.load_idx}(1, ${all_indices});

        ${output.store_idx}(0, ${all_indices}, ${nonlinear}0(psi0, psi1, ${t}));
        ${output.store_idx}(1, ${all_indices}, ${nonlinear}1(psi0, psi1, ${t}));
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(nonlinear=nonlinear_module))


def get_nonlinear2(state_arr, real_dtype, nonlinear_module, dt):
    # k2 = N(psi_I + k1 / 2, t + dt / 2)
    # k3 = N(psi_I + k2 / 2, t + dt / 2)
    # psi_4 = psi_I + k3 (argument for the 4-th step k-propagation)
    # psi_k = psi_I + (k1 + 2(k2 + k3)) / 6 (argument for the final k-propagation)
    return PureParallel(
        [
            Parameter('psi_k', Annotation(state_arr, 'o')),
            Parameter('psi_4', Annotation(state_arr, 'o')),
            Parameter('psi_I', Annotation(state_arr, 'i')),
            Parameter('k1', Annotation(state_arr, 'i')),
            Parameter('t', Annotation(real_dtype))],
        """
        <%
            all_indices = ', '.join(idxs)
        %>

        ${psi_k.ctype} psi_I_0 = ${psi_I.load_idx}(0, ${all_indices});
        ${psi_k.ctype} psi_I_1 = ${psi_I.load_idx}(1, ${all_indices});
        ${psi_k.ctype} k1_0 = ${k1.load_idx}(0, ${all_indices});
        ${psi_k.ctype} k1_1 = ${k1.load_idx}(1, ${all_indices});

        ${psi_k.ctype} k2_0 = ${nonlinear}0(
            psi_I_0 + ${div}(k1_0, 2),
            psi_I_1 + ${div}(k1_1, 2),
            ${t} + ${dt} / 2);
        ${psi_k.ctype} k2_1 = ${nonlinear}1(
            psi_I_0 + ${div}(k1_0, 2),
            psi_I_1 + ${div}(k1_1, 2),
            ${t} + ${dt} / 2);

        ${psi_k.ctype} k3_0 = ${nonlinear}0(
            psi_I_0 + ${div}(k2_0, 2),
            psi_I_1 + ${div}(k2_1, 2),
            ${t} + ${dt} / 2);
        ${psi_k.ctype} k3_1 = ${nonlinear}1(
            psi_I_0 + ${div}(k2_0, 2),
            psi_I_1 + ${div}(k2_1, 2),
            ${t} + ${dt} / 2);

        ${psi_4.store_idx}(0, ${all_indices}, psi_I_0 + k3_0);
        ${psi_4.store_idx}(1, ${all_indices}, psi_I_1 + k3_1);

        ${psi_k.store_idx}(
            0, ${all_indices},
            psi_I_0 + ${div}(k1_0, 6) + ${div}(k2_0, 3) + ${div}(k3_0, 3));
        ${psi_k.store_idx}(
            1, ${all_indices},
            psi_I_1 + ${div}(k1_1, 6) + ${div}(k2_1, 3) + ${div}(k3_1, 3));
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(
            nonlinear=nonlinear_module,
            dt=dtypes.c_constant(dt, real_dtype),
            div=functions.div(state_arr.dtype, numpy.int32, out_dtype=state_arr.dtype)))


def get_nonlinear3(state_arr, real_dtype, nonlinear_module, dt):
    # k4 = N(D(psi_4), t + dt)
    # output = D(psi_k) + k4 / 6
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('kprop_psi_k', Annotation(state_arr, 'i')),
            Parameter('kprop_psi_4', Annotation(state_arr, 'i')),
            Parameter('t', Annotation(real_dtype))],
        """
        <%
            all_indices = ', '.join(idxs)
        %>

        ${output.ctype} psi4_0 = ${kprop_psi_4.load_idx}(0, ${all_indices});
        ${output.ctype} psi4_1 = ${kprop_psi_4.load_idx}(1, ${all_indices});
        ${output.ctype} psik_0 = ${kprop_psi_k.load_idx}(0, ${all_indices});
        ${output.ctype} psik_1 = ${kprop_psi_k.load_idx}(1, ${all_indices});

        ${output.ctype} k4_0 = ${nonlinear}0(psi4_0, psi4_1, ${t} + ${dt});
        ${output.ctype} k4_1 = ${nonlinear}1(psi4_0, psi4_1, ${t} + ${dt});

        ${output.store_idx}(0, ${all_indices}, psik_0 + ${div}(k4_0, 6));
        ${output.store_idx}(1, ${all_indices}, psik_1 + ${div}(k4_1, 6));
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(
            nonlinear=nonlinear_module,
            dt=dtypes.c_constant(dt, real_dtype),
            div=functions.div(state_arr.dtype, numpy.int32, out_dtype=state_arr.dtype)))


class RK4IPStepper(Computation):
    """
    The integration method is RK4IP taken from the thesis by B. Caradoc-Davies
    "Vortex Dynamics in Bose-Einstein Condensates" (2000),
    namely Eqns. B.10 (p. 166).
    """

    def __init__(self, shape, box, drift, ensembles=1, kinetic_coeff=-0.5, diffusion=None):
        #state_arr, dt, box=None, kinetic_coeff=1, nonlinear_module=None):

        real_dtype = dtypes.real_for(drift.dtype)

        if diffusion is not None:
            assert diffusion.dtype == drift.dtype
            assert diffusion.components == drift.components
            self._noise = True
            dW_dtype = real_dtype if diffusion.real_noise else drift.dtype
            dW_arr = Type(dW_dtype, (diffusion.noise_sources, ensembles) + shape)
        else:
            dW_arr = None
            self._noise = False

        state_arr = Type(drift.dtype, (drift.components, ensembles) + shape)

        Computation.__init__(self,
            [Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i'))]
            + ([Parameter('dW', Annotation(dW_arr, 'i'))] if self._noise else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))])

        ksquared = get_ksquared(shape, box)
        self._kprop = numpy.exp(ksquared * (-1j * kinetic_coeff * dt / 2)).astype(state_arr.dtype)
        self._kprop_trf = Transformation(
            [
                Parameter('output', Annotation(state_arr, 'o')),
                Parameter('input', Annotation(state_arr, 'i')),
                Parameter('kprop', Annotation(self._kprop, 'i'))],
            """
            ${kprop.ctype} kprop_coeff = ${kprop.load_idx}(${', '.join(idxs[2:])});
            ${output.store_same}(${mul}(${input.load_same}, kprop_coeff));
            """,
            render_kwds=dict(mul=functions.mul(state_arr.dtype, self._kprop.dtype)))

        self._fft = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop.parameter.output.connect(
            self._kprop_trf, self._kprop_trf.input,
            output_prime=self._kprop_trf.output,
            kprop=self._kprop_trf.kprop)

        nonlinear_wrapper = get_nonlinear_wrapper(state_arr.dtype, nonlinear_module, dt)
        self._N1 = get_nonlinear1(state_arr, real_dtype, nonlinear_wrapper)
        self._N2 = get_nonlinear2(state_arr, real_dtype, nonlinear_wrapper, dt)
        self._N3 = get_nonlinear3(state_arr, real_dtype, nonlinear_wrapper, dt)

    def _add_kprop(self, plan, output, input_, kprop_device):
        temp = plan.temp_array_like(output)
        plan.computation_call(self._fft_with_kprop, temp, kprop_device, input_)
        plan.computation_call(self._fft, output, temp, inverse=True)

    def _build_plan(self, plan_factory, device_params, output, input_, t):

        plan = plan_factory()

        kprop_device = plan.persistent_array(self._kprop)

        # psi_I = D(psi)
        psi_I = plan.temp_array_like(output)
        self._add_kprop(plan, psi_I, input_, kprop_device)

        # k1 = D(N(psi, t))
        k1 = plan.temp_array_like(output)
        temp = plan.temp_array_like(output)
        plan.computation_call(self._N1, temp, input_, t)
        self._add_kprop(plan, k1, temp, kprop_device)

        # k2 = N(psi_I + k1 / 2, t + dt / 2)
        # k3 = N(psi_I + k2 / 2, t + dt / 2)
        # psi_4 = psi_I + k3 (argument for the 4-th step k-propagation)
        # psi_k = psi_I + (k1 + 2(k2 + k3)) / 6 (argument for the final k-propagation)
        psi_4 = plan.temp_array_like(output)
        psi_k = plan.temp_array_like(output)
        plan.computation_call(self._N2, psi_k, psi_4, psi_I, k1, t)

        # k4 = N(D(psi_4), t + dt)
        # output = D(psi_k) + k4 / 6
        kprop_psi_k = plan.temp_array_like(output)
        self._add_kprop(plan, kprop_psi_k, psi_k, kprop_device)
        kprop_psi_4 = plan.temp_array_like(output)
        self._add_kprop(plan, kprop_psi_4, psi_4, kprop_device)
        plan.computation_call(self._N3, output, kprop_psi_k, kprop_psi_4, t)

        return plan
