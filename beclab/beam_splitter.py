import numpy

from reikna.cluda import dtypes, functions
from reikna.core import Computation, Parameter, Annotation, Type
from reikna.pureparallel import PureParallel


def get_splitter(state_arr, comp1, comp2):
    real_dtype = dtypes.real_for(state_arr.dtype)
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('theta', Annotation(Type(real_dtype))),
            Parameter('phi', Annotation(Type(real_dtype)))],
        """
        <%
            all_indices = ', '.join(idxs)
            r_ctype = dtypes.ctype(dtypes.real_for(output.dtype))
        %>
        %for comp in range(components):
        const ${output.ctype} psi_${comp} = ${input.load_idx}(${comp}, ${all_indices});
        %endfor

        ${r_ctype} sin_half_theta = sin(${theta} / 2);
        ${r_ctype} cos_half_theta = cos(${theta} / 2);

        ${output.ctype} minus_i = COMPLEX_CTR(${output.ctype})(0, -1);
        ${output.ctype} k2 = ${mul_sr}(
            ${mul_ss}(minus_i, ${polar_unit}(-${phi})),
            sin_half_theta);
        ${output.ctype} k3 = ${mul_sr}(
            ${mul_ss}(minus_i, ${polar_unit}(${phi})),
            sin_half_theta);

        const ${output.ctype} psi_${comp1}_new =
            ${mul_sr}(psi_${comp1}, cos_half_theta) + ${mul_ss}(psi_${comp2}, k2);
        const ${output.ctype} psi_${comp2}_new =
            ${mul_ss}(psi_${comp1}, k3) + ${mul_sr}(psi_${comp2}, cos_half_theta);

        %for comp in range(components):
        %if comp in (comp1, comp2):
        ${output.store_idx}(${comp}, ${all_indices}, psi_${comp}_new);
        %else:
        ${output.store_idx}(${comp}, ${all_indices}, psi_${comp});
        %endif
        %endfor
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(
            comp1=comp1,
            comp2=comp2,
            components=state_arr.shape[0],
            polar_unit=functions.polar_unit(real_dtype),
            mul_ss=functions.mul(state_arr.dtype, state_arr.dtype),
            mul_sr=functions.mul(state_arr.dtype, real_dtype)))


class BeamSplitterMatrix(Computation):

    def __init__(self, state_arr, comp1, comp2):

        real_dtype = dtypes.real_for(state_arr.dtype)
        Computation.__init__(self,
            [Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('theta', Annotation(Type(real_dtype))),
            Parameter('phi', Annotation(Type(real_dtype)))])

        self._splitter = get_splitter(state_arr, comp1, comp2)

    def _build_plan(self, plan_factory, device_params, output, input_, theta, phi):

        plan = plan_factory()
        plan.computation_call(self._splitter, output, input_, theta, phi)
        return plan


class BeamSplitter:

    def __init__(self, thr, state_arr, comp1, comp2, starting_phase=0, f_detuning=0, f_rabi=0):
        self._starting_phase = starting_phase
        self._detuning = 2 * numpy.pi * f_detuning
        self._f_rabi = f_rabi
        self._splitter = BeamSplitterMatrix(state_arr, comp1, comp2).compile(thr)

    def __call__(self, psi, t, theta):
        phi = t * self._detuning + self._starting_phase
        t_pulse = (theta / numpy.pi / 2.0) / self._f_rabi
        self._splitter(psi, psi, theta, phi)
        return t + t_pulse
