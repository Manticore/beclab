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


def get_xpropagate(state_arr, drift, diffusion=None, dW_arr=None):

    real_dtype = dtypes.real_for(state_arr.dtype)
    if diffusion is not None:
        noise_dtype = dW_arr.dtype
    else:
        noise_dtype = real_dtype

    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('omega', Annotation(state_arr, 'io')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('kinput', Annotation(state_arr, 'i')),
            Parameter('ai', Annotation(real_dtype)),
            Parameter('bi', Annotation(real_dtype)),
            Parameter('ci', Annotation(real_dtype)),
            Parameter('stage', Annotation(numpy.int32))]
            + ([Parameter('dW', Annotation(dW_arr, 'i'))] if diffusion is not None else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))],
        """
        <%
            all_indices = ', '.join(idxs)
            components = drift.components
            if diffusion is not None:
                noise_sources = diffusion.noise_sources
            idx_args = ", ".join(idxs[1:])
            psi_args = ", ".join("psi_" + str(c) for c in range(components))

            if diffusion is None:
                dW = None
        %>

        %for comp in range(components):
        ${output.ctype} omega_${comp};
        if (${stage} == 0)
        {
            omega_${comp} = ${dtypes.c_constant(0, output.dtype)};
        }
        else
        {
            omega_${comp} = ${omega.load_idx}(${comp}, ${all_indices});
        }
        ${output.ctype} psi_${comp} = ${input.load_idx}(${comp}, ${all_indices});
        ${output.ctype} kpsi_${comp} = ${kinput.load_idx}(${comp}, ${all_indices});
        ${output.ctype} dpsi_${comp};
        %endfor

        %if diffusion is not None:
        %for ncomp in range(noise_sources):
        ${dW.ctype} dW_${ncomp} = ${dW.load_idx}(${ncomp}, ${all_indices});
        %endfor
        %endif

        %for comp in range(components):
        dpsi_${comp} =
            ${mul_cr}(
                kpsi_${comp}
                + ${drift.module}${comp}(${idx_args}, ${psi_args}, ${t} + ${dt} * ${ci}),
                ${dt})
            %if diffusion is not None:
            %for ncomp in range(noise_sources):
            + ${mul_cn}(${diffusion.module}${comp}_${ncomp}(
                ${idx_args}, ${psi_args}, ${t} + ${dt} * ${ci}), dW_${ncomp})
            %endfor
            %endif
            ;
        %endfor

        ${output.ctype} new_omega, new_u;
        %for comp in range(components):
        new_omega = ${mul_cr}(omega_${comp}, ${ai}) + dpsi_${comp};
        new_u = psi_${comp} + ${mul_cr}(new_omega, ${bi});
        if (${stage} < 5)
        {
            ${omega.store_idx}(${comp}, ${all_indices}, new_omega);
        }
        ${output.store_idx}(${comp}, ${all_indices}, new_u);
        %endfor
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(
            drift=drift,
            diffusion=diffusion,
            mul_cr=functions.mul(state_arr.dtype, real_dtype),
            mul_cn=functions.mul(state_arr.dtype, noise_dtype)))


class RK46NLStepper(Computation):
    """
    6-step 4th order RK optimized for minimum dissipation and minimum temporary space.
    """

    def __init__(self, shape, box, drift, ensembles=1, kinetic_coeff=0.5, diffusion=None):

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
        self._kprop = (-ksquared * kinetic_coeff).astype(real_dtype)
        kprop_trf = Transformation(
            [
                Parameter('output', Annotation(state_arr, 'o')),
                Parameter('input', Annotation(state_arr, 'i')),
                Parameter('kprop', Annotation(self._kprop, 'i'))],
            """
            ${kprop.ctype} kprop = ${kprop.load_idx}(${', '.join(idxs[2:])});
            ${output.ctype} kprop_coeff = COMPLEX_CTR(${output.ctype})(0, kprop);
            ${output.store_same}(${mul}(${input.load_same}, kprop_coeff));
            """,
            render_kwds=dict(
                mul=functions.mul(state_arr.dtype, state_arr.dtype),
                ))

        self._fft = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop.parameter.output.connect(
            kprop_trf, kprop_trf.input,
            output_prime=kprop_trf.output, kprop=kprop_trf.kprop)

        self._xpropagate = get_xpropagate(state_arr, drift, diffusion=diffusion, dW_arr=dW_arr)

        self._ai = numpy.array([
            0.0, -0.737101392796, -1.634740794341,
            -0.744739003780, -1.469897351522, -2.813971388035])
        self._bi = numpy.array([
            0.032918605146, 0.823256998200, 0.381530948900,
            0.200092213184, 1.718581042715, 0.27])
        self._ci = numpy.array([
            0.0, 0.032918605146, 0.249351723343,
            0.466911705055, 0.582030414044, 0.847252983783])

    def _build_plan(self, plan_factory, device_params, *args):

        if self._noise:
            output, input_, dW, t, dt = args
            dt_args = dW, t, dt
        else:
            output, input_, t, dt = args
            dt_args = t, dt

        plan = plan_factory()

        kprop_device = plan.persistent_array(self._kprop)
        kdata = plan.temp_array_like(output)
        omega = plan.temp_array_like(output)
        result = plan.temp_array_like(output)

        data_out = input_

        for stage in range(6):

            data_in = data_out
            if stage == 5:
                data_out = output
            else:
                data_out = result

            plan.computation_call(self._fft_with_kprop, kdata, kprop_device, data_in)
            plan.computation_call(self._fft, kdata, kdata, inverse=True)

            plan.computation_call(
                self._xpropagate, data_out, omega, data_in, kdata,
                self._ai[stage], self._bi[stage], self._ci[stage], stage, *dt_args)

        return plan
