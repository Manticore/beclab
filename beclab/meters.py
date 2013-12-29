import numpy

from reikna.cluda import dtypes, functions
from reikna.core import Computation, Parameter, Annotation, Type, Transformation
from reikna.transformations import mul_const, norm_const, add_const
from reikna.algorithms import Reduce, predicate_sum
from reikna.fft import FFT

import beclab.constants as const
from beclab.wavefunction import REPR_CLASSICAL, REPR_WIGNER
from beclab.integrator.helpers import get_ksquared


class _PopulationMeter(Computation):

    def __init__(self, wfs):

        assert wfs.representation in (REPR_CLASSICAL, REPR_WIGNER)

        real_dtype = dtypes.real_for(wfs.dtype)
        pop_arr = Type(real_dtype, (wfs.components, wfs.ensembles))
        Computation.__init__(self, [
            Parameter(
                'populations', Annotation(pop_arr, 'o')),
            Parameter('wfs', Annotation(wfs.data, 'i'))])

        real_arr = Type(real_dtype, wfs.shape)
        self._reduce = Reduce(
            real_arr, predicate_sum(real_dtype),
            axes=list(range(2, len(wfs.shape))))

        norm = norm_const(wfs.data, 2)
        scale = mul_const(real_arr, dtypes.cast(real_dtype)(wfs.grid.dV))
        self._reduce.parameter.input.connect(scale, scale.output, norm=scale.input)
        self._reduce.parameter.norm.connect(norm, norm.output, wfs=norm.input)

        if wfs.representation == REPR_WIGNER:
            add = add_const(pop_arr, -wfs.grid.modes / 2.)
            self._reduce.parameter.output.connect(add, add.input, populations=add.output)

    def _build_plan(self, plan_factory, device_params, populations, wfs):
        plan = plan_factory()
        plan.computation_call(self._reduce, populations, wfs)
        return plan


class PopulationMeter:

    def __init__(self, thr, wfs):
        self._meter = _PopulationMeter(wfs).compile(thr)
        self._out = thr.empty_like(self._meter.parameter.populations)

    def __call__(self, wfs):
        self._meter(self._out, wfs.data)
        return self._out.get()


def get_energy_trf(wfs, kinetic_coeff, states, freqs, scattering):
    real_dtype = dtypes.real_for(wfs.dtype)
    return Transformation(
        [
            Parameter('energy', Annotation(Type(real_dtype, wfs.shape[1:]), 'o')),
            Parameter('data', Annotation(wfs.data, 'i')),
            Parameter('kdata', Annotation(wfs.data, 'i'))],
        """
        <%
            s_ctype = dtypes.ctype(s_dtype)
            r_ctype = dtypes.ctype(r_dtype)
            r_const = lambda x: dtypes.c_constant(x, r_dtype)
        %>
        %for comp in range(components):
        ${kdata.ctype} data_${comp} = ${data.load_idx}(${comp}, ${', '.join(idxs)});
        ${kdata.ctype} kdata_${comp} = ${kdata.load_idx}(${comp}, ${', '.join(idxs)});
        %endfor

        ## FIXME: to be generalized as ScalarField
        %for dim in range(grid.dimensions):
        const ${r_ctype} x_${dim} =
            ${r_const(grid.xs[dim][0])} + ${r_const(grid.dxs[dim])} * ${idxs[dim + 1]};
        %endfor
        const ${r_ctype} V =
            ${r_const(states[comp].m)}
            * (
                %for dim in range(grid.dimensions):
                + ${r_const((2 * numpy.pi * freqs[dim]) ** 2)}
                    * x_${dim} * x_${dim}
                %endfor
            ) / 2;

        %for comp in range(components):
        ${r_ctype} n_${comp} = ${norm}(data_${comp});
        %endfor

        <%
            # FIXME: kinetic and scattering coefficients can be very low and amount to 0
            # in case of single precision.
            # So for the time being we are splitting them in two.

            sign_kcoeff = 1 if kinetic_coeff > 0 else -1
            sqrt_kcoeff = numpy.sqrt(abs(kinetic_coeff))
        %>
        ${r_ctype} E =
            %for comp in range(components):
            + (${sign_kcoeff}) * ${r_const(sqrt_kcoeff)}
                * (${mul_ss}(data_${comp}, kdata_${comp})).x
                * ${r_const(sqrt_kcoeff)}
            + V * n_${comp}
                %for other_comp in range(components):
                <%
                    sqrt_g = numpy.sqrt(scattering[comp, other_comp])
                %>
                + (${r_const(sqrt_g)} * n_${comp})
                    * (${r_const(sqrt_g)} * n_${other_comp}) / 2
                %endfor
            %endfor
            ;

        ${energy.store_same}(E);
        """,
        render_kwds=dict(
            components=wfs.components,
            kinetic_coeff=kinetic_coeff,
            HBAR=const.HBAR,
            states=states,
            grid=wfs.grid,
            s_dtype=wfs.dtype,
            r_dtype=real_dtype,
            freqs=freqs,
            scattering=scattering,
            mul_ss=functions.mul(wfs.dtype, wfs.dtype),
            norm=functions.norm(wfs.dtype),
            ))


def get_ksquared_trf(state_arr, ksquared_arr):
    return Transformation(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('ksquared', Annotation(ksquared_arr, 'i'))],
        """
        ${ksquared.ctype} ksquared = ${ksquared.load_idx}(${', '.join(idxs[2:])});
        ${output.store_same}(${mul}(${input.load_same}, -ksquared));
        """,
        render_kwds=dict(
            mul=functions.mul(
                state_arr.dtype, ksquared_arr.dtype, out_dtype=state_arr.dtype)))


class _EnergyMeter(Computation):

    def __init__(self, wfs, kinetic_coeff, states, freqs, scattering):

        # FIXME: generalize for Wigner?
        assert wfs.representation == REPR_CLASSICAL

        real_dtype = dtypes.real_for(wfs.dtype)
        energy_arr = Type(real_dtype, (wfs.ensembles,))
        Computation.__init__(self, [
            Parameter('energy', Annotation(energy_arr, 'o')),
            Parameter('wfs', Annotation(wfs.data, 'i'))])

        self._ksquared = get_ksquared(wfs.grid.shape, wfs.grid.box).astype(real_dtype)
        ksquared_trf = get_ksquared_trf(wfs.data, self._ksquared)

        self._fft = FFT(wfs.data, axes=range(2, len(wfs.shape)))
        self._fft_with_kprop = FFT(wfs.data, axes=range(2, len(wfs.shape)))
        self._fft_with_kprop.parameter.output.connect(
            ksquared_trf, ksquared_trf.input,
            output_prime=ksquared_trf.output, ksquared=ksquared_trf.ksquared)

        real_arr = Type(real_dtype, wfs.shape[1:])
        self._reduce = Reduce(
            real_arr, predicate_sum(real_dtype),
            axes=list(range(1, len(wfs.shape) - 1)))

        scale = mul_const(real_arr, dtypes.cast(real_dtype)(wfs.grid.dV))
        self._reduce.parameter.input.connect(scale, scale.output, energy=scale.input)

        energy = get_energy_trf(wfs, kinetic_coeff, states, freqs, scattering)
        self._reduce.parameter.energy.connect(energy, energy.energy,
            data=energy.data, kdata=energy.kdata)

    def _build_plan(self, plan_factory, device_params, energy, wfs):
        plan = plan_factory()
        kdata = plan.temp_array_like(wfs)
        ksquared_device = plan.persistent_array(self._ksquared)
        plan.computation_call(self._fft_with_kprop, kdata, ksquared_device, wfs)
        plan.computation_call(self._fft, kdata, kdata, inverse=True)
        plan.computation_call(self._reduce, energy, wfs, kdata)
        return plan


class EnergyMeter:

    def __init__(self, thr, wfs, kinetic_coeff, states, freqs, scattering):
        self._meter = _EnergyMeter(wfs, kinetic_coeff, states, freqs, scattering).compile(thr)
        self._out = thr.empty_like(self._meter.parameter.energy)

    def __call__(self, wfs):
        self._meter(self._out, wfs.data)
        return self._out.get()
