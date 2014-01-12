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

    def __init__(self, wfs_meta):

        assert wfs_meta.representation in (REPR_CLASSICAL, REPR_WIGNER)

        real_dtype = dtypes.real_for(wfs_meta.dtype)
        pop_arr = Type(real_dtype, (wfs_meta.trajectories, wfs_meta.components))
        Computation.__init__(self, [
            Parameter(
                'populations', Annotation(pop_arr, 'o')),
            Parameter('wfs_data', Annotation(wfs_meta.data, 'i'))])

        real_arr = Type(real_dtype, wfs_meta.shape)
        self._reduce = Reduce(
            real_arr, predicate_sum(real_dtype),
            axes=list(range(2, len(wfs_meta.shape))))

        norm = norm_const(wfs_meta.data, 2)
        scale = mul_const(real_arr, dtypes.cast(real_dtype)(wfs_meta.grid.dV))
        self._reduce.parameter.input.connect(scale, scale.output, norm=scale.input)
        self._reduce.parameter.norm.connect(norm, norm.output, wfs_data=norm.input)

        if wfs_meta.representation == REPR_WIGNER:
            add = add_const(pop_arr, -wfs_meta.grid.modes / 2.)
            self._reduce.parameter.output.connect(add, add.input, populations=add.output)

    def _build_plan(self, plan_factory, device_params, populations, wfs_data):
        plan = plan_factory()
        plan.computation_call(self._reduce, populations, wfs_data)
        return plan


class PopulationMeter:

    def __init__(self, wfs_meta):
        thread = wfs_meta.thread
        self._meter = _PopulationMeter(wfs_meta).compile(thread)
        self._out = thread.empty_like(self._meter.parameter.populations)

    def __call__(self, wfs_data):
        self._meter(self._out, wfs_data)
        return self._out.get()


def get_energy_trf(wfs_meta, system):

    real_dtype = dtypes.real_for(wfs_meta.dtype)
    if system.potential is not None:
        potential = system.potential.get_module(
            wfs_meta.dtype, wfs_meta.grid, system.components)
    else:
        potential = None

    return Transformation(
        [
            Parameter('energy', Annotation(
                Type(real_dtype, (wfs_meta.shape[0],) + wfs_meta.shape[2:]), 'o')),
            Parameter('data', Annotation(wfs_meta.data, 'i')),
            Parameter('kdata', Annotation(wfs_meta.data, 'i'))],
        """
        <%
            s_ctype = dtypes.ctype(s_dtype)
            r_ctype = dtypes.ctype(r_dtype)
            r_const = lambda x: dtypes.c_constant(x, r_dtype)

            trajectory = idxs[0]
            coords = ", ".join(idxs[1:])
        %>
        %for comp in range(components):
        ${kdata.ctype} data_${comp} = ${data.load_idx}(${trajectory}, ${comp}, ${coords});
        ${kdata.ctype} kdata_${comp} = ${kdata.load_idx}(${trajectory}, ${comp}, ${coords});
        %endfor

        %if potential is not None:
        %for comp in range(components):
        const ${r_ctype} V_${comp} = ${potential}${comp}(${coords}, 0);
        %endfor
        %endif

        %for comp in range(components):
        ${r_ctype} n_${comp} = ${norm}(data_${comp});
        %endfor

        ${r_ctype} E =
            %for comp in range(components):
            + (${r_const(system.kinetic_coeff)})
                * (${mul_ss}(data_${comp}, kdata_${comp})).x
            + V_${comp} * n_${comp}
                %for other_comp in range(components):
                <%
                    sqrt_g = numpy.sqrt(system.interactions[comp, other_comp])
                %>
                + (${r_const(system.interactions[comp, other_comp])})
                    * n_${comp} * n_${other_comp} / 2
                %endfor
            %endfor
            ;

        ${energy.store_same}(E);
        """,
        render_kwds=dict(
            dimensions=wfs_meta.grid.dimensions,
            components=wfs_meta.components,
            potential=potential,
            system=system,
            HBAR=const.HBAR,
            s_dtype=wfs_meta.dtype,
            r_dtype=real_dtype,
            mul_ss=functions.mul(wfs_meta.dtype, wfs_meta.dtype),
            norm=functions.norm(wfs_meta.dtype),
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

    def __init__(self, wfs_meta, system):

        # FIXME: generalize for Wigner?
        assert wfs_meta.representation == REPR_CLASSICAL

        real_dtype = dtypes.real_for(wfs_meta.dtype)
        energy_arr = Type(real_dtype, (wfs_meta.trajectories,))
        Computation.__init__(self, [
            Parameter('energy', Annotation(energy_arr, 'o')),
            Parameter('wfs_data', Annotation(wfs_meta.data, 'i'))])

        self._ksquared = get_ksquared(wfs_meta.grid.shape, wfs_meta.grid.box).astype(real_dtype)
        ksquared_trf = get_ksquared_trf(wfs_meta.data, self._ksquared)

        self._fft = FFT(wfs_meta.data, axes=range(2, len(wfs_meta.shape)))
        self._fft_with_kprop = FFT(wfs_meta.data, axes=range(2, len(wfs_meta.shape)))
        self._fft_with_kprop.parameter.output.connect(
            ksquared_trf, ksquared_trf.input,
            output_prime=ksquared_trf.output, ksquared=ksquared_trf.ksquared)

        real_arr = Type(real_dtype, (wfs_meta.shape[0],) + wfs_meta.shape[2:])
        self._reduce = Reduce(
            real_arr, predicate_sum(real_dtype),
            axes=list(range(1, len(wfs_meta.shape) - 1)))

        scale = mul_const(real_arr, dtypes.cast(real_dtype)(wfs_meta.grid.dV))
        self._reduce.parameter.input.connect(scale, scale.output, energy=scale.input)

        energy = get_energy_trf(wfs_meta, system)
        self._reduce.parameter.energy.connect(energy, energy.energy,
            data=energy.data, kdata=energy.kdata)

    def _build_plan(self, plan_factory, device_params, energy, wfs_data):
        plan = plan_factory()
        kdata = plan.temp_array_like(wfs_data)
        ksquared_device = plan.persistent_array(self._ksquared)
        plan.computation_call(self._fft_with_kprop, kdata, ksquared_device, wfs_data)
        plan.computation_call(self._fft, kdata, kdata, inverse=True)
        plan.computation_call(self._reduce, energy, wfs_data, kdata)
        return plan


class EnergyMeter:

    def __init__(self, wfs_meta, system):
        thread = wfs_meta.thread
        self._meter = _EnergyMeter(wfs_meta, system).compile(thread)
        self._out = thread.empty_like(self._meter.parameter.energy)

    def __call__(self, wfs_data):
        self._meter(self._out, wfs_data)
        return self._out.get()
