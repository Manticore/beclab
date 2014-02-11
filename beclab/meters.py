from reikna.cluda import dtypes, functions
from reikna.core import Computation, Parameter, Annotation, Type, Transformation
from reikna.transformations import mul_const, norm_const, add_const
from reikna.algorithms import Reduce, predicate_sum
from reikna.fft import FFT

import beclab.constants as const
from beclab.wavefunction import REPR_CLASSICAL, REPR_WIGNER
from beclab.integrator import get_ksquared


class _ReduceNorm(Computation):
    """
    Calculates (abs(psi) ** 2 + modifier).sum(axes) * scale.
    """

    def __init__(self, wfs_meta, axes=None, modifier=0, scale=1):

        real_dtype = dtypes.real_for(wfs_meta.dtype)

        real_arr = Type(real_dtype, wfs_meta.shape)
        self._reduce = Reduce(real_arr, predicate_sum(real_dtype), axes=axes)

        result_arr = self._reduce.parameter.output
        reduce_size = self._reduce.parameter.input.size // self._reduce.parameter.output.size

        norm_trf = norm_const(wfs_meta.data, 2)
        self._reduce.parameter.input.connect(norm_trf, norm_trf.output, wfs_data=norm_trf.input)

        scale_trf = mul_const(result_arr, dtypes.cast(real_dtype)(scale))
        self._reduce.parameter.output.connect(scale_trf, scale_trf.input, scaled=scale_trf.output)

        if modifier != 0:
            scaled_modifier = modifier * reduce_size * scale
            add_trf = add_const(result_arr, scaled_modifier)
            self._reduce.parameter.scaled.connect(add_trf, add_trf.input, result=add_trf.output)

        Computation.__init__(self, [
            Parameter('result', Annotation(result_arr, 'o')),
            Parameter('wfs_data', Annotation(wfs_meta.data, 'i'))])

    def _build_plan(self, plan_factory, device_params, populations, wfs_data):
        plan = plan_factory()
        plan.computation_call(self._reduce, populations, wfs_data)
        return plan


class PopulationMeter:
    """
    Measures the per-trajectory populations.

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    """

    def __init__(self, wfs_meta):
        thread = wfs_meta.thread
        scale = wfs_meta.grid.dV
        if wfs_meta.representation == REPR_WIGNER:
            modifier = -wfs_meta.modes / wfs_meta.grid.V / 2
        else:
            modifier = 0
        self._meter = _ReduceNorm(
            wfs_meta, axes=list(range(2, len(wfs_meta.shape))),
            scale=scale, modifier=modifier).compile(thread)
        self._out = thread.empty_like(self._meter.parameter.result)

    def __call__(self, wfs_data):
        """
        Returns a numpy array with the shape ``(trajectories, components)``
        with component populations.
        """
        self._meter(self._out, wfs_data)
        return self._out.get()


class Density1DMeter:
    """
    Measures the projection of per-component density on a chosen axis
    (in other words, integrates the density over all the other axes).

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    :param axis: the index of an axis to project to.
    """

    def __init__(self, wfs_meta, axis=-1):
        thread = wfs_meta.thread
        scale = wfs_meta.grid.dV

        scale = wfs_meta.grid.dV / wfs_meta.grid.dxs[axis]
        if wfs_meta.representation == REPR_WIGNER:
            modifier = -wfs_meta.modes / wfs_meta.grid.V / 2
        else:
            modifier = 0

        if axis < 0:
            axis = len(wfs_meta.shape) + axis
        else:
            axis = axis + 2
        axes = set(range(2, len(wfs_meta.shape)))
        axes.remove(axis)
        axes = tuple(axes)

        self._meter = _ReduceNorm(
            wfs_meta, axes=axes, scale=scale, modifier=modifier).compile(thread)
        self._out = thread.empty_like(self._meter.parameter.result)

    def __call__(self, wfs_data):
        """
        Returns a numpy array with the shape ``(trajectories, components, size)``
        with the projected density.
        """
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
    r"""
    Measures the energy of a BEC:

    .. math::

        E = \sum_{j=1}^C \int \Psi_j^* \left(
                - \frac{\nabla^2}{2 m} + V_j
                + \sum_{k=1}^C \frac{g_{jk}}{2} \vert \Psi_k \vert^2
            \right) \Psi_j d\mathbf{x}.

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    :param system: the :py:class:`~beclab.System` object the wavefunction corresponds to.
    """

    def __init__(self, wfs_meta, system):
        thread = wfs_meta.thread
        self._meter = _EnergyMeter(wfs_meta, system).compile(thread)
        self._out = thread.empty_like(self._meter.parameter.energy)

    def __call__(self, wfs_data):
        self._meter(self._out, wfs_data)
        return self._out.get()
