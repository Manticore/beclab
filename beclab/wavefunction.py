import numpy

from reikna.cluda import dtypes, functions
from reikna.core import Computation, Parameter, Annotation, Transformation, Type
from reikna.cbrng import CBRNG
from reikna.fft import FFT
from reikna.algorithms import PureParallel
from reikna.transformations import ignore, broadcast_const


REPR_CLASSICAL = "classical"
REPR_WIGNER = "Wigner"
REPR_POSITIVE_P = "positive-P"


def trf_combine(psi_w_arr, psi_c_arr):
    return Transformation(
        [Parameter('psi_w', Annotation(psi_w_arr, 'o')),
        Parameter('psi_c', Annotation(psi_c_arr, 'i')),
        Parameter('noise', Annotation(psi_w_arr, 'i'))],
        """
        ${psi_c.ctype} psi_c = ${psi_c.load_idx}(0, ${', '.join(idxs[1:])});
        ${noise.ctype} noise = ${noise.load_same};
        ${psi_w.store_same}(psi_c + noise);
        """)


class WignerCoherent(Computation):

    def __init__(self, grid, data_out, data_in, seed=None):
        Computation.__init__(self, [
            Parameter('output', Annotation(data_out, 'o')),
            Parameter('input', Annotation(data_in, 'i'))])

        scale = numpy.sqrt(0.5) / numpy.sqrt(grid.dV / grid.size)

        self._rng = CBRNG.normal_bm(
            data_out, len(data_out.shape),
            seed=seed, sampler_kwds=dict(std=scale))
        zeros = broadcast_const(
            self._rng.parameter.counters,
            numpy.zeros(tuple(), self._rng.parameter.counters.dtype))
        self._rng.parameter.counters.connect(zeros, zeros.output)
        trf_ignore = ignore(self._rng.parameter.counters)
        self._rng.parameter.counters.connect(trf_ignore, trf_ignore.input)

        self._fft = FFT(data_out, axes=range(2, len(data_out.shape)))
        combine = trf_combine(data_out, data_in)
        self._fft.parameter.output.connect(
            combine, combine.noise,
            psi_c=combine.psi_c, psi_w=combine.psi_w)

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()

        randoms = plan.temp_array_like(output)
        plan.computation_call(self._rng, randoms)
        plan.computation_call(self._fft, output, input_, randoms, inverse=True)

        return plan


def get_multiply(wfs_meta):

    real_dtype = dtypes.real_for(wfs_meta.dtype)

    return PureParallel(
        [
            Parameter('output', Annotation(wfs_meta.data, 'o')),
            Parameter('input', Annotation(wfs_meta.data, 'i'))]
            + [Parameter('coeff' + str(i), Annotation(real_dtype))
                for i in range(wfs_meta.components)],
        """
        <%
            trajectory = idxs[0]
            coords = ", ".join(idxs[1:])
        %>
        %for comp in range(components):
        ${output.ctype} psi_${comp} = ${input.load_idx}(${trajectory}, ${comp}, ${coords});
        ${output.store_idx}(
            ${trajectory}, ${comp}, ${coords},
            ${mul}(psi_${comp}, ${locals()['coeff' + str(comp)]}));
        %endfor
        """,
        guiding_array=(wfs_meta.shape[0],) + wfs_meta.shape[1:],
        render_kwds=dict(
            components=wfs_meta.components,
            mul=functions.mul(wfs_meta.dtype, real_dtype)))


class WavefunctionSetMetadata:

    def __init__(self, thread, dtype, grid,
            components=1, trajectories=1, representation=REPR_CLASSICAL,
            energy_cutoff=None):

        self.thread = thread
        self.grid = grid
        self.components = components
        self.trajectories = trajectories
        self.dtype = dtype
        self.representation = representation
        self.shape = (trajectories, components) + grid.shape
        self.data = Type(dtype, self.shape)
        self.energy_cutoff = energy_cutoff


class WavefunctionSet(WavefunctionSetMetadata):

    def __init__(self, thread, dtype, grid,
            components=1, trajectories=1, representation=REPR_CLASSICAL,
            energy_cutoff=None):

        WavefunctionSetMetadata.__init__(
            self, thread, dtype, grid,
            components=components, trajectories=trajectories, representation=representation,
            energy_cutoff=energy_cutoff)
        self.data = thread.array(self.shape, dtype)
        self._multiply = get_multiply(self)

    @classmethod
    def for_meta(cls, wfs_meta):
        return cls(
            wfs_meta.thread, wfs_meta.dtype, wfs_meta.grid,
            components=wfs_meta.components,
            trajectories=wfs_meta.trajectories,
            representation=wfs_meta.representation,
            energy_cutoff=wfs_meta.energy_cutoff)

    def fill_with(self, data):
        data_dims = len(data.shape)
        assert data.shape == self.shape[-data_dims:]
        assert data.dtype == self.dtype
        data = numpy.tile(data, self.shape[:-data_dims] + (1,) * data_dims)
        self.thread.to_device(data, dest=self.data)

    def multiply_by(self, coeffs):
        self._multiply(self.data, self.data, *coeffs)

    def to_trajectories(self, trajectories):
        assert self.trajectories == 1
        wf = WavefunctionSet(
            self.thread, self.dtype, self.grid,
            components=self.components, trajectories=trajectories,
            representation=self.representation, energy_cutoff=self.energy_cutoff)
        # FIXME: need to copy on the device
        # (will require some support for copy with broadcasting)
        wf.fill_with(self.data.get())
        return wf

    def to_wigner_coherent(self, trajectories, seed=None):
        assert self.representation == REPR_CLASSICAL
        assert self.trajectories == 1
        wf = WavefunctionSet(
            self.thread, self.dtype, self.grid,
            components=self.components, trajectories=trajectories,
        wcoh = WignerCoherent(self.grid, wf.data, self.data).compile(self.thread)
            representation=REPR_WIGNER, energy_cutoff=self.energy_cutoff)
        wcoh(wf.data, self.data)
        return wf

    def to_positivep_coherent(self, trajectories):
        assert self.trajectories == 1
        wf = WavefunctionSet(
            self.thread, self.dtype, self.grid,
            components=self.components, trajectories=trajectories,
            representation=REPR_POSITIVE_P, energy_cutoff=self.energy_cutoff)
        # FIXME: need to copy on the device
        # (will require some support for copy with broadcasting)
        wf.fill_with(self.data.get())
        return wf
