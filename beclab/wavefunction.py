import numpy

from reikna.cluda import dtypes, functions
from reikna.core import Computation, Parameter, Annotation, Transformation, Type
from reikna.cbrng import CBRNG
from reikna.fft import FFT
from reikna.algorithms import PureParallel
from reikna.transformations import ignore, broadcast_const


#: "Classical" representation (wavefunction)
REPR_CLASSICAL = "classical"
#: Wigner representation
REPR_WIGNER = "Wigner"
#: Positive-P representation
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


def trf_mask(psi_arr, mask_arr):
    return Transformation(
        [Parameter('output', Annotation(psi_arr, 'o')),
        Parameter('input', Annotation(psi_arr, 'i')),
        Parameter('mask', Annotation(mask_arr, 'i'))],
        """
        ${mask.ctype} mask = ${mask.load_idx}(${', '.join(idxs[2:])});

        ${input.ctype} psi;
        if (mask)
        {
            psi = ${input.load_same};
        }
        else
        {
            psi = ${dtypes.c_constant(0, input.dtype)};
        }

        ${output.store_same}(psi);
        """)


class WignerCoherent(Computation):

    def __init__(self, grid, data_out, data_in, cutoff=None, seed=None):
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

        self._cutoff = cutoff
        if cutoff is not None:
            self._cutoff_mask = cutoff.get_mask(grid)
            mask = trf_mask(data_out, self._cutoff_mask)
            self._fft.parameter.input.connect(
                mask, mask.output, unmasked_input=mask.input, mask=mask.mask)

        combine = trf_combine(data_out, data_in)
        self._fft.parameter.output.connect(
            combine, combine.noise,
            psi_c=combine.psi_c, psi_w=combine.psi_w)

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()

        randoms = plan.temp_array_like(output)
        plan.computation_call(self._rng, randoms)

        if self._cutoff is None:
            plan.computation_call(self._fft, output, input_, randoms, inverse=True)
        else:
            mask_device = plan.persistent_array(self._cutoff_mask)
            plan.computation_call(self._fft, output, input_, randoms, mask_device, inverse=True)

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
    """
    Metadata for a wavefunction container object.

    .. py:attribute:: components

        The number of components in this wavefunction.

    .. py:attribute:: cutoff

        A :py:class:`Cutoff` object.

    .. py:attribute:: data

        A Reikna ``Type`` object with the container array metadata.

    .. py:attribute:: dtype

        A ``numpy.dtype`` object of the container data type.

    .. py:attribute:: grid

        A :py:class:`Grid` object.

    .. py:attribute:: representation

        One of :py:data:`REPR_CLASSICAL`, :py:data:`REPR_WIGNER`, :py:data:`REPR_POSITIVE_P`.

    .. py:attribute:: shape

        A tuple with the full shape of the data container.

    .. py:attribute:: thread

        A Reikna ``Thread`` object the wavefunction is connected to.

    .. py:attribute:: trajectories

        The number of independent wavefunctions in this container.

    """

    def __init__(self, thread, dtype, grid,
            components=1, trajectories=1, representation=REPR_CLASSICAL,
            cutoff=None):

        self.thread = thread
        self.grid = grid
        self.components = components
        self.trajectories = trajectories
        self.dtype = dtype
        self.representation = representation
        self.shape = (trajectories, components) + grid.shape
        self.data = Type(dtype, self.shape)
        self.cutoff = cutoff

    @property
    def modes(self):
        if self.cutoff is not None:
            return self.cutoff.get_modes_number(self.grid)
        else:
            return self.grid.size


class WavefunctionSet(WavefunctionSetMetadata):
    """
    Wavefunction container.

    .. py:attribute:: data

        A Reikna ``Array`` object.
    """


    def __init__(self, thread, dtype, grid,
            components=1, trajectories=1, representation=REPR_CLASSICAL,
            cutoff=None):

        WavefunctionSetMetadata.__init__(
            self, thread, dtype, grid,
            components=components, trajectories=trajectories, representation=representation,
            cutoff=cutoff)
        self.data = thread.array(self.shape, dtype)
        self._multiply = get_multiply(self)

    @classmethod
    def for_meta(cls, wfs_meta):
        return cls(
            wfs_meta.thread, wfs_meta.dtype, wfs_meta.grid,
            components=wfs_meta.components,
            trajectories=wfs_meta.trajectories,
            representation=wfs_meta.representation,
            cutoff=wfs_meta.cutoff)

    def fill_with(self, data):
        if isinstance(data, type(self.data)):
            assert data.shape == self.shape
            assert data.dtype == self.dtype
            self.thread.copy_array(data, dest=self.data)
        else:
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
            representation=self.representation, cutoff=self.cutoff)
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
            representation=REPR_WIGNER, cutoff=self.cutoff)
        wcoh = WignerCoherent(self.grid, wf.data, self.data,
            cutoff=self.cutoff, seed=seed).compile(self.thread)
        wcoh(wf.data, self.data)
        return wf

    def to_positivep_coherent(self, trajectories):
        assert self.trajectories == 1
        wf = WavefunctionSet(
            self.thread, self.dtype, self.grid,
            components=self.components, trajectories=trajectories,
            representation=REPR_POSITIVE_P, cutoff=self.cutoff)
        # FIXME: need to copy on the device
        # (will require some support for copy with broadcasting)
        wf.fill_with(self.data.get())
        return wf
