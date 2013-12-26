import numpy

import reikna.cluda.dtypes as dtypes
from reikna.core import Computation, Parameter, Annotation, Transformation
from reikna.cbrng import CBRNG
from reikna.fft import FFT


REPR_CLASSICAL = "classical"
REPR_WIGNER = "Wigner"
REPR_POSITIVE_P = "positive-P"


# FIXME: should be moved to Reikna
# Stub for RNG counters, so that we do not have to keep the whole counter array.
# We will only the counters once anyway.
def trf_zero_stub(arr_t):
    return Transformation(
        [
            Parameter('output', Annotation(arr_t, 'o'))],
        """
        ${output.ctype} val;
        %for n in range(output.dtype.fields['v'][0].shape[0]):
        val.v[${n}] = 0;
        %endfor
        ${output.store_same}(val);
        """)

# FIXME: should be moved to Reikna
# Stub for the resulting counters --- we will just ignore them.
def trf_ignore_stub(arr_t):
    return Transformation(
        [
            Parameter('input', Annotation(arr_t, 'i'))],
        """
        """)


def trf_combine(psi_w_arr, psi_c_arr):
    return Transformation(
        [Parameter('psi_w', Annotation(psi_w_arr, 'o')),
        Parameter('psi_c', Annotation(psi_c_arr, 'i')),
        Parameter('noise', Annotation(psi_w_arr, 'i'))],
        """
        ${psi_c.ctype} psi_c = ${psi_c.load_idx}(${idxs[0]}, 0, ${', '.join(idxs[2:])});
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
        zeros = trf_zero_stub(self._rng.parameter.counters)
        self._rng.parameter.counters.connect(zeros, zeros.output)
        ignore = trf_ignore_stub(self._rng.parameter.counters)
        self._rng.parameter.counters.connect(ignore, ignore.input)

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


def get_multiply(wfs):

    real_dtype = dtypes.real_for(wfs.dtype)

    return PureParallel(
        [
            Parameter('output', Annotation(wfs.data, 'o')),
            Parameter('input', Annotation(wfs.data, 'i'))]
            + [Parameter('coeff' + str(i), Annotation(real_dtype)) for i in range(wfs.components)],
        """
        <%
            all_indices = ', '.join(idxs)
        %>
        %for comp in range(components):
        ${output.ctype} psi_${comp} = ${input.load_idx}(${comp}, ${all_indices});
        ${output.store_idx}(${comp}, ${all_indices},
            ${mul}(psi_${comp}, ${locals()['coeff' + str(comp)]}));
        %endfor
        """,
        guiding_array=wfs.shape[1:],
        render_kwds=dict(
            components=wfs.components,
            coeffs=coeffs,
            mul=functions.mul(wfs.dtype, real_dtype)))


class WavefunctionSet:

    def __init__(self, thr, grid, dtype, components=1, ensembles=1, representation=REPR_CLASSICAL):
        self._thr = thr
        self.grid = grid
        self.components = components
        self.ensembles = ensembles
        self.dtype = dtype
        self.representation = representation
        self.shape = (components, ensembles) + grid.shape
        self.data = thr.array(self.shape, dtype)
        self._multiply = get_multiply(self)

    def fill_with(self, data):
        data_dims = len(data.shape)
        assert data.shape == self.shape[-data_dims:]
        assert data.dtype == self.dtype
        data = numpy.tile(data, self.shape[:-data_dims] + (1,) * data_dims)
        self._thr.to_device(data, dest=self.data)

    def multiply_by(self, coeffs):
        self._multiply(self.data, self.data, *coeffs)

    def to_ensembles(self, ensembles):
        assert self.ensembles == 1
        wf = WavefunctionSet(
            self._thr, self.grid, self.dtype,
            components=self.components, ensembles=ensembles,
            representation=self.representation)
        # FIXME: need to copy on the device
        # (will require some support for copy with broadcasting)
        wf.fill_with(self.data.get())
        return wf

    def to_wigner_coherent(self, ensembles, seed=None):
        assert self.representation == REPR_CLASSICAL
        assert self.ensembles == 1
        wf = WavefunctionSet(
            self._thr, self.grid, self.dtype,
            components=self.components, ensembles=ensembles,
            representation=REPR_WIGNER)
        wcoh = WignerCoherent(self.grid, wf.data, self.data).compile(self._thr)
        wcoh(wf.data, self.data)
        return wf

    def to_positivep_coherent(self, ensembles):
        assert self.ensembles == 1
        wf = WavefunctionSet(
            self._thr, self.grid, self.dtype,
            components=self.components, ensembles=ensembles,
            representation=REPR_POSITIVE_P)
        # FIXME: need to copy on the device
        # (will require some support for copy with broadcasting)
        wf.fill_with(self.data.get())
        return wf

