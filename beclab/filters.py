import numpy

from reikna.cluda import dtypes, functions
from reikna.core import Parameter, Annotation
from reikna.algorithms import PureParallel

from beclab.meters import PopulationMeter


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
            all_indices = ', '.join(idxs)
        %>
        %for comp in range(components):
        ${output.ctype} psi_${comp} = ${input.load_idx}(${comp}, ${all_indices});
        ${output.store_idx}(${comp}, ${all_indices},
            ${mul}(psi_${comp}, ${locals()['coeff' + str(comp)]}));
        %endfor
        """,
        guiding_array=wfs_meta.shape[1:],
        render_kwds=dict(
            components=wfs_meta.components,
            mul=functions.mul(wfs_meta.dtype, real_dtype)))


class NormalizationFilter:

    def __init__(self, wfs_meta, target_Ns):
        self._population = PopulationMeter(wfs_meta)
        self._target_Ns = target_Ns
        self._multiply = get_multiply(wfs_meta).compile(wfs_meta.thread)

    def __call__(self, wfs_data, t):
        Ns = self._population(wfs_data).mean(0)
        coeffs = [
            numpy.sqrt(target_N / N) if N > 0 else 0
            for target_N, N in zip(self._target_Ns, Ns)]
        self._multiply(wfs_data, wfs_data, *coeffs)
