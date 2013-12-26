import numpy

import beclab.constants as const
from beclab.wavefunction import WavefunctionSet


def get_TF_state(thr, grid, dtype, states, freqs, Ns):
    if grid.dimensions != 3:
        raise NotImplementedError()

    wfs = WavefunctionSet(thr, grid, dtype, components=len(states))
    psi_TF = numpy.empty(wfs.shape, wfs.dtype)
    for i, state in enumerate(states):
        N = Ns[i]

        if N == 0:
            psi_TF[i] = 0
            continue

        mu = const.muTF3D(freqs, N, state)
        a, _ = const.get_interaction_constants(None, state, state)
        g = const.get_scattering_constant(a, state.m)

        xxs = numpy.meshgrid(*grid.xs, indexing="ij")
        V = state.m / 2 * sum(
            (2 * numpy.pi * f * coords) ** 2 for f, coords in zip(freqs, xxs))
        psi_TF[i] = numpy.sqrt((mu - V).clip(0) / g)

        # renormalize to account for coarse grids
        N0 = (numpy.abs(psi_TF) ** 2).sum() * grid.dV
        psi_TF[i] *= numpy.sqrt(N / N0)

    wfs.fill_with(psi_TF)
    return wfs
