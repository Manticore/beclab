import numpy

import beclab.constants as const


def get_TF_state(grid, state, freqs, N):
    if grid.dimensions == 3:
        mu = const.muTF3D(freqs, N, state)
    else:
        raise NotImplementedError()

    a, _ = const.get_interaction_constants(None, state, state)
    g = const.get_scattering_constant(a, state.m)

    xxs = numpy.meshgrid(*grid.xs, indexing="ij")
    V = state.m / 2 * sum(
        (2 * numpy.pi * f * coords) ** 2 for f, coords in zip(freqs, xxs))
    psi_TF = numpy.sqrt((mu - V).clip(0) / g)

    # renormalize to account for coarse grids
    N0 = (numpy.abs(psi_TF) ** 2).sum() * grid.dV
    psi_TF *= numpy.sqrt(N / N0)

    return psi_TF


def to_wigner(grid, rng, psi):
    random_normals = (rng.normal(size=psi.shape) + 1j * rng.normal(size=psi.shape)) / 2
    fft_scale = numpy.sqrt(grid.dV / grid.size)
    return numpy.fft.ifftn(random_normals, axes=range(2, len(psi.shape))) / fft_scale + psi
