import numpy

import beclab.constants as const


class HarmonicPotential:

    def __init__(self, freqs):
        if not isinstance(freqs, tuple):
            raise NotImplementedError

        self.trap_frequencies = freqs


class System:

    def __init__(self, components, scattering, potential=None):

        self.potential = potential
        self.components = components
        self.scattering = scattering

        self.kinetic_coeff = 1j * const.HBAR / (2 * components[0].m)


    def get_potential(self, grid, comp_num):
        xxs = numpy.meshgrid(*grid.xs, indexing="ij")
        freqs = self.potential.trap_frequencies
        return self.components[comp_num].m / 2 * sum(
            (2 * numpy.pi * f * coords) ** 2 for f, coords in zip(freqs, xxs))


def box_for_tf(system, comp_num, N, border=1.2):
    comp = system.components[comp_num]
    freqs = system.potential.trap_frequencies
    mu = const.mu_tf_3d(freqs, N, comp.m, system.scattering[comp_num, comp_num])
    diameter = lambda f: 2.0 * border * numpy.sqrt(2.0 * mu / (comp.m * (2 * numpy.pi * f) ** 2))
    return tuple(diameter(f) for f in freqs)
