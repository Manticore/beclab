import numpy

import beclab.constants as const
from beclab.integrator import get_padded_ksquared_cutoff, get_ksquared_cutoff_mask


def get_padded_energy_cutoff(grid, component, pad=1):
    ksquared_cutoff = get_padded_ksquared_cutoff(grid.shape, grid.box, pad=pad)
    return ksquared_cutoff * const.HBAR ** 2 / (2 * component.m)


def get_energy_cutoff_mask(grid, component, energy_cutoff=None):
    if energy_cutoff is None:
        ksquared_cutoff = None
    else:
        ksquared_cutoff = energy_cutoff / (const.HBAR ** 2 / (2 * component.m))
    return get_ksquared_cutoff_mask(grid.shape, grid.box, ksquared_cutoff=ksquared_cutoff)

