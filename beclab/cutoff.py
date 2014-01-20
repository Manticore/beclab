import numpy

import beclab.constants as const
from beclab.integrator import get_padded_ksquared_cutoff, get_ksquared_cutoff_mask


class WavelengthCutoff:

    def __init__(self, ksquared):
        self.ksquared = ksquared

    @classmethod
    def for_energy(cls, energy, component):
        return cls(energy / (const.HBAR ** 2 / (2 * component.m)))

    @classmethod
    def padded(cls, grid, pad=1):
        ksquared_cutoff = get_padded_ksquared_cutoff(grid.shape, grid.box, pad=pad)
        return cls(ksquared_cutoff)

    def get_mask(self, grid):
        return get_ksquared_cutoff_mask(
            grid.shape, grid.box, ksquared_cutoff=self.ksquared)

    def get_modes_number(self, grid):
        return self.get_mask(grid).sum()
