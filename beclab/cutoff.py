import beclab.constants as const
from beclab.integrator import get_padded_ksquared_cutoff, get_ksquared_cutoff_mask


class Cutoff:
    """
    The base class for mode cutoffs.
    """

    def get_mask(self, grid):
        """
        Returns a numpy array with the cutoff mask for the given ``grid``.
        Elements of the array are ``0`` in places of the modes that are projected out,
        and ``1`` otherwise.
        """
        raise NotImplementedError

    def get_modes_number(self, grid):
        """
        Returns the total number of active modes for the given ``grid``.
        """
        raise NotImplementedError


class WavelengthCutoff(Cutoff):
    r"""
    Cutoff based on wave lengths in a plane wave basis (essentially, an energy cutoff).
    All the modes with :math:`k^2 > \mathrm{cutoff}` are projected out.
    """

    def __init__(self, ksquared):
        self.ksquared = ksquared

    @classmethod
    def for_energy(cls, energy, component):
        """
        Returns the cutoff expressed in terms of energy for the given component.
        """
        return cls(energy / (const.HBAR ** 2 / (2 * component.m)))

    @classmethod
    def padded(cls, grid, pad=1):
        """
        Returns the cutoff calculated to ensure the ellipse of active modes in a rectangular grid
        is padded.
        ``pad = 1`` means that the ellipse is touching the sides of the box,
        ``pad = 4`` corresponds to the "safe" padding used to avoid aliasing problems during FFT
        transformations.
        """
        ksquared_cutoff = get_padded_ksquared_cutoff(grid.shape, grid.box, pad=pad)
        return cls(ksquared_cutoff)

    def get_mask(self, grid):
        return get_ksquared_cutoff_mask(
            grid.shape, grid.box, ksquared_cutoff=self.ksquared)

    def get_modes_number(self, grid):
        return self.get_mask(grid).sum()
