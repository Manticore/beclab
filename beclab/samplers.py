import numpy

from beclab.integrator import Sampler
from beclab.wavefunction import REPR_WIGNER


class PsiSampler(Sampler):

    def __init__(self):
        Sampler.__init__(self, no_average=True)

    def __call__(self, psi, t):
        return psi.get().transpose(1, 0, *range(2, len(psi.shape)))


class PopulationSampler(Sampler):

    def __init__(self, wfs, beam_splitter=None, theta=None):
        Sampler.__init__(self)
        self._thr = wfs._thr
        self._wigner = (wfs.representation == REPR_WIGNER)
        self._grid = wfs.grid
        self._beam_splitter = beam_splitter
        if beam_splitter is not None:
            self._psi_temp = self._thr.empty_like(wfs.data)
            self._theta = theta

    def __call__(self, psi, t):
        self._thr.copy_array(psi, dest=self._psi_temp)
        self._beam_splitter(self._psi_temp, t, self._theta)
        psi = self._psi_temp.get()

        # (components, ensembles, x_points)
        density = numpy.abs(psi) ** 2 - (0.5 / self._grid.dV if self._wigner else 0)
        return density.sum((2, 3, 4)) * self._grid.dV


class VisibilitySampler(Sampler):

    def __init__(self, wfs):
        Sampler.__init__(self)
        self._wigner = (wfs.representation == REPR_WIGNER)
        self._grid = wfs.grid

    def __call__(self, psi, t):
        psi = psi.get()

        # (components, ensembles, x_points)
        density = numpy.abs(psi) ** 2 - (0.5 / self._grid.dV if self._wigner else 0)

        N = (density.sum((2, 3, 4)) * self._grid.dV).mean(1)
        I = numpy.abs((psi[0].conj() * psi[1]).sum((1, 2, 3))) * self._grid.dV

        return 2 * I / N.sum()


class Density1DSampler(Sampler):

    def __init__(self, wfs, axis=-1, beam_splitter=None, theta=None):
        Sampler.__init__(self)
        self._thr = wfs._thr
        self._wigner = (wfs.representation == REPR_WIGNER)
        self._grid = wfs.grid
        self._beam_splitter = beam_splitter

        if axis < 0:
            axis = len(wfs.shape) + axis
        else:
            axis = axis + 2
        sum_over = set(range(2, len(wfs.shape)))
        sum_over.remove(axis)
        self._sum_over = tuple(sum_over)

        self._dV = 1.
        for a in self._sum_over:
            self._dV *= self._grid.dxs[a - 2]

        if beam_splitter is not None:
            self._psi_temp = self._thr.empty_like(wfs.data)
            self._theta = theta

    def __call__(self, psi, t):
        self._thr.copy_array(psi, dest=self._psi_temp)
        self._beam_splitter(self._psi_temp, t, self._theta)
        psi = self._psi_temp.get()

        # (components, ensembles, x_points)
        density = numpy.abs(psi) ** 2 - (0.5 / self._grid.dV if self._wigner else 0)
        density_1d = density.sum(self._sum_over) * self._dV

        return density_1d.transpose(1, 0, 2)
