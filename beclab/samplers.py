import numpy

from beclab.integrator import Sampler, StopIntegration
from beclab.wavefunction import REPR_WIGNER
from beclab.meters import EnergyMeter


class PsiSampler(Sampler):

    def __init__(self):
        Sampler.__init__(self, no_mean=True, no_stderr=True)

    def __call__(self, wfs_data, t):
        return wfs_data.get()


class PopulationSampler(Sampler):

    def __init__(self, wfs_meta, beam_splitter=None, theta=0):
        Sampler.__init__(self)
        self._thr = wfs_meta.thread
        self._wigner = (wfs_meta.representation == REPR_WIGNER)
        self._grid = wfs_meta.grid
        self._theta = theta

        self._beam_splitter = beam_splitter
        if beam_splitter is not None:
            self._psi_temp = self._thr.empty_like(wfs_meta.data)

    def __call__(self, wfs_data, t):
        if self._beam_splitter is not None:
            self._thr.copy_array(wfs_data, dest=self._psi_temp)
            self._beam_splitter(self._psi_temp, t, self._theta)
            psi = self._psi_temp.get()
        else:
            psi = wfs_data.get()

        density = numpy.abs(psi) ** 2 - (0.5 / self._grid.dV if self._wigner else 0)
        return density.sum((2, 3, 4)) * self._grid.dV


class InteractionSampler(Sampler):

    def __init__(self, wfs):
        Sampler.__init__(self)
        self._wigner = (wfs.representation == REPR_WIGNER)
        self._grid = wfs.grid

    def __call__(self, psi, t):
        psi = psi.get()
        return (psi[:,0].conj() * psi[:,1]).sum((1, 2, 3)) * self._grid.dV


class VisibilitySampler(Sampler):

    def __init__(self, wfs):
        Sampler.__init__(self)
        self._wigner = (wfs.representation == REPR_WIGNER)
        self._grid = wfs.grid

    def __call__(self, psi, t):
        psi = psi.get()

        density = numpy.abs(psi) ** 2 - (0.5 / self._grid.dV if self._wigner else 0)

        N = (density.sum((2, 3, 4)) * self._grid.dV).mean(0)
        I = numpy.abs((psi[:,0].conj() * psi[:,1]).sum((1, 2, 3))) * self._grid.dV

        return 2 * I / N.sum()


class Density1DSampler(Sampler):

    def __init__(self, wfs_meta, axis=-1, beam_splitter=None, theta=0):
        Sampler.__init__(self)
        self._thr = wfs_meta.thread
        self._wigner = (wfs_meta.representation == REPR_WIGNER)
        self._grid = wfs_meta.grid
        self._beam_splitter = beam_splitter

        self._part_dV = self._grid.dV / self._grid.dxs[axis]

        if axis < 0:
            axis = len(wfs_meta.shape) + axis
        else:
            axis = axis + 2
        sum_over = set(range(2, len(wfs_meta.shape)))
        sum_over.remove(axis)
        self._sum_over = tuple(sum_over)

        self._beam_splitter = beam_splitter
        if beam_splitter is not None:
            self._psi_temp = self._thr.empty_like(wfs_meta.data)
            self._theta = theta

    def __call__(self, psi, t):
        if self._beam_splitter is not None:
            self._thr.copy_array(psi, dest=self._psi_temp)
            self._beam_splitter(self._psi_temp, t, self._theta)
            psi = self._psi_temp.get()
        else:
            psi = psi.get()

        # (components, trajectories, x_points)
        density = numpy.abs(psi) ** 2 - (0.5 / self._grid.dV if self._wigner else 0)
        density_1d = density.sum(self._sum_over) * self._part_dV

        return density_1d


class EnergySampler(Sampler):

    def __init__(self, wfs_meta, system):
        Sampler.__init__(self)
        self._energy = EnergyMeter(wfs_meta, system)

    def __call__(self, wfs_data, t):
        return self._energy(wfs_data)


class StoppingEnergySampler(Sampler):

    def __init__(self, wfs, system, limit=1e-6):
        Sampler.__init__(self)
        self._energy = EnergyMeter(wfs, system)
        self._previous_E = None
        self._limit = limit

    def __call__(self, psi, t):
        E = self._energy(psi)

        if self._previous_E is not None:
            if abs(E[0] - self._previous_E[0]) / abs(E[0]) < self._limit:
                raise StopIntegration(E)

        self._previous_E = E
        return E
