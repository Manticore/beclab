import numpy

from beclab.integrator import Sampler, StopIntegration
from beclab.wavefunction import WavefunctionSet
from beclab.meters import EnergyMeter, PopulationMeter, Density1DMeter


class PsiSampler(Sampler):

    def __init__(self):
        Sampler.__init__(self, no_mean=True, no_stderr=True)

    def __call__(self, wfs_data, t):
        return wfs_data.get()


class PopulationSampler(Sampler):

    def __init__(self, wfs_meta, beam_splitter=None, theta=0):
        Sampler.__init__(self)
        self._theta = theta
        self._pmeter = PopulationMeter(wfs_meta)

        self._beam_splitter = beam_splitter
        if beam_splitter is not None:
            self._wfs_temp = WavefunctionSet.for_meta(wfs_meta)

    def __call__(self, wfs_data, t):
        if self._beam_splitter is not None:
            self._wfs_temp.fill_with(wfs_data)
            self._beam_splitter(self._wfs_temp.data, t, self._theta)
            data = self._wfs_temp.data
        else:
            data = wfs_data

        return self._pmeter(data)


class InteractionSampler(Sampler):

    def __init__(self, wfs_meta):
        Sampler.__init__(self)
        self._grid = wfs_meta.grid

    def __call__(self, wfs_data, t):
        psi = wfs_data.get()
        dimensions = tuple(range(1, len(psi.shape) - 1))
        return (psi[:,0].conj() * psi[:,1]).sum(dimensions) * self._grid.dV


class VisibilitySampler(Sampler):

    def __init__(self, wfs_meta):
        Sampler.__init__(self)
        self._psampler = PopulationSampler(wfs_meta)
        self._isampler = InteractionSampler(wfs_meta)

    def __call__(self, wfs_data, t):
        Ns = self._psampler(wfs_data, t)
        Is = self._isampler(wfs_data, t)
        return 2 * numpy.abs(Is) / Ns.sum(1)


class Density1DSampler(Sampler):

    def __init__(self, wfs_meta, axis=-1, beam_splitter=None, theta=0):
        Sampler.__init__(self)
        self._dmeter = Density1DMeter(wfs_meta, axis=axis)
        self._theta = theta

        self._beam_splitter = beam_splitter
        if beam_splitter is not None:
            self._wfs_temp = WavefunctionSet.for_meta(wfs_meta)

    def __call__(self, wfs_data, t):
        if self._beam_splitter is not None:
            self._wfs_temp.fill_with(wfs_data)
            self._beam_splitter(self._wfs_temp.data, t, self._theta)
            data = self._wfs_temp.data
        else:
            data = wfs_data

        return self._dmeter(data)


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
