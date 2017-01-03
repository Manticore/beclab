import numpy

from reiknacontrib.integrator import Sampler, StopIntegration
from beclab.wavefunction import WavefunctionSet
from beclab.meters import EnergyMeter, DensityIntegralMeter, DensitySliceMeter


class PsiSampler(Sampler):
    """
    Bases: ``reiknacontrib.integrator.Sampler``

    Collects the wavefunction for each trajectory.
    """

    def __init__(self):
        Sampler.__init__(self, no_mean=True, no_stderr=True)

    def __call__(self, wfs_data, t):
        return wfs_data.get()


class PopulationSampler(Sampler):
    """
    Bases: ``reiknacontrib.integrator.Sampler``

    Collects the component populations (both mean and per-trajectory).

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    :param beam_splitter: a :py:class:`~beclab.BeamSplitter` object.
        If given, it will be applied to the wavefunction before measuring populations.
    :param theta: a rotation angle to pass to the beam splitter.
    """

    def __init__(self, wfs_meta, beam_splitter=None, theta=0):
        Sampler.__init__(self)
        self._theta = theta
        self._pmeter = DensityIntegralMeter(wfs_meta, axes=list(range(wfs_meta.grid.dimensions)))

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
    r"""
    Bases: ``reiknacontrib.integrator.Sampler``

    Collects the integral interaction :math:`I = \int \Psi_1^* \Psi_2 d\mathbf{x}`
    (both mean and per-component).

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    """

    def __init__(self, wfs_meta):
        Sampler.__init__(self)
        self._grid = wfs_meta.grid

    def __call__(self, wfs_data, t):
        psi = wfs_data.get()
        dimensions = tuple(range(1, len(psi.shape) - 1))
        return (psi[:,0].conj() * psi[:,1]).sum(dimensions) * self._grid.dV


class VisibilitySampler(Sampler):
    r"""
    Bases: ``reiknacontrib.integrator.Sampler``

    Collects the visibility :math:`V = 2 \int \Psi_1^* \Psi_2 d\mathbf{x} / (N_1 + N_2)`
    (both mean and per-component).

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    """

    def __init__(self, wfs_meta):
        Sampler.__init__(self)
        self._psampler = PopulationSampler(wfs_meta)
        self._isampler = InteractionSampler(wfs_meta)

    def __call__(self, wfs_data, t):
        Ns = self._psampler(wfs_data, t)
        Is = self._isampler(wfs_data, t)
        return 2 * numpy.abs(Is) / Ns.sum(1)


class DensityIntegralSampler(Sampler):
    """
    Bases: ``reiknacontrib.integrator.Sampler``

    Collects the density integrated over given axes.

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    :param axes: indices of axes to integrate over (integrates over all axes if not given).
    :param beam_splitter: a :py:class:`~beclab.BeamSplitter` object.
        If given, it will be applied to the wavefunction before measuring populations.
    :param theta: a rotation angle to pass to the beam splitter.
    :param no_values: if ``True``, no per-trajectory values will be preserved.
    """

    def __init__(self, wfs_meta, axes=None, beam_splitter=None, theta=0, no_values=False):
        Sampler.__init__(self, no_values=no_values)

        self._dmeter = DensityIntegralMeter(wfs_meta, axes=axes)
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


class DensitySliceSampler(Sampler):
    """
    Bases: ``reiknacontrib.integrator.Sampler``

    Collects a density slice with given fixed indices.

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    :param fixed_axes: a dictionary ``{axis: value}`` of fixed indices for the slice.
    :param beam_splitter: a :py:class:`~beclab.BeamSplitter` object.
        If given, it will be applied to the wavefunction before measuring populations.
    :param theta: a rotation angle to pass to the beam splitter.
    :param no_values: if ``True``, no per-trajectory values will be preserved.
    """

    def __init__(self, wfs_meta, fixed_axes={}, beam_splitter=None, theta=0, no_values=False):
        Sampler.__init__(self, no_values=no_values)

        self._dmeter = DensitySliceMeter(wfs_meta, fixed_axes=fixed_axes)
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
    """
    Bases: ``reiknacontrib.integrator.Sampler``

    Collects the total energy of a BEC (see :py:class:`beclab.meters.EnergyMeter` for details).

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    :param system: the :py:class:`~beclab.System` object the wavefunction corresponds to.
    """

    def __init__(self, wfs_meta, system):
        Sampler.__init__(self)
        self._energy = EnergyMeter(wfs_meta, system)

    def __call__(self, wfs_data, t):
        return self._energy(wfs_data)


class StoppingEnergySampler(Sampler):
    """
    Bases: ``reiknacontrib.integrator.Sampler``

    Same as :py:class:`EnergySampler`, but raises ``reiknacontrib.integrator.StopIntegration``
    when the relative difference of the newly collected sample
    and the previous one is less than ``limit``.
    """

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
